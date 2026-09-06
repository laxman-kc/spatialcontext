"""One-GPU LoRA loop with complete checkpoints at optimizer boundaries."""
import json
import math
import random
import time
from pathlib import Path

from .artifacts import atomic_json, digest, file_digest


def verify_checkpoint(path):
    path = Path(path)
    metadata = json.loads((path / "checkpoint.json").read_text())
    if not metadata.get("files"):
        raise ValueError("Checkpoint has no integrity manifest")
    for name, expected in metadata["files"].items():
        target = (path / name).resolve()
        if not target.is_relative_to(path.resolve()) or file_digest(target) != expected:
            raise ValueError("Checkpoint content integrity failure")
    return metadata


def train(runtime, records, config, output_dir, *, protocol_digest, resume=None, diagnostic=False,
          stop_after_updates=None):
    import numpy as np
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import get_cosine_schedule_with_warmup

    if not records or any(r["split"] != "train" for r in records):
        raise ValueError("Training accepts a nonempty training split only")
    if not diagnostic and any(r["audit"]["status"] != "approved" for r in records):
        raise ValueError("Unapproved examples cannot enter research training")
    if stop_after_updates is not None and (not diagnostic or stop_after_updates < 1):
        raise ValueError("Controlled early stop is only available for explicit diagnostics")
    settings = config["training"]
    if settings["epochs"] != 1:
        raise ValueError("This pilot implements exactly one epoch")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed = settings["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = runtime.add_lora()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable adapters")
    optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"], betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=settings["weight_decay"])
    accumulation = settings["gradient_accumulation_steps"]
    updates = math.ceil(len(records) / accumulation)
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(updates * settings["warmup_ratio"]), updates)
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    start_index, step, prior_seconds = 0, 0, 0.0
    identity = digest({"protocol": protocol_digest, "records": records, "settings": settings, "diagnostic": diagnostic})
    for existing in out.glob("checkpoint-*"):
        metadata = verify_checkpoint(existing)
        if metadata.get("identity") != identity:
            raise ValueError("Output directory contains checkpoints from another run")
    if (out / "summary.json").exists():
        if json.loads((out / "summary.json").read_text()).get("identity") != identity:
            raise ValueError("Output directory belongs to another run")
    if resume:
        checkpoint = Path(resume)
        verify_checkpoint(checkpoint)
        state = torch.load(checkpoint / "training.pt", map_location="cpu", weights_only=False)
        if state["identity"] != identity:
            raise ValueError("Resume checkpoint belongs to a different run")
        weights = load_file(str(checkpoint / "adapter_model.safetensors"))
        expected = get_peft_model_state_dict(model)
        if weights.keys() != expected.keys() or any(weights[k].shape != expected[k].shape for k in weights):
            raise ValueError("Checkpoint adapter tensors do not match this model")
        set_peft_model_state_dict(model, weights)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        order, start_index, step = state["order"], state["next_index"], state["step"]
        prior_seconds = state["elapsed_seconds"]
    elif (out / "summary.json").exists() or list(out.glob("checkpoint-*")):
        raise ValueError("Output already contains training; use a fresh directory or explicit resume")
    started = time.monotonic()
    last_checkpoint = None

    def elapsed():
        return prior_seconds + time.monotonic() - started

    def save_checkpoint(next_index):
        nonlocal last_checkpoint
        target = out / f"checkpoint-{step:06d}"
        if target.exists():
            metadata = verify_checkpoint(target)
            if metadata.get("identity") != identity or metadata.get("next_index") != next_index:
                raise ValueError("Checkpoint collision with inconsistent identity or data position")
            last_checkpoint = target
            return
        temp = out / f".checkpoint-{step:06d}.tmp"
        temp.mkdir(exist_ok=True)
        model.save_pretrained(temp, safe_serialization=True)
        runtime.processor.save_pretrained(temp / "processor")
        torch.save({"identity": identity, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                    "order": order, "next_index": next_index, "step": step, "elapsed_seconds": elapsed()},
                   temp / "training.pt")
        files = {str(p.relative_to(temp)): file_digest(p) for p in sorted(temp.rglob("*"))
                 if p.is_file() and p.name != "checkpoint.json"}
        atomic_json(temp / "checkpoint.json", {"identity": identity, "protocol_digest": protocol_digest,
                    "step": step, "next_index": next_index, "diagnostic": diagnostic, "config": config,
                    "files": files})
        temp.rename(target)
        last_checkpoint = target

    next_index = start_index
    model.train()
    torch.cuda.reset_peak_memory_stats()
    for offset in range(start_index, len(order), accumulation):
        if stop_after_updates is not None and step >= stop_after_updates:
            break
        if elapsed() >= settings["max_cumulative_seconds"]:
            break
        chunk = order[offset:offset + accumulation]
        optimizer.zero_grad(set_to_none=True)
        losses, token_counts = [], []
        for index in chunk:
            record = records[index]
            encoded = runtime.encode(record, "original", answer=record["answer"])
            token_counts.append(int(encoded["input_ids"].shape[-1]))
            inputs = runtime.move_inputs(encoded)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**inputs).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            losses.append(float(loss.detach()))
            (loss / len(chunk)).backward()
            del inputs, encoded, loss
        grad = torch.nn.utils.clip_grad_norm_(parameters, settings["max_grad_norm"])
        if not torch.isfinite(grad):
            raise FloatingPointError("Nonfinite gradient norm")
        lr_used = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        step += 1
        next_index = offset + len(chunk)
        event = {"step": step, "examples_seen": next_index, "mean_loss": sum(losses)/len(losses),
                 "gradient_norm": float(grad), "lr_used": lr_used,
                 "lr": scheduler.get_last_lr()[0], "tokens": token_counts,
                 "elapsed_seconds": elapsed(), "peak_vram_bytes": torch.cuda.max_memory_allocated()}
        with (out / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(event, allow_nan=False) + "\n")
            handle.flush()
        print(json.dumps(event), flush=True)
        if step % settings["checkpoint_every_optimizer_steps"] == 0:
            save_checkpoint(next_index)
    save_checkpoint(next_index)
    summary = {"identity": identity, "protocol_digest": protocol_digest, "diagnostic": diagnostic,
               "updates": step, "examples_seen": next_index, "examples_total": len(records),
               "stop_reason": ("epoch_complete" if next_index == len(records) else
                               "diagnostic_stop" if stop_after_updates is not None and step >= stop_after_updates
                               else "time_cap"),
               "elapsed_seconds": elapsed(), "checkpoint": last_checkpoint.name,
               "trainable_parameters": sum(p.numel() for p in parameters),
               "peak_vram_bytes": torch.cuda.max_memory_allocated()}
    atomic_json(out / "summary.json", summary)
    return summary
