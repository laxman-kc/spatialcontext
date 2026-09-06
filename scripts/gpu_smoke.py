"""Run explicit real-model smoke checks; invoke only on the coordinated GPU worker.

This writes a disposable adapter and JSON evidence, never experiment predictions.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path


def main():
    import torch
    import torch.nn.functional as functional
    import yaml

    from ecqa.model import CHOICES, ModelRuntime

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    records = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    record = next(row for row in records if row.get("answer") in CHOICES)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    runtime = ModelRuntime(config, args.data_root)
    evidence = {"id": record["id"], "device": str(runtime.device), "checks": {},
                "purpose": "single-record model plumbing diagnostic, not an accuracy evaluation",
                "runtime_profile": config["model"], "audit_status": record.get("audit", {}).get("status"),
                "tolerances": {"score_atol": 0.03, "same_argmax_required": True,
                               "loss_atol": 1e-5, "loss_rtol": 1e-5}}

    def compare(first, second, tolerance=0.03):
        difference = max(abs(first[label] - second[label]) for label in CHOICES)
        if difference > tolerance:
            raise AssertionError(f"Candidate log probabilities differ by {difference}, tolerance={tolerance}")
        if max(first, key=first.get) != max(second, key=second.get):
            raise AssertionError("Candidate rankings disagree")
        return difference

    base = runtime.scores(record)
    reference = runtime.scores(record, reference=True)
    evidence["base_scores"] = base
    evidence["checks"]["reference_max_difference"] = compare(base, reference)
    evidence["text_only_scores"] = runtime.scores(record, "text_only")
    encoded = runtime.encode(record, answer=record["answer"])
    supervised = encoded["labels"][encoded["labels"] != -100].tolist()
    evidence["supervised_token_ids"] = supervised
    evidence["supervised_text"] = runtime.processor.tokenizer.decode(supervised, skip_special_tokens=False)
    evidence["input_length"] = encoded["input_ids"].shape[1]
    evidence["video_grid_thw"] = encoded["video_grid_thw"].tolist()
    inputs = runtime.move_inputs(encoded)
    with torch.no_grad():
        result = runtime.model(**inputs, use_cache=False)
        manual = functional.cross_entropy(
            result.logits[:, :-1].float().reshape(-1, result.logits.shape[-1]),
            inputs["labels"][:, 1:].reshape(-1), ignore_index=-100,
        )
        if not torch.isfinite(result.loss) or not torch.allclose(result.loss, manual, atol=1e-5, rtol=1e-5):
            raise AssertionError("Built-in loss disagrees with manual causal cross entropy")
        evidence["checks"]["causal_loss"] = float(result.loss.item())
    del result, manual

    runtime.add_lora()
    evidence["checks"]["zero_adapter_max_difference"] = compare(base, runtime.scores(record))
    trainable = [(name, value) for name, value in runtime.model.named_parameters() if value.requires_grad]
    before = {name: value.detach().cpu().clone() for name, value in trainable}
    frozen = [(name, value) for name, value in runtime.model.named_parameters() if not value.requires_grad]
    representatives = [frozen[0], next(item for item in frozen if ".base_layer.weight" in item[0])]
    frozen_before = {name: value.detach().clone() for name, value in representatives}
    runtime.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    runtime.model.train()
    optimizer = torch.optim.AdamW([value for _, value in trainable], lr=1e-5)
    result = runtime.model(**inputs, use_cache=False)
    if not torch.isfinite(result.loss):
        raise AssertionError("Nonfinite training loss")
    result.loss.backward()
    if not any(value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
               for _, value in trainable):
        raise AssertionError("No valid adapter gradients")
    if any(value.grad is not None for _, value in frozen):
        raise AssertionError("A frozen original parameter received a gradient")
    torch.nn.utils.clip_grad_norm_([value for _, value in trainable], 1.0, error_if_nonfinite=True)
    optimizer.step()
    if not any(not torch.equal(before[name], value.detach().cpu()) for name, value in trainable):
        raise AssertionError("No adapter parameter changed")
    if any(not torch.equal(frozen_before[name], value) for name, value in representatives):
        raise AssertionError("Frozen parameter changed")
    evidence["checks"]["adapter_updates"] = True
    evidence["trainable_parameters"] = sum(value.numel() for _, value in trainable)
    adapter = output / "disposable_adapter"
    runtime.model.save_pretrained(adapter)
    tuned = runtime.scores(record)
    evidence["after_step_scores"] = tuned
    # Release all parameter references before the second base load.
    del inputs, encoded, result, optimizer, trainable, frozen, representatives, before, frozen_before, runtime
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = ModelRuntime(config, args.data_root, adapter=adapter)
    evidence["checks"]["reload_max_difference"] = compare(tuned, reloaded.scores(record))
    evidence["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    (output / "smoke.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
