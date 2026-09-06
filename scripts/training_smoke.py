"""Diagnostic-only comparison of uninterrupted and interrupted/resumed training.

Four deliberately duplicated examples with unique IDs exercise the loop. They
are not an experimental dataset and no accuracy conclusion is computed.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import time

# Declared before measurement; these are absolute numerical tolerances.
STATE_ATOL = 1e-5
LOSS_ATOL = 1e-5
SCORE_ATOL = 0.03
OUTPUT_CREATED_BY_THIS_PROCESS: Path | None = None


def main():
    global OUTPUT_CREATED_BY_THIS_PROCESS
    import numpy as np
    import torch
    import yaml
    from safetensors.torch import load_file

    from ecqa.artifacts import atomic_json, digest, file_digest
    from ecqa.model import ModelRuntime
    from ecqa.train import train, verify_checkpoint

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--deterministic-math", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    OUTPUT_CREATED_BY_THIS_PROCESS = output
    started = time.monotonic()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.deterministic_math:
        config["model"].update(deterministic=True, attention_backend="sdpa_math")
    config["training"].update(epochs=1, seed=42, gradient_accumulation_steps=1,
                              warmup_ratio=0.0, checkpoint_every_optimizer_steps=2,
                              max_cumulative_seconds=300)
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    original = next(row for row in rows if row["split"] == "train")
    records = []
    for index in range(4):
        row = deepcopy(original)
        row["id"] = f"diagnostic-repeat-{index}:{original['id']}"
        records.append(row)
    protocol = digest({"purpose": "four-update-resume-plumbing-only", "config": config, "records": records})
    (output / "resolved-config.yaml").write_text(yaml.safe_dump(config))
    (output / "diagnostic-records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    evidence = {"purpose": "diagnostic-only controlled checkpoint resume",
                "source_id": original["id"], "protocol_digest": protocol,
                "runtime_profile": config["model"],
                "tolerances": {"state_atol": STATE_ATOL, "loss_atol": LOSS_ATOL,
                               "score_atol": SCORE_ATOL, "same_argmax_required": True}}

    def release(runtime):
        # Remove the model reference while this helper still owns runtime.
        runtime.model = None
        gc.collect()
        torch.cuda.empty_cache()

    runtime = ModelRuntime(config, args.data_root)
    full_summary = train(runtime, records, config, output / "uninterrupted",
                         protocol_digest=protocol, diagnostic=True)
    full_scores = runtime.scores(original)
    release(runtime)
    del runtime
    if full_summary["updates"] != 4 or full_summary["stop_reason"] != "epoch_complete":
        raise AssertionError("Uninterrupted diagnostic did not complete four updates")

    runtime = ModelRuntime(config, args.data_root)
    stopped = train(runtime, records, config, output / "resumed", protocol_digest=protocol,
                    diagnostic=True, stop_after_updates=2)
    release(runtime)
    del runtime
    if stopped["updates"] != 2 or stopped["stop_reason"] != "diagnostic_stop":
        raise AssertionError("Controlled stop did not occur at the second optimizer boundary")
    resume_path = output / "resumed" / stopped["checkpoint"]
    runtime = ModelRuntime(config, args.data_root)
    resumed_summary = train(runtime, records, config, output / "resumed", protocol_digest=protocol,
                            diagnostic=True, resume=resume_path)
    resumed_scores = runtime.scores(original)
    release(runtime)
    del runtime
    if resumed_summary["updates"] != 4 or resumed_summary["stop_reason"] != "epoch_complete":
        raise AssertionError("Resumed diagnostic did not complete the remaining two updates")

    def compare_tree(left, right, path="state"):
        """Compare numeric state while retaining exact structure and discrete values."""
        if isinstance(left, torch.Tensor):
            if left.shape != right.shape or left.dtype != right.dtype:
                raise AssertionError(f"Tensor metadata differs at {path}")
            if left.is_floating_point():
                delta = float((left.float() - right.float()).abs().max().item()) if left.numel() else 0.0
                if not torch.isfinite(left).all() or not torch.isfinite(right).all() or delta > STATE_ATOL:
                    raise AssertionError(f"Tensor state differs at {path}: {delta}")
                return delta
            if not torch.equal(left, right):
                raise AssertionError(f"Discrete tensor differs at {path}")
        elif isinstance(left, np.ndarray):
            if not np.array_equal(left, right):
                raise AssertionError(f"Array state differs at {path}")
        elif isinstance(left, dict):
            if left.keys() != right.keys():
                raise AssertionError(f"Dictionary keys differ at {path}")
            return max((compare_tree(left[key], right[key], f"{path}.{key}") for key in left), default=0.0)
        elif isinstance(left, (list, tuple)):
            if type(left) is not type(right) or len(left) != len(right):
                raise AssertionError(f"Sequence differs at {path}")
            return max((compare_tree(a, b, f"{path}[{i}]") for i, (a, b) in enumerate(zip(left, right))), default=0.0)
        elif left != right:
            raise AssertionError(f"Scalar state differs at {path}: {left!r}, {right!r}")
        return 0.0

    def checkpoint(run, step):
        location = output / run / f"checkpoint-{step:06d}"
        metadata = verify_checkpoint(location)
        # These files were just produced locally by this diagnostic run.
        state = torch.load(location / "training.pt", map_location="cpu", weights_only=False)
        adapter = load_file(str(location / "adapter_model.safetensors"))
        return location, metadata, state, adapter

    evidence["checkpoint_comparisons"] = {}
    for step in (2, 4):
        full_path, full_meta, full_state, full_adapter = checkpoint("uninterrupted", step)
        resumed_path, resumed_meta, resumed_state, resumed_adapter = checkpoint("resumed", step)
        if full_meta["identity"] != resumed_meta["identity"]:
            raise AssertionError("Paired checkpoint identities differ")
        if any(state["step"] != step or state["next_index"] != step for state in (full_state, resumed_state)):
            raise AssertionError("Checkpoint optimizer/data position is wrong")
        components = ("optimizer", "scheduler", "order", "next_index", "step", "python_rng",
                      "numpy_rng", "torch_rng", "cuda_rng", "identity")
        differences = {key: compare_tree(full_state[key], resumed_state[key], key) for key in components}
        differences["adapter_max_absolute_difference"] = compare_tree(full_adapter, resumed_adapter, "adapter")
        differences["full_adapter_sha256"] = file_digest(full_path / "adapter_model.safetensors")
        differences["resumed_adapter_sha256"] = file_digest(resumed_path / "adapter_model.safetensors")
        evidence["checkpoint_comparisons"][str(step)] = differences
    _, _, _, initial_progress = checkpoint("resumed", 2)
    _, _, _, final_progress = checkpoint("resumed", 4)
    progress_delta = max(float((initial_progress[key] - final_progress[key]).abs().max().item())
                         for key in initial_progress)
    if progress_delta <= 0:
        raise AssertionError("Adapter weights did not change after resume")

    def events(run):
        return [json.loads(line) for line in (output / run / "events.jsonl").read_text().splitlines()]

    first_events, second_events = events("uninterrupted"), events("resumed")
    if [event["step"] for event in first_events] != [1, 2, 3, 4] or [event["step"] for event in second_events] != [1, 2, 3, 4]:
        raise AssertionError("Resume duplicated or skipped optimizer events")
    loss_delta = max(abs(first["mean_loss"] - second["mean_loss"])
                     for first, second in zip(first_events, second_events))
    if loss_delta > LOSS_ATOL:
        raise AssertionError(f"Loss continuation differs: {loss_delta}")
    used_lrs = [config["training"]["learning_rate"]] + [event["lr"] for event in second_events[:-1]]
    if not all(lr > 0 for lr in used_lrs):
        raise AssertionError("A diagnostic update used zero learning rate")
    if not all(event["gradient_norm"] > 0 for event in second_events):
        raise AssertionError("A diagnostic update had no gradient signal")
    score_delta = max(abs(full_scores[label] - resumed_scores[label]) for label in full_scores)
    if score_delta > SCORE_ATOL or max(full_scores, key=full_scores.get) != max(resumed_scores, key=resumed_scores.get):
        raise AssertionError(f"Resumed model scores differ: {score_delta}")
    evidence.update(status="passed", full_summary=full_summary, stopped_summary=stopped,
                    resumed_summary=resumed_summary, full_scores=full_scores, resumed_scores=resumed_scores,
                    score_max_absolute_difference=score_delta, loss_max_absolute_difference=loss_delta,
                    resumed_adapter_progress_max_difference=progress_delta, applied_learning_rates=used_lrs,
                    full_losses=[event["mean_loss"] for event in first_events],
                    resumed_losses=[event["mean_loss"] for event in second_events],
                    elapsed_seconds=time.monotonic() - started)
    atomic_json(output / "training-smoke.json", evidence)
    print(json.dumps(evidence, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if OUTPUT_CREATED_BY_THIS_PROCESS is not None:
            target = OUTPUT_CREATED_BY_THIS_PROCESS
            with (target / "failure.json").open("x") as handle:
                handle.write(json.dumps({
                    "status": "failed", "error_type": type(error).__name__, "error": str(error),
                    "state_atol": STATE_ATOL, "loss_atol": LOSS_ATOL, "score_atol": SCORE_ATOL,
                }, indent=2) + "\n")
        raise
