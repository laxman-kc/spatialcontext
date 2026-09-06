"""Batch stages for a single auditable experiment."""
import argparse
import json
from pathlib import Path
import sys

from .artifacts import atomic_json, load_config, read_jsonl, verify_freeze


def main():
    if sys.argv[1:2] == ["memory"]:
        from .memory_study import main as memory_main
        return memory_main(sys.argv[2:])
    parser = argparse.ArgumentParser(prog="ecqa")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("memory", help="persistent observation stores, answers and memory diagnostics")
    for name in ("freeze", "train", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--manifest", required=True)
        command.add_argument("--data-root", required=True)
        command.add_argument("--protocol", required=True)
        if name in {"train", "evaluate"}:
            command.add_argument("--output", required=True)
        if name == "train":
            command.add_argument("--resume")
        if name == "evaluate":
            command.add_argument("--adapter")
            command.add_argument("--split", choices=["val", "test"], default="test")
    command = sub.add_parser("report")
    command.add_argument("--protocol", required=True)
    command.add_argument("--manifest", required=True)
    command.add_argument("--predictions", required=True, nargs="+")
    command.add_argument("--output", required=True)
    args = parser.parse_args()
    records = read_jsonl(args.manifest)
    if args.command == "report":
        from .analysis import write_report
        from .artifacts import verify_report_inputs
        rows = [row for path in args.predictions for row in read_jsonl(path)]
        protocol = verify_report_inputs(args.protocol, records, rows)
        analysis = protocol["config"]["analysis"]
        write_report(records, rows, args.output,
                     bootstrap_samples=analysis["bootstrap_samples"], seed=analysis["seed"])
        return
    config = load_config(args.config)
    if args.command == "freeze":
        from .artifacts import freeze
        print(json.dumps(freeze(config, records, args.data_root, args.protocol), indent=2))
        return
    protocol = verify_freeze(args.protocol, config, records, args.data_root)
    adapter = getattr(args, "adapter", None)
    selection_path = Path(args.protocol).with_name("selected-model.json")
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else None
    if selection and (selection.get("protocol_digest") != protocol["protocol_digest"]
                      or selection.get("rule") != "end_of_run"):
        raise ValueError("Run directory model selection belongs to another protocol")
    if args.command == "evaluate":
        from .evaluate import model_identity
        if args.split == "test":
            if not selection:
                raise ValueError("Final adapter selection must be frozen before either test model")
            if adapter and model_identity(config, adapter) != selection["model_digest"]:
                raise ValueError("Test adapter differs from the selected final checkpoint")
        if adapter:
            from .train import verify_checkpoint
            metadata = verify_checkpoint(adapter)
            if metadata["diagnostic"] or metadata["protocol_digest"] != protocol["protocol_digest"]:
                raise ValueError("Adapter is diagnostic or belongs to another protocol")
    from .model import ModelRuntime
    runtime = ModelRuntime(config, args.data_root, adapter=adapter)
    if args.command == "train":
        from .train import train
        from .evaluate import model_identity
        summary = train(runtime, [r for r in records if r["split"] == "train"], config, args.output,
                        protocol_digest=protocol["protocol_digest"], resume=args.resume)
        selected = Path(args.output) / summary["checkpoint"]
        selected_state = {
            "protocol_digest": protocol["protocol_digest"], "rule": "end_of_run",
            "model_digest": model_identity(config, selected), "training_identity": summary["identity"],
            "updates": summary["updates"], "checkpoint": summary["checkpoint"]}
        if selection is not None and selection != selected_state:
            raise ValueError("Final model selection is already frozen; use a new protocol/run directory")
        atomic_json(selection_path, selected_state)
        print(json.dumps(summary, indent=2))
    else:
        from .evaluate import evaluate, model_identity
        name = "tuned" if adapter else "base"
        rows = evaluate(runtime, records, Path(args.output) / "rows", model_name=name,
                        model_digest=model_identity(config, adapter), protocol_digest=protocol["protocol_digest"],
                        split=args.split)
        out = Path(args.output) / f"{name}-{args.split}.jsonl"
        temp = out.with_suffix(".jsonl.tmp")
        temp.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))
        temp.replace(out)
        atomic_json(Path(args.output) / f"{name}-{args.split}-complete.json", {"rows": len(rows),
                    "protocol_digest": protocol["protocol_digest"]})


if __name__ == "__main__":
    main()
