"""Verify and display a saved memory diagnostic using only the standard library.

Required: metrics.json and snapshot.json with files:{name:{sha256,bytes}}.
Optional: per_question.json with flat id/split/condition/answer/base_prediction/
tuned_prediction rows, and gpu-smoke.json. Every consumed file must be inventoried.
This does not perform inference, verify visual labels, or access original media.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def close(actual, expected, label):
    if (isinstance(actual, bool) or not isinstance(actual, (int, float))
            or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12)):
        raise ValueError(f"Saved numeric result differs: {label}")


def inspect(folder):
    folder = Path(folder).resolve()
    snapshot = json.loads((folder / "snapshot.json").read_text())
    files = snapshot["files"]
    if not isinstance(files, dict) or "metrics.json" not in files:
        raise ValueError("Snapshot inventory must include metrics.json")
    for name, expected in files.items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder) or path == folder or (folder / name).is_symlink():
            raise ValueError("Snapshot path escapes its directory or is a symlink")
        content = path.read_bytes()
        if (type(expected["bytes"]) is not int or len(content) != expected["bytes"]
                or hashlib.sha256(content).hexdigest() != expected["sha256"]):
            raise ValueError(f"Snapshot file checksum mismatch: {name}")

    def read(name):
        if name not in files:
            raise ValueError(f"Consumed file is absent from snapshot inventory: {name}")
        return json.loads((folder / name).read_text())

    metrics = read("metrics.json")
    if metrics.get("schema") != "external-memory-evaluation-v1" or not metrics.get("splits"):
        raise ValueError("Unknown or empty memory summary schema")
    protocol = metrics["identities"]["protocol_digest"]
    if snapshot.get("protocol_digest", protocol) != protocol:
        raise ValueError("Snapshot and metrics protocol identities differ")
    expected, cells = {}, 0
    for split, body in metrics["splits"].items():
        if split not in {"train", "val", "test"} or not body["conditions"]:
            raise ValueError("Unknown or empty split")
        counts = set()
        for condition, detail in body["conditions"].items():
            if set(detail["models"]) != {"base", "tuned"}:
                raise ValueError("Both distinct base and tuned model summaries are required")
            for model, stats in detail["models"].items():
                count, correct = stats["count"], stats["correct"]
                if type(count) is not int or count < 1 or type(correct) is not int or not 0 <= correct <= count:
                    raise ValueError("Invalid count or correct total")
                close(stats["accuracy"], correct / count, f"{split}/{condition}/{model}")
                classes = stats["by_gold_class"]
                if set(classes) != set("ABCD"):
                    raise ValueError("Missing A-D class breakdown")
                for label, item in classes.items():
                    n, k = item["count"], item["correct"]
                    if type(n) is not int or type(k) is not int or n < 0 or not 0 <= k <= n:
                        raise ValueError("Invalid class counts")
                    if n:
                        close(item["accuracy"], k / n, f"{split}/{condition}/{model}/{label}")
                    elif item["accuracy"] is not None:
                        raise ValueError("Empty classes must have unavailable accuracy")
                if sum(c["count"] for c in classes.values()) != count or sum(c["correct"] for c in classes.values()) != correct:
                    raise ValueError("Class totals differ from model aggregate")
                expected[(split, condition, model)] = (count, correct)
                counts.add(count)
                cells += count
        if len(counts) != 1:
            raise ValueError("Conditions/models do not share the same split size")
    if type(metrics["prediction_cells"]) is not int or cells != metrics["prediction_cells"]:
        raise ValueError("Prediction cell count differs from summaries")
    questions = read("per_question.json") if "per_question.json" in files else metrics.get("per_question")
    if questions is not None:
        if not isinstance(questions, list) or not questions:
            raise ValueError("Per-question rows must be a nonempty list")
        seen, correct, question_sets, labels = set(), Counter(), defaultdict(set), {}
        for row in questions:
            key = (row["split"], row["id"], row["condition"])
            if key in seen or row["answer"] not in tuple("ABCD"):
                raise ValueError("Duplicate per-question cell or invalid gold")
            seen.add(key)
            question_sets[(row["split"], row["condition"])].add(row["id"])
            if labels.setdefault(key[:2], row["answer"]) != row["answer"]:
                raise ValueError("Question gold differs across conditions")
            for model in ("base", "tuned"):
                group = (row["split"], row["condition"], model)
                prediction = row[f"{model}_prediction"]
                if group not in expected or prediction not in tuple("ABCD"):
                    raise ValueError("Unexpected prediction cell or invalid answer")
                correct[group] += prediction == row["answer"]
        if len(seen) * 2 != cells:
            raise ValueError("Per-question prediction cells are incomplete")
        for group, (count, hits) in expected.items():
            if len(question_sets[group[:2]]) != count or correct[group] != hits:
                raise ValueError(f"Per-question outcomes disagree: {group}")
        for split in metrics["splits"]:
            sets = [ids for (s, _), ids in question_sets.items() if s == split]
            if any(ids != sets[0] for ids in sets):
                raise ValueError("Conditions contain different question cohorts")
    print("External-memory saved diagnostics — no inference or media access")
    print(f"Verified {len(files)} snapshot files; {cells} model/condition prediction cells.")
    print(f"{'Split':<21} {'Condition':<30} {'Base':>12} {'Tuned':>12} {'Change pp':>10}")
    for split, body in metrics["splits"].items():
        label = "old test (regression)" if split == "test" else split
        for condition, detail in body["conditions"].items():
            base, tuned = (detail["models"][model] for model in ("base", "tuned"))
            left, right = (f"{s['correct']}/{s['count']}" for s in (base, tuned))
            change = 100 * (tuned["accuracy"] - base["accuracy"])
            print(f"{label:<21} {condition:<30} {left:>12} {right:>12} {change:>+10.2f}")
    if "gpu-smoke.json" in files:
        smoke = read("gpu-smoke.json")
        if smoke.get("protocol_digest") != protocol:
            raise ValueError("GPU smoke and metrics protocol identities differ")
        print(f"Recorded GPU smoke status: {smoke.get('status', 'unavailable')}; "
              f"recorded checks: {len(smoke.get('checks', []))}")
    print("Per-question counts and accuracy recomputed." if questions is not None
          else "Aggregate consistency only: per-question rows are unavailable.")
    print("Train = fixed-set fit; validation = diagnostic; old test = descriptive regression, not a fresh holdout.")
    for limitation in metrics.get("limitations", []):
        print(f"Limit: {limitation}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "memory")
    args = parser.parse_args()
    inspect(args.root)


if __name__ == "__main__":
    main()
