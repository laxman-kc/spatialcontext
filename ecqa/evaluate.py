"""Atomic per-input predictions, resumable only under identical identities."""
import json
import math
import time
from pathlib import Path

from .artifacts import atomic_json, digest, file_digest, input_identity


def _checked_scores(scores):
    if not isinstance(scores, dict) or set(scores) != set("ABCD"):
        raise ValueError("Fixed-choice scores must contain exactly A, B, C, D")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           for value in scores.values()):
        raise ValueError("Fixed-choice scores must be finite numbers")
    ordered = sorted(scores, key=lambda label: (-scores[label], label))
    return ordered[0], scores[ordered[0]] - scores[ordered[1]]


def model_identity(config, adapter=None):
    body = {"model": config["model"]}
    if adapter:
        root = Path(adapter)
        files = sorted(root.glob("adapter*"))
        if not any(p.suffix == ".safetensors" for p in files):
            raise ValueError("Adapter weights missing")
        body["adapter"] = {p.name: file_digest(p) for p in files if p.is_file()}
    return digest(body)


def evaluate(runtime, records, output_dir, *, model_name, model_digest, protocol_digest, split="test"):
    if model_name not in {"base", "tuned"}:
        raise ValueError("Unknown model state")
    if split not in {"val", "test"}:
        raise ValueError("Evaluation accepts validation or test records only")
    if any(not isinstance(value, str) or not value for value in (model_digest, protocol_digest)):
        raise ValueError("Evaluation requires nonempty model and protocol identities")
    records = [r for r in records if r["split"] == split]
    if not records:
        raise ValueError("Empty evaluation split")
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("Duplicate IDs in evaluation cohort")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        conditions = ["original", "neutral", "competing", "text_only"] if split == "test" else ["original"]
        for condition in conditions:
            identity = {"id": record["id"], "condition": condition, "model": model_name,
                        "model_digest": model_digest, "protocol_digest": protocol_digest,
                        "input_digest": input_identity(record, condition, runtime.data_root),
                        "scorer_version": "contextual_answer_likelihood_v1"}
            path = output / (digest(identity) + ".json")
            if path.exists():
                existing = json.loads(path.read_text())
                if any(existing.get(key) != value for key, value in identity.items()):
                    raise ValueError("Prediction identity mismatch")
                if existing["status"] == "ok":
                    predicted, margin = _checked_scores(existing.get("scores"))
                    seconds = existing.get("seconds")
                    if existing.get("prediction") != predicted or existing.get("margin") != margin:
                        raise ValueError("Cached prediction disagrees with its scores")
                    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                            or not math.isfinite(seconds) or seconds < 0):
                        raise ValueError("Cached prediction timing is invalid")
                    rows.append(existing)
                    continue
            started = time.monotonic()
            try:
                scores = runtime.scores(record, condition)
                predicted, margin = _checked_scores(scores)
                row = {**identity, "scores": scores, "prediction": predicted,
                       "margin": margin, "status": "ok",
                       "seconds": time.monotonic() - started}
            except Exception as error:
                atomic_json(path, {**identity, "status": "error", "error": type(error).__name__,
                                   "message": str(error), "seconds": time.monotonic() - started})
                raise
            atomic_json(path, row)
            rows.append(row)
            print(json.dumps({"id": record["id"], "model": model_name, "condition": condition,
                              "status": "ok", "seconds": row["seconds"]}), flush=True)
    return rows
