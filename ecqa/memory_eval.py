"""Sealed-input evaluation and descriptive reports for external-memory diagnostics.

Cases keep gold and target annotations outside ``record``. A case's ``evidence``
may contain target_episode_id, target_clip_index (one based), target_pair_count,
delay_clips, retrieval_seconds, and store_bytes. Target coverage is calculated
only after resolving the already selected observations from a verified store.
The previously evaluated test set is a regression diagnostic, never a blind test.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import time

from .artifacts import atomic_json, code_identity, digest, input_identity
from .evaluate import _checked_scores

CONDITIONS = ("full", "oracle", "recent", "uniform", "retrieval", "wrong", "empty")
SCHEMA = "external-memory-evaluation-v1"


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    return value


def _number(value, field, *, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0
            or (integer and not isinstance(value, int))):
        raise ValueError(f"{field} must be finite and nonnegative")
    return value


def _reader_record(record):
    """Copy the exact reader allowlist; audit/gold are never passed to runtime."""
    question = _text(record.get("question"), "question")
    options = record.get("options")
    if not isinstance(options, dict) or set(options) != set("ABCD"):
        raise ValueError("Reader requires exactly A-D options")
    for value in options.values():
        _text(value, "option")
    safe = {"question": question, "options": deepcopy(options)}
    if "memory" in record:
        memory = record["memory"]
        if not isinstance(memory, dict) or set(memory) != {"episode_id", "store_digest", "observation_ids"}:
            raise ValueError("Memory reader references must use the exact allowlist")
        _text(memory["episode_id"], "episode_id")
        _text(memory["store_digest"], "store_digest")
        ids = memory["observation_ids"]
        if (not isinstance(ids, list) or any(isinstance(i, bool) or not isinstance(i, int) or i < 1 for i in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("Memory observation IDs must be distinct positive integers")
        safe["memory"] = deepcopy(memory)
    else:
        original = record["conditions"]["original"]
        safe["conditions"] = {"original": {key: deepcopy(original[key])
                                               for key in ("frames", "clip_ranges", "fps")}}
    return safe


def _store_snapshot(root, reference):
    from .memory import MemoryStore

    with MemoryStore.open(root) as store:
        manifest = store.verify()
        if manifest["store_digest"] != reference["store_digest"]:
            raise ValueError("Reader reference differs from verified store digest")
        available = {row["observation_id"]: row for row in store.observations(reference["episode_id"])}
        if not set(reference["observation_ids"]).issubset(available):
            raise ValueError("Reader selected observations absent from its episode/store")
        selected = [available[i] for i in reference["observation_ids"]]
    # Byte counts describe the actual sealed directory, including metadata/database.
    size = sum(path.stat().st_size for path in Path(root).rglob("*") if path.is_file())
    return selected, size


def _coverage(evidence, selected):
    result = {"selected_pairs": len(selected) if selected is not None else None,
              "target_clip_hit": None, "target_pairs_selected": None, "target_pair_recall": None}
    episode = evidence.get("target_episode_id")
    clip = evidence.get("target_clip_index")
    count = evidence.get("target_pair_count")
    if episode is not None:
        _text(episode, "target_episode_id")
    if clip is not None and (_number(clip, "target_clip_index", integer=True) == 0):
        raise ValueError("Target clip index is one based")
    if count is not None and (_number(count, "target_pair_count", integer=True) == 0):
        raise ValueError("Target pair denominator must be positive")
    if selected is not None and episode is not None and clip is not None:
        matches = {(row["episode_id"], row["observation_id"]) for row in selected
                   if row["episode_id"] == episode and row["clip_index"] == clip}
        result.update(target_clip_hit=bool(matches), target_pairs_selected=len(matches))
        if count is not None:
            if len(matches) > count:
                raise ValueError("Selected target pairs exceed the audit denominator")
            result["target_pair_recall"] = len(matches) / count
    return result


def _prepare_case(case, default_root):
    identifier = _text(case.get("id"), "id")
    split, condition, gold = case.get("split"), case.get("condition"), case.get("gold")
    if split not in {"train", "val", "test"}:
        raise ValueError("Case requires a known split")
    _text(condition, "condition")
    if not isinstance(gold, str) or gold not in tuple("ABCD"):
        raise ValueError("Gold must be one A-D letter")
    record = _reader_record(case["record"])
    root = Path(case.get("data_root", default_root)).resolve()
    evidence = deepcopy(case.get("evidence", case.get("evidence_info", {})))
    if not isinstance(evidence, dict):
        raise ValueError("Case evidence must be a dictionary")
    selected, store_bytes = None, None
    if "memory" in record:
        selected, store_bytes = _store_snapshot(root, record["memory"])
        store_digest = record["memory"]["store_digest"]
        if case.get("store_digest", store_digest) != store_digest:
            raise ValueError("Case and reader store digests differ")
        input_digest = digest({"record": record, "observations": selected})
    else:
        # Full-context baseline retains exactly its original synthetic clip map.
        input_digest = input_identity(record, "original", root)
        store_digest = _text(case.get("store_digest"), "baseline store_digest")
        if (root / "manifest.json").exists():
            from .memory import MemoryStore

            with MemoryStore.open(root) as store:
                manifest = store.verify()
                if manifest["store_digest"] != store_digest:
                    raise ValueError("Baseline store identity differs from verified store")
                by_frames = {tuple(obs["frames"]): obs for obs in manifest["observations"]}
                frames = record["conditions"]["original"]["frames"]
                pairs = [tuple(frames[i:i + 2]) for i in range(0, len(frames), 2)]
                if not set(pairs).issubset(by_frames) or len(set(pairs)) != len(pairs):
                    raise ValueError("Baseline frames do not match distinct retained pairs")
                selected = [by_frames[pair] for pair in pairs]
            store_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    retrieval_seconds = evidence.get("retrieval_seconds")
    if retrieval_seconds is not None:
        _number(retrieval_seconds, "retrieval_seconds")
    delay = evidence.get("delay_clips")
    if delay is not None:
        _number(delay, "delay_clips", integer=True)
    if evidence.get("store_bytes") is not None:
        _number(evidence["store_bytes"], "store_bytes", integer=True)
    metrics = {**_coverage(evidence, selected), "retrieval_seconds": retrieval_seconds,
               "store_bytes": store_bytes, "delay_clips": delay}
    # The evidence digest binds audit inputs, selected evidence, and measured storage.
    identity = {"id": identifier, "split": split, "condition": condition,
                "input_digest": input_digest, "store_digest": store_digest,
                "evidence_digest": digest({"audit": evidence, "selected": selected, "metrics": metrics}),
                "gold_digest": digest(gold), "case_digest": digest({
                    "gold": gold, "source_group": case.get("source_group"),
                    "question_class": case.get("question_class", "unspecified"),
                    "retrieval": case.get("retrieval"),
                    "evidence": evidence})}
    return {"identity": identity, "record": record, "root": root, "metrics": metrics,
            "gold": gold, "source_group": case.get("source_group"),
            "question_class": case.get("question_class", "unspecified")}


def _row_digest(row):
    return digest({key: value for key, value in row.items() if key != "row_digest"})


def _validate_row(row, identity=None):
    if not isinstance(row, dict) or row.get("row_digest") != _row_digest(row):
        raise ValueError("Prediction cache integrity mismatch")
    if identity and any(row.get(key) != value for key, value in identity.items()):
        raise ValueError("Prediction identity mismatch")
    _number(row.get("seconds"), "prediction timing")
    if row.get("status") == "ok":
        prediction, margin = _checked_scores(row.get("scores"))
        if row.get("prediction") != prediction or row.get("margin") != margin:
            raise ValueError("Cached prediction disagrees with scores")
    elif row.get("status") == "error":
        _text(row.get("error"), "error type")
        if not isinstance(row.get("message"), str):
            raise ValueError("Error cache requires an error message")
    else:
        raise ValueError("Unknown prediction status")


def evaluate_cases(runtime, cases, output_dir, *, model_name, model_digest, protocol_digest):
    """Return atomic score rows, refusing changed identities in an existing run.

    ``runtime.scores(record, 'original')`` must return finite A-D likelihoods.
    Runtime roots are set explicitly per case and restored even on failure. Error
    rows are saved and retried on an identical resume; corrupt caches fail closed.
    """
    if model_name not in {"base", "tuned"}:
        raise ValueError("Model name must distinguish base and tuned")
    for field, value in (("model_digest", model_digest), ("protocol_digest", protocol_digest)):
        _text(value, field)
    cases = list(cases)
    if not cases:
        raise ValueError("Empty memory evaluation cohort")
    prepared = [_prepare_case(case, runtime.data_root) for case in cases]
    cells = [(p["identity"]["id"], p["identity"]["split"], p["identity"]["condition"]) for p in prepared]
    if len(set(cells)) != len(cells):
        raise ValueError("Duplicate evaluation cells")
    shared = {"schema": SCHEMA, "model": model_name, "model_digest": model_digest,
              "protocol_digest": protocol_digest, "config_digest": digest(runtime.config),
              "code_digest": code_identity(), "scorer_version": "contextual_answer_likelihood_v1"}
    manifest = {**shared, "cases": [p["identity"] for p in prepared]}
    folder = Path(output_dir) / model_name
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Frozen evaluation manifest mismatch; use a distinct run for changed inputs")
    else:
        if list(folder.glob("*.json")):
            raise ValueError("Prediction files exist without a frozen manifest")
        atomic_json(manifest_path, manifest)
    rows = []
    for item in prepared:
        identity = {**shared, **item["identity"]}
        path = folder / (digest({key: identity[key] for key in ("id", "split", "condition")}) + ".json")
        if path.exists():
            previous = json.loads(path.read_text())
            _validate_row(previous, identity)
            if previous["status"] == "ok":
                rows.append(previous)
                continue
        started, original_root = time.monotonic(), runtime.data_root
        try:
            runtime.data_root = item["root"]
            scores = runtime.scores(deepcopy(item["record"]), "original")
            prediction, margin = _checked_scores(scores)
            row = {**identity, "status": "ok", "scores": scores, "prediction": prediction,
                   "margin": margin, "seconds": time.monotonic() - started,
                   "evidence_metrics": item["metrics"]}
        except Exception as error:
            row = {**identity, "status": "error", "error": type(error).__name__,
                   "message": str(error), "seconds": time.monotonic() - started}
            row["row_digest"] = _row_digest(row)
            atomic_json(path, row)
            raise
        finally:
            runtime.data_root = original_root
        row["row_digest"] = _row_digest(row)
        _validate_row(row, identity)
        atomic_json(path, row)
        rows.append(row)
    return rows


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _paired(left, right, gold, groups, samples, seed, identifiers=None):
    differences = [int(b == answer) - int(a == answer) for a, b, answer in zip(left, right, gold)]
    estimate = sum(differences) / len(differences)
    clusters = defaultdict(list)
    for index, group in enumerate(groups):
        clusters[group].append(index)
    interval = None
    if None not in clusters and len(clusters) >= 2 and samples:
        rng = random.Random(seed)
        units = list(clusters.values())
        boot = []
        for _ in range(samples):
            indices = [index for unit in rng.choices(units, k=len(units)) for index in unit]
            boot.append(sum(differences[i] for i in indices) / len(indices))
        boot.sort()
        from .analysis import _percentile
        interval = [_percentile(boot, 0.025), _percentile(boot, 0.975)]
    result = {"accuracy_change": estimate, "corrected": differences.count(1),
              "regressed": differences.count(-1), "unchanged": differences.count(0),
              "source_groups": len(clusters) if None not in clusters else None, "ci95": interval}
    if identifiers is not None:
        result["corrected_ids"] = [identifier for identifier, delta in zip(identifiers, differences) if delta == 1]
        result["regressed_ids"] = [identifier for identifier, delta in zip(identifiers, differences) if delta == -1]
    return result


def summarize_cases(cases, rows, *, expected_models=("base", "tuned"), expected_conditions=None,
                    bootstrap_samples=2000, seed=20260906):
    """Validate complete pairing and summarize already scored, fixed cases.

    ``expected_conditions`` maps split to its required conditions; by default
    train requires full and val/test require all seven diagnostics. No missing,
    duplicate, failed, unexpected, or differently identified cells are discarded.
    """
    cases, rows = list(cases), list(rows)
    if not cases or not rows:
        raise ValueError("Empty memory cohort or prediction table")
    if (not expected_models or len(set(expected_models)) != len(expected_models)
            or not set(expected_models).issubset({"base", "tuned"})):
        raise ValueError("Expected models must be distinct base/tuned identities")
    _number(bootstrap_samples, "bootstrap_samples", integer=True)
    _number(seed, "seed", integer=True)
    if expected_conditions is None:
        expected_conditions = {"train": ("full",), "val": CONDITIONS, "test": CONDITIONS}
    for conditions in expected_conditions.values():
        if not conditions or len(set(conditions)) != len(conditions):
            raise ValueError("Expected conditions must be distinct and nonempty")
    by_case, per_question = {}, defaultdict(set)
    prepared_cases = {}
    for case in cases:
        key = (case["split"], case["id"], case["condition"])
        if key in by_case:
            raise ValueError("Duplicate case cells")
        if case["gold"] not in tuple("ABCD"):
            raise ValueError("Invalid case gold")
        by_case[key] = case
        prepared_cases[key] = _prepare_case(case, case.get("data_root"))
        per_question[key[:2]].add(case["condition"])
    for (split, _), conditions in per_question.items():
        if split not in expected_conditions or conditions != set(expected_conditions[split]):
            raise ValueError("Missing or unexpected case conditions")
    question_bindings = defaultdict(set)
    for key, item in by_case.items():
        question_bindings[key[:2]].add(digest({"question": item["record"]["question"],
                                              "options": item["record"]["options"], "gold": item["gold"],
                                              "source_group": item.get("source_group")}))
    if any(len(bindings) != 1 for bindings in question_bindings.values()):
        raise ValueError("Paired conditions differ in question/options/gold/source grouping")
    table, identities = {}, defaultdict(set)
    for row in rows:
        _validate_row(row)
        if row["status"] != "ok":
            raise ValueError("Failed prediction cells cannot enter a report")
        key = (row["split"], row["id"], row["condition"])
        cell = (*key, row["model"])
        if key not in by_case or row["model"] not in expected_models or cell in table:
            raise ValueError("Duplicate or unexpected prediction cells")
        prepared = prepared_cases[key]
        if any(row.get(field) != value for field, value in prepared["identity"].items()):
            raise ValueError("Prediction case/input/evidence identity mismatch")
        if row.get("evidence_metrics") != prepared["metrics"]:
            raise ValueError("Prediction evidence metrics differ from verified selected evidence")
        table[cell] = row
        for field in ("protocol_digest", "config_digest", "code_digest"):
            identities[field].add(row[field])
        identities["model:" + row["model"]].add(row["model_digest"])
    if len(table) != len(by_case) * len(expected_models) or any(len(values) != 1 for values in identities.values()):
        raise ValueError("Missing prediction cells or mixed frozen identities")
    if (set(expected_models) == {"base", "tuned"}
            and identities["model:base"] == identities["model:tuned"]):
        raise ValueError("Base and tuned comparisons require distinct model digests")
    for key in by_case:
        pair = [table[(*key, model)] for model in expected_models]
        for field in ("input_digest", "store_digest", "evidence_digest", "case_digest"):
            if len({row[field] for row in pair}) != 1:
                raise ValueError("Paired model inputs/evidence differ")
    splits = {}
    for split in sorted({key[0] for key in by_case}):
        conditions = {}
        for condition in expected_conditions[split]:
            keys = sorted(key for key in by_case if key[0] == split and key[2] == condition)
            gold = [by_case[key]["gold"] for key in keys]
            groups = [by_case[key].get("source_group") for key in keys]
            model_stats = {}
            for model in expected_models:
                selected = [table[(*key, model)] for key in keys]
                correct = [row["prediction"] == answer for row, answer in zip(selected, gold)]
                breakdown = {}
                for label in "ABCD":
                    indices = [i for i, answer in enumerate(gold) if answer == label]
                    hits = sum(correct[i] for i in indices)
                    breakdown[label] = {"count": len(indices), "correct": hits,
                                        "accuracy": hits / len(indices) if indices else None}
                model_stats[model] = {"count": len(keys), "correct": sum(correct),
                                      "accuracy": sum(correct) / len(keys), "by_gold_class": breakdown,
                                      "mean_gold_negative_logscore": _mean(
                                          -row["scores"][answer] for row, answer in zip(selected, gold)),
                                      "gold_negative_logscore_definition":
                                          "Answer-token negative log-likelihood in nats; excludes EOS; not training loss",
                                      "mean_scoring_seconds": _mean(row["seconds"] for row in selected)}
            evidence = [table[(*key, expected_models[0])]["evidence_metrics"] for key in keys]
            detail = {"models": model_stats, "evidence": {
                "target_coverage_cases": sum(e["target_clip_hit"] is not None for e in evidence),
                "target_clip_hit_rate": _mean(e["target_clip_hit"] for e in evidence),
                "mean_target_pair_recall": _mean(e["target_pair_recall"] for e in evidence),
                "mean_selected_pairs": _mean(e["selected_pairs"] for e in evidence),
                "mean_store_bytes": _mean(e["store_bytes"] for e in evidence),
                "mean_retrieval_seconds": _mean(e["retrieval_seconds"] for e in evidence)},
                "delay_clips_counts": dict(Counter(str(e["delay_clips"]) for e in evidence)),
                "question_class_counts": dict(Counter(by_case[k].get("question_class", "unspecified") for k in keys))}
            if set(expected_models) == {"base", "tuned"}:
                detail["tuned_minus_base"] = _paired(
                    [table[(*key, "base")]["prediction"] for key in keys],
                    [table[(*key, "tuned")]["prediction"] for key in keys],
                    gold, groups, bootstrap_samples, seed, [key[1] for key in keys])
            conditions[condition] = detail
        contrasts = {}
        pairs = [("memory_full", "full"), ("retrieval", "uniform"), ("retrieval", "recent"),
                 ("oracle_all", "oracle"), ("oracle", "retrieval"), ("retrieval", "wrong"),
                 ("retrieval", "empty")]
        pairs.extend((name, "competing_" + name) for name in expected_conditions[split]
                     if not name.startswith("competing_"))
        for left, right in pairs:
            if left not in conditions or right not in conditions:
                continue
            keys = sorted(key for key in by_case if key[0] == split and key[2] == left)
            right_keys = [(key[0], key[1], right) for key in keys]
            if any(by_case[a]["gold"] != by_case[b]["gold"] for a, b in zip(keys, right_keys)):
                raise ValueError("Paired conditions have different gold labels")
            contrasts[f"{left}_minus_{right}"] = {
                model: _paired([table[(*key, model)]["prediction"] for key in right_keys],
                               [table[(*key, model)]["prediction"] for key in keys],
                               [by_case[key]["gold"] for key in keys],
                               [by_case[key].get("source_group") for key in keys], bootstrap_samples, seed,
                               [key[1] for key in keys])
                for model in expected_models}
        splits[split] = {"interpretation": {"train": "training fit diagnostic",
                                            "val": "validation diagnostic",
                                            "test": "previously used test: descriptive regression only"}[split],
                         "conditions": conditions, "within_model_contrasts": contrasts}
    return {"schema": SCHEMA, "prediction_cells": len(rows), "splits": splits,
            "identities": {key: next(iter(value)) for key, value in identities.items()},
            "bootstrap": {"samples": bootstrap_samples, "seed": seed,
                          "method": "paired source-group percentile bootstrap; question-weighted"},
            "limitations": ["Labels were AI-reviewed; no human validation is claimed.",
                            "Old test results are diagnostic and do not establish generalization.",
                            "Source-group intervals depend on available grouping; hidden overlap may remain.",
                            "Coverage measures retained evidence availability, not whether the reader used it.",
                            "Store bytes exclude model weights; scoring and retrieval latency are reported separately."]}


def write_memory_report(cases, rows, output_dir, **kwargs):
    """Write a checked JSON summary and compact Markdown report atomically."""
    summary = summarize_cases(cases, rows, **kwargs)
    root = Path(output_dir)
    atomic_json(root / "memory-summary.json", summary)
    lines = ["External-memory diagnostic results", "",
             "These are training/validation diagnostics and regression checks on a previously used test set.", "",
             "| Split | Condition | Model | Correct / count | Accuracy | Gold NLL (nats) | Scoring seconds |",
             "|---|---|---|---:|---:|---:|---:|"]
    for split, body in summary["splits"].items():
        for condition, detail in body["conditions"].items():
            for model, stats in detail["models"].items():
                lines.append(f"| {split} | {condition} | {model} | {stats['correct']} / {stats['count']} "
                             f"| {stats['accuracy']:.4f} | {stats['mean_gold_negative_logscore']:.4f} "
                             f"| {stats['mean_scoring_seconds']:.3f} |")
    lines += ["", "Gold NLL is answer-token negative log-likelihood, excluding EOS; it is not training loss.", "",
              "| Split | Condition | Target clip hit rate | Target pair recall | Selected pairs | Store bytes | Retrieval seconds |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for split, body in summary["splits"].items():
        for condition, detail in body["conditions"].items():
            values = [detail["evidence"][key] for key in ("target_clip_hit_rate", "mean_target_pair_recall",
                      "mean_selected_pairs", "mean_store_bytes", "mean_retrieval_seconds")]
            lines.append(f"| {split} | {condition} | " + " | ".join(
                "unavailable" if value is None else f"{value:.4f}" for value in values) + " |")
    lines += ["", "| Split | Comparison | Model | Accuracy change | Corrected | Regressed | Source-group CI95 |",
              "|---|---|---|---:|---:|---:|---|"]
    for split, body in summary["splits"].items():
        comparisons = dict(body["within_model_contrasts"])
        comparisons.update({f"{name}: tuned minus base": {"base→tuned": detail["tuned_minus_base"]}
                            for name, detail in body["conditions"].items() if "tuned_minus_base" in detail})
        for comparison, models in comparisons.items():
            for model, stats in models.items():
                lines.append(f"| {split} | {comparison} | {model} | {stats['accuracy_change']:.4f} "
                             f"| {stats['corrected']} | {stats['regressed']} | {stats['ci95']} |")
    lines += ["", *["- " + value for value in summary["limitations"]], ""]
    from .analysis import _atomic_write
    _atomic_write(root / "memory-report.md", "\n".join(lines))
    return summary
