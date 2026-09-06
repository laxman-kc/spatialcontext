"""Strict paired analysis of the frozen two-model, four-condition experiment.

This module needs only the Python standard library. It deliberately refuses
partial results: failed or missing predictions must be resolved before a final
comparison can be produced. Report generation never calls a model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

LABELS = ("A", "B", "C", "D")
MODELS = ("base", "tuned")
CONDITIONS = ("original", "neutral", "competing", "text_only")
InputRows = str | Path | Sequence[Mapping[str, Any]]


class AnalysisError(ValueError):
    """Inputs cannot support the predefined complete paired comparison."""


def _read_rows(value: InputRows, name: str) -> list[dict[str, Any]]:
    if isinstance(value, (str, Path)):
        result = []
        with Path(value).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AnalysisError(f"{name} line {line_number}: invalid JSON") from exc
                if not isinstance(row, dict):
                    raise AnalysisError(f"{name} line {line_number}: expected an object")
                result.append(row)
        return result
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise AnalysisError(f"{name} must be a JSONL path or a sequence of objects")
    if any(not isinstance(row, Mapping) for row in value):
        raise AnalysisError(f"every {name} row must be an object")
    return [dict(row) for row in value]


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnalysisError(f"{field} must be a nonempty string")
    return value


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"{field} must be a finite number")
    if not math.isfinite(value):
        raise AnalysisError(f"{field} must be a finite number")
    return float(value)


def _validate(
    records: InputRows, predictions: InputRows
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]], dict[str, Any]]:
    raw_records = _read_rows(records, "records")
    seen: set[str] = set()
    test_records = []
    for record in raw_records:
        identifier = _text(record.get("id"), "record.id")
        if identifier in seen:
            raise AnalysisError(f"duplicate record id: {identifier}")
        seen.add(identifier)
        if record.get("split") not in ("train", "validation", "val", "test"):
            raise AnalysisError(f"{identifier}: missing or invalid split")
        if record["split"] != "test":
            continue
        _text(record.get("source_group"), f"{identifier}.source_group")
        if record.get("answer") not in LABELS:
            raise AnalysisError(f"{identifier}: answer must be A, B, C, or D")
        conditions = record.get("conditions")
        if not isinstance(conditions, Mapping) or set(conditions) != set(CONDITIONS):
            raise AnalysisError(f"{identifier}: expected exactly the four frozen conditions")
        if "audit" in record:
            audit = record["audit"]
            if not isinstance(audit, Mapping) or audit.get("status") != "approved":
                raise AnalysisError(f"{identifier}: final analysis requires an approved audit")
        test_records.append(record)
    if not test_records:
        raise AnalysisError("no test records to analyze")
    test_records.sort(key=lambda record: record["id"])
    by_id = {record["id"]: record for record in test_records}
    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    model_digests: dict[str, str] = {}
    protocol_digest: str | None = None
    input_digests: dict[tuple[str, str], str] = {}
    for row in _read_rows(predictions, "predictions"):
        identifier = _text(row.get("id"), "prediction.id")
        model, condition = row.get("model"), row.get("condition")
        if identifier not in by_id:
            raise AnalysisError(f"prediction is outside the test cohort: {identifier}")
        if model not in MODELS or condition not in CONDITIONS:
            raise AnalysisError(f"{identifier}: invalid model or condition")
        key = (identifier, model, condition)
        if key in indexed:
            raise AnalysisError(f"duplicate prediction: {key}")
        if row.get("status") != "ok":
            raise AnalysisError(f"prediction did not succeed: {key}; status={row.get('status')!r}")
        scores = row.get("scores")
        if not isinstance(scores, Mapping) or set(scores) != set(LABELS):
            raise AnalysisError(f"{key}: scores must contain exactly A, B, C, D")
        checked_scores = {label: _finite_number(scores[label], f"{key}.scores.{label}") for label in LABELS}
        selected = max(LABELS, key=checked_scores.__getitem__)
        if row.get("prediction") != selected:
            raise AnalysisError(f"{key}: prediction disagrees with scores (ties use A/B/C/D order)")
        if _finite_number(row.get("seconds"), f"{key}.seconds") < 0:
            raise AnalysisError(f"{key}: seconds cannot be negative")
        model_digest = _text(row.get("model_digest"), f"{key}.model_digest")
        if model in model_digests and model_digests[model] != model_digest:
            raise AnalysisError(f"inconsistent model identity for {model}")
        model_digests[model] = model_digest
        protocol = _text(row.get("protocol_digest"), f"{key}.protocol_digest")
        if protocol_digest is not None and protocol_digest != protocol:
            raise AnalysisError("inconsistent protocol identity across predictions")
        protocol_digest = protocol
        digest = _text(row.get("input_digest"), f"{key}.input_digest")
        input_key = (identifier, condition)
        if input_key in input_digests and input_digests[input_key] != digest:
            raise AnalysisError(f"base/tuned input identity mismatch: {input_key}")
        input_digests[input_key] = digest
        condition_spec = by_id[identifier]["conditions"][condition]
        if isinstance(condition_spec, Mapping) and "input_digest" in condition_spec:
            if condition_spec["input_digest"] != digest:
                raise AnalysisError(f"input identity differs from frozen manifest: {input_key}")
        indexed[key] = row
    missing = [
        (record["id"], model, condition)
        for record in test_records for model in MODELS for condition in CONDITIONS
        if (record["id"], model, condition) not in indexed
    ]
    if missing:
        raise AnalysisError(f"incomplete prediction matrix: {len(missing)} missing cells; first={missing[0]}")
    if model_digests["base"] == model_digests["tuned"]:
        raise AnalysisError("base and tuned model state identities must differ")
    identity = {"protocol_digest": protocol_digest, "model_digests": model_digests}
    return test_records, indexed, identity


def _values(value: Any, name: str) -> list[str]:
    if isinstance(value, str):
        return [_text(value, name)]
    if not isinstance(value, (list, tuple)):
        raise AnalysisError(f"{name} must be a string or list of strings")
    return [_text(item, name) for item in value]


def _group_tokens(record: Mapping[str, Any]) -> tuple[set[str], bool]:
    """Recognize explicit source groups and repeated donor IDs conservatively."""
    tokens = {"source:" + record["source_group"]}
    donor_metadata_complete = True
    audit = record.get("audit", {})
    provenance = record.get("provenance", {})
    locations = [record, audit, provenance, *record["conditions"].values()]
    for location in locations:
        if not isinstance(location, Mapping):
            continue
        for field in ("donor_source_group", "donor_source_groups", "donor_group", "donor_groups"):
            if field in location:
                tokens.update("source:" + value for value in _values(location[field], field))
        for field in ("donor_id", "donor_ids"):
            if field in location:
                tokens.update("donor:" + value for value in _values(location[field], field))
        if "source_media_sha256" in location:
            tokens.add("media:" + _text(location["source_media_sha256"], "source_media_sha256"))
        for field in ("source_clip_fingerprints", "donor_source_clip_fingerprints"):
            if field in location:
                tokens.update("clip:" + value for value in _values(location[field], field))
        if "prepared_sha256" in location:
            tokens.update("frame:" + value for value in _values(location["prepared_sha256"], "prepared_sha256"))
        if "donor_sha256" in location:
            hashes = location["donor_sha256"]
            if not isinstance(hashes, Mapping):
                raise AnalysisError("donor_sha256 must map conditions to clip/frame hashes")
            for clips in hashes.values():
                if not isinstance(clips, Mapping):
                    raise AnalysisError("donor_sha256 condition must map clip indices to frame hashes")
                for values in clips.values():
                    tokens.update("frame:" + value for value in _values(values, "donor_sha256 frame hashes"))
    for name in ("neutral", "competing"):
        condition = record["conditions"][name]
        has_groups = any(
            location.get(field)
            for location in (record, audit, provenance, condition) if isinstance(location, Mapping)
            for field in ("donor_source_group", "donor_source_groups", "donor_group", "donor_groups")
        )
        if not has_groups:
            donor_metadata_complete = False
    return tokens, donor_metadata_complete


def _clusters(records: list[dict[str, Any]]) -> tuple[list[list[int]], dict[str, str], list[str]]:
    parent = list(range(len(records)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    token_owner: dict[str, int] = {}
    caveats: set[str] = set()
    for index, record in enumerate(records):
        tokens, donors_complete = _group_tokens(record)
        for token in sorted(tokens):
            if token in token_owner:
                a, b = root(index), root(token_owner[token])
                parent[max(a, b)] = min(a, b)
            else:
                token_owner[token] = index
        provenance_status = record.get(
            "provenance_status", record.get("audit", {}).get(
                "provenance_status", record.get("provenance", {}).get("provenance_status")
            )
        )
        verified = record.get("provenance_complete") is True or provenance_status in (
            "verified", "complete", "verified_complete"
        )
        if not verified:
            caveats.add("Some source provenance is unverified; supplied group labels do not establish independence.")
        if not donors_complete:
            caveats.add("Some donor source groups are unavailable; repeated donor IDs are merged when supplied, but hidden overlap may remain.")
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        grouped[root(index)].append(index)
    groups = list(grouped.values())
    assignments = {
        records[index]["id"]: f"cluster_{number:04d}"
        for number, members in enumerate(groups, 1) for index in members
    }
    if len(groups) < 2:
        caveats.add("Fewer than two source/donor clusters remain; bootstrap confidence intervals are unavailable.")
    elif len(groups) < 10:
        caveats.add("Few source/donor clusters remain; bootstrap interval coverage may be poor.")
    return groups, assignments, sorted(caveats)


def _effects(accuracies: Sequence[float]) -> dict[str, float]:
    bo, bn, bd, bt, to, tn, td, tt = accuracies
    return {
        "original_gain": to - bo,
        "neutral_gain": tn - bn,
        "competing_gain": td - bd,
        "text_only_gain": tt - bt,
        "base_semantic_distraction_penalty": bn - bd,
        "tuned_semantic_distraction_penalty": tn - td,
        "semantic_penalty_reduction": (bn - bd) - (tn - td),
        "base_original_to_competing_drop": bo - bd,
        "tuned_original_to_competing_drop": to - td,
        "base_visual_benefit": bo - bt,
        "tuned_visual_benefit": to - tt,
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def analyze(
    records: InputRows,
    predictions: InputRows,
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Validate complete saved predictions and compute paired, grouped estimates.

    Accuracies and effects are fractions, not percentages. The cluster bootstrap
    samples connected source/donor groups with replacement, keeps every model and
    condition together, and divides by the number of questions in each draw.
    """
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 1:
        raise AnalysisError("bootstrap_samples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise AnalysisError("seed must be an integer")
    test_records, indexed, identity = _validate(records, predictions)
    n = len(test_records)
    groups, assignments, caveats = _clusters(test_records)
    cell_order = [(model, condition) for model in MODELS for condition in CONDITIONS]
    observations = [
        [int(indexed[(record["id"], model, condition)]["prediction"] == record["answer"])
         for model, condition in cell_order]
        for record in test_records
    ]
    correct = [sum(row[column] for row in observations) for column in range(8)]
    accuracy_values = [value / n for value in correct]
    estimates = _effects(accuracy_values)
    samples: dict[str, list[float]] = {key: [] for key in estimates}
    accuracy_samples: list[list[float]] = [[] for _ in range(8)]
    if len(groups) >= 2:
        group_sums = [[sum(observations[index][column] for index in members) for column in range(8)] for members in groups]
        rng = random.Random(seed)
        for _ in range(bootstrap_samples):
            selected = [rng.randrange(len(groups)) for _ in groups]
            denominator = sum(len(groups[index]) for index in selected)
            accuracies = [sum(group_sums[index][column] for index in selected) / denominator for column in range(8)]
            for column, value in enumerate(accuracies):
                accuracy_samples[column].append(value)
            for key, value in _effects(accuracies).items():
                samples[key].append(value)

    def interval(values: Sequence[float]) -> list[float] | None:
        return [_percentile(values, 0.025), _percentile(values, 0.975)] if values else None

    accuracies: dict[str, dict[str, Any]] = {model: {} for model in MODELS}
    for column, (model, condition) in enumerate(cell_order):
        accuracies[model][condition] = {
            "correct": correct[column], "total": n, "accuracy": accuracy_values[column],
            "ci95": interval(accuracy_samples[column]),
        }
    effects = {key: {"estimate": value, "ci95": interval(samples[key])} for key, value in estimates.items()}
    paired_changes = {}
    for condition in CONDITIONS:
        corrected, regressed, unchanged_correct, unchanged_incorrect = [], [], [], []
        for record in test_records:
            identifier, answer = record["id"], record["answer"]
            base = indexed[(identifier, "base", condition)]["prediction"] == answer
            tuned = indexed[(identifier, "tuned", condition)]["prediction"] == answer
            bucket = corrected if tuned and not base else regressed if base and not tuned else unchanged_correct if base else unchanged_incorrect
            bucket.append(identifier)
        paired_changes[condition] = {
            "corrected": len(corrected), "regressed": len(regressed),
            "unchanged_correct": len(unchanged_correct), "unchanged_incorrect": len(unchanged_incorrect),
            "corrected_ids": corrected, "regressed_ids": regressed,
        }
    per_question = []
    for record in test_records:
        identifier = record["id"]
        per_question.append({
            "id": identifier, "source_group": record["source_group"], "cluster": assignments[identifier],
            "answer": record["answer"],
            "conditions": {
                condition: {
                    "base_prediction": indexed[(identifier, "base", condition)]["prediction"],
                    "tuned_prediction": indexed[(identifier, "tuned", condition)]["prediction"],
                    "base_correct": indexed[(identifier, "base", condition)]["prediction"] == record["answer"],
                    "tuned_correct": indexed[(identifier, "tuned", condition)]["prediction"] == record["answer"],
                }
                for condition in CONDITIONS
            },
        })
    answer_counts = {label: sum(record["answer"] == label for record in test_records) for label in LABELS}
    primary = estimates["competing_gain"]
    direction = "higher" if primary > 0 else "lower" if primary < 0 else "equal"
    review_counts = {kind: sum(record.get("audit", {}).get("reviewer_type") == kind for record in test_records)
                     for kind in ("human", "ai")}
    canonical_inputs = {
        "records": test_records,
        "predictions": [indexed[key] for key in sorted(indexed)],
    }
    inputs_digest = hashlib.sha256(json.dumps(canonical_inputs, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {
        "schema_version": 1,
        "unit": "accuracy_fraction",
        "cohort": {"test_questions": n, "prediction_cells": len(indexed), "source_groups": len({record["source_group"] for record in test_records}), "source_donor_clusters": len(groups), "answer_label_counts": answer_counts, "reviewer_types": review_counts},
        "identity": {**identity, "analysis_inputs_digest": inputs_digest},
        "accuracies": accuracies,
        "effects": effects,
        "paired_changes": paired_changes,
        "per_question": per_question,
        "bootstrap": {
            "method": "paired connected-source/donor cluster percentile bootstrap",
            "estimand": "question-weighted accuracy and paired differences",
            "requested_samples": bootstrap_samples,
            "completed_samples": bootstrap_samples if len(groups) >= 2 else 0,
            "seed": seed, "confidence_level": 0.95,
            "conditional_on_available_grouping": True,
            "group_assignments": assignments,
            "caveats": caveats,
        },
        "interpretation": {
            "primary_effect": "competing_gain",
            "observed_competing_accuracy": direction,
            "descriptive_promising_pattern": primary > 0 and estimates["original_gain"] >= 0 and estimates["neutral_gain"] >= 0,
            "statement": ("This is a one-question engineering feasibility check; it cannot estimate model quality. " if n == 1 else "") + f"Tuned competing-condition accuracy is {direction} on this frozen cohort. This is an exploratory observed comparison; it does not establish a memory mechanism, population improvement, or noninferiority.",
            "limitations": [
                "Intervals are conditional on the available source/donor grouping; independence is not established by the group labels.",
                "One training run leaves variation across training seeds unmeasured.",
                "This custom subset and edited-condition diagnostic is not a full official SIS-Bench score.",
                "Unknown upstream model-training contamination and preparation uncertainty limit generalization claims.",
                f"Recorded test reviewers: {review_counts['human']} human, {review_counts['ai']} AI. AI review is not human validation.",
            ],
        },
    }


def render_markdown(metrics: Mapping[str, Any]) -> str:
    """Render a deterministic report from the saved metrics object alone."""
    def percent(value: float) -> str:
        return f"{100 * value:.2f}"

    def ci(value: Sequence[float] | None) -> str:
        return f"[{percent(value[0])}, {percent(value[1])}]" if value is not None else "unavailable"

    cohort = metrics["cohort"]
    lines = [
        "# Earlier-clip QA: paired pilot results", "",
        metrics["interpretation"]["statement"], "",
        f"The complete matrix contains {cohort['test_questions']} questions and {cohort['prediction_cells']} predictions across {cohort['source_groups']} supplied source groups and {cohort['source_donor_clusters']} connected source/donor clusters.", "",
        "## Accuracies", "",
        "| Condition | Base correct/total | Base accuracy (%) [95% CI] | Tuned correct/total | Tuned accuracy (%) [95% CI] |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        base, tuned = (metrics["accuracies"][model][condition] for model in MODELS)
        lines.append(f"| {condition} | {base['correct']}/{base['total']} | {percent(base['accuracy'])} {ci(base['ci95'])} | {tuned['correct']}/{tuned['total']} | {percent(tuned['accuracy'])} {ci(tuned['ci95'])} |")
    lines.extend(["", "## Paired effects", "", "All effects and interval bounds below are percentage points. The primary effect is competing_gain.", "", "| Effect | Estimate (pp) | 95% CI (pp) |", "|---|---:|---:|"])
    for name in _effects([0.0] * 8):
        result = metrics["effects"][name]
        lines.append(f"| {name} | {percent(result['estimate'])} | {ci(result['ci95'])} |")
    lines.extend(["", "## Corrections and regressions", "", "| Condition | Corrected | Regressed | Unchanged correct | Unchanged incorrect |", "|---|---:|---:|---:|---:|"])
    for name in CONDITIONS:
        result = metrics["paired_changes"][name]
        lines.append(f"| {name} | {result['corrected']} | {result['regressed']} | {result['unchanged_correct']} | {result['unchanged_incorrect']} |")
    bootstrap = metrics["bootstrap"]
    lines.extend(["", "## Uncertainty and interpretation", "", f"Paired source/donor-cluster percentile bootstrap: {bootstrap['completed_samples']} resamples, seed {bootstrap['seed']}, question-weighted estimates. Every resample retains all conditions and model predictions together.", "", "These intervals are conditional on the available provenance groups. No independence, causal mechanism, maintained-performance, or publication-level claim follows from the intervals alone.", ""])
    for caveat in [*bootstrap["caveats"], *metrics["interpretation"]["limitations"]]:
        lines.append(f"- {caveat}")
    primary = metrics["effects"]["competing_gain"]
    if primary["ci95"] is not None and primary["ci95"][0] <= 0 <= primary["ci95"][1]:
        lines.extend(["", "The conditional primary-effect interval includes zero; the observed difference remains uncertain under this resampling scheme."])
    pattern = metrics["interpretation"]["descriptive_promising_pattern"]
    lines.extend(["", "The predefined descriptive promising pattern is " + ("met" if pattern else "not met") + ": higher observed competing accuracy with equal or higher observed original and neutral accuracy. This is a descriptive pilot criterion, not proof of improvement or noninferiority.", "", "## Reproducibility", "", f"- Protocol digest: `{metrics['identity']['protocol_digest']}`", f"- Base model digest: `{metrics['identity']['model_digests']['base']}`", f"- Tuned model digest: `{metrics['identity']['model_digests']['tuned']}`", f"- Analysis input digest: `{metrics['identity']['analysis_inputs_digest']}`", "", "Question-level paired outcomes, corrected/regressed IDs, answer-label counts, cluster assignments, and all numerical estimates are retained in metrics.json.", ""])
    return "\n".join(lines)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = stream.name
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def write_report(
    records: InputRows,
    predictions: InputRows,
    output_dir: str | Path,
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Produce metrics.json and report.md after complete-matrix validation."""
    metrics = analyze(records, predictions, bootstrap_samples=bootstrap_samples, seed=seed)
    output = Path(output_dir)
    _atomic_write(output / "metrics.json", json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _atomic_write(output / "report.md", render_markdown(metrics))
    return metrics
