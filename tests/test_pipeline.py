"""Synthetic software fixtures for the offline pipeline; no model or GPU runs."""

from copy import deepcopy
import json
import sys

import pytest
from PIL import Image

from ecqa.analysis import AnalysisError, render_markdown, write_report
from ecqa.artifacts import (
    digest, file_digest, freeze, verify_freeze, verify_report_inputs,
)
from ecqa.evaluate import evaluate


CONDITIONS = ("original", "neutral", "competing", "text_only")


@pytest.fixture
def synthetic_cohort(tmp_path):
    """Approved metadata describes generated fixture pixels, not audited real data."""
    root = tmp_path / "data"
    root.mkdir()
    records = []
    for number, split in enumerate(("train", "val", "test", "test")):
        identifier = f"fixture-{number}"
        directory = root / identifier
        directory.mkdir()
        originals = []
        for position in range(8):
            path = directory / f"original-{position}.png"
            Image.new("RGB", (128, 128), (number * 40 + position, 20, 30)).save(path)
            originals.append(path.relative_to(root).as_posix())
        clip_ranges = [[0, 4], [4, 8]]
        conditions = {
            "original": {"frames": originals, "clip_ranges": deepcopy(clip_ranges), "fps": 2.0},
            "text_only": {"frames": [], "clip_ranges": deepcopy(clip_ranges), "fps": 2.0},
        }
        audit = {
            "status": "approved", "reviewer": "synthetic software fixture generator", "reviewer_type": "ai",
            "provenance_status": "verified", "boundaries_verified": True, "evidence_verified": True,
            "answer_verified": True, "viewpoint_verified": True,
            "source_clip_ranges": [[0, 20], [20, 40]], "target_clip": 1,
            "evidence_frame_range": [4, 17], "viewpoint": "synthetic image coordinates",
        }
        if split == "test":
            for condition, color in (("neutral", 100), ("competing", 200)):
                frames = originals[:4]
                for position in range(4, 8):
                    path = directory / f"{condition}-{position}.png"
                    Image.new("RGB", (128, 128), (number * 40 + position, color, 30)).save(path)
                    frames.append(path.relative_to(root).as_posix())
                conditions[condition] = {"frames": frames, "clip_ranges": deepcopy(clip_ranges), "fps": 2.0}
            audit.update({
                "replacement_clip_indices": [1], "replacement_frame_indices": [4, 5, 6, 7],
                "donor_source_groups": [f"fixture-donor-{number}-neutral", f"fixture-donor-{number}-competing"],
                "donor_source_clip_fingerprints": [
                    digest({"donor_source": number, "condition": condition})
                    for condition in ("neutral", "competing")
                ],
                "condition_review": {
                    condition: {"answer_preserved": True, "role_verified": True}
                    for condition in ("neutral", "competing")
                },
            })
        records.append({
            "id": identifier, "split": split, "source_group": f"fixture-source-{number}",
            "question": "Synthetic fixture question about the first clip.",
            "options": {"A": "fixture answer", "B": "fixture distractor", "C": "third option", "D": "fourth option"},
            "answer": "A", "audit": audit, "conditions": conditions,
            "provenance": {
                "dataset": "synthetic_software_fixture", "decoded_frame_count": 40,
                "frame_indices": [4, 8, 12, 16, 24, 28, 32, 36],
                "source_media_sha256": digest({"generated_source": number}),
                "source_clip_fingerprint_method": "rgb24-all-frames-sha256-v1",
                "source_clip_fingerprints": [digest({"generated_source": number, "clip": i}) for i in range(2)],
                "prepared_sha256": [file_digest(root / name) for name in originals],
            },
        })
    config = {
        "schema_version": 1,
        "model": {"id": "synthetic-test-runtime", "revision": "a" * 40, "attention_backend": "sdpa"},
        "preprocessing": {"frames": 8, "synthetic_fps": 2.0, "max_sequence_tokens": 4096},
        "training": {
            "epochs": 1, "seed": 42, "gradient_accumulation_steps": 4, "max_cumulative_seconds": 7200,
            "learning_rate": 1e-5, "warmup_ratio": 0.0, "weight_decay": 0.01, "max_grad_norm": 1.0,
            "checkpoint_every_optimizer_steps": 25, "checkpoint_selection": "end_of_run",
            "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.05,
        },
        "analysis": {"bootstrap_samples": 50, "seed": 42},
        "exposure_exclusions": {"ids": [], "source_clip_fingerprints": [], "source_groups": [], "media_sha256": []},
    }
    return root, records, config


class FakeRuntime:
    """Deterministic mock predictions test plumbing, not model capability."""

    def __init__(self, data_root, *, tuned=False, fail_once=None):
        self.data_root = data_root
        self.tuned = tuned
        self.fail_once = fail_once
        self.calls = []

    def scores(self, record, condition):
        key = (record["id"], condition)
        self.calls.append(key)
        if key == self.fail_once:
            self.fail_once = None
            raise RuntimeError("synthetic inference interruption")
        selected = "A" if self.tuned or record["id"] == "fixture-2" else "B"
        return {label: -0.1 if label == selected else -3.0 for label in "ABCD"}


def run_mock(runtime, records, output, *, model="base", protocol="fixture-protocol", model_digest=None):
    return evaluate(
        runtime, records, output, model_name=model,
        model_digest=model_digest or digest({"synthetic_model": model}), protocol_digest=protocol,
    )


def saved_rows(directory):
    return [(path, json.loads(path.read_text(encoding="utf-8"))) for path in sorted(directory.glob("*.json"))]


def test_frozen_pipeline_produces_complete_paired_report_and_regenerates(synthetic_cohort, tmp_path, monkeypatch):
    root, records, config = synthetic_cohort
    config["analysis"] = {"bootstrap_samples": 37, "seed": 17}
    protocol_path = tmp_path / "protocol.json"
    protocol = freeze(config, records, root, protocol_path)
    assert verify_freeze(protocol_path, config, records, root) == protocol
    base, tuned = FakeRuntime(root), FakeRuntime(root, tuned=True)
    rows = run_mock(base, records, tmp_path / "base", protocol=protocol["protocol_digest"],
                    model_digest=digest({"model": config["model"]}))
    rows += run_mock(tuned, records, tmp_path / "tuned", model="tuned", protocol=protocol["protocol_digest"])
    (tmp_path / "selected-model.json").write_text(json.dumps({
        "protocol_digest": protocol["protocol_digest"], "rule": "end_of_run",
        "model_digest": digest({"synthetic_model": "tuned"}),
    }), encoding="utf-8")
    assert len(base.calls) == len(tuned.calls) == 8
    assert len(rows) == 16
    assert {row["condition"] for row in rows} == set(CONDITIONS)
    verify_report_inputs(protocol_path, records, rows)
    result = write_report(records, rows, tmp_path / "report", bootstrap_samples=37, seed=17)
    assert result["accuracies"]["base"]["original"]["correct"] == 1
    assert result["accuracies"]["tuned"]["competing"]["correct"] == 2
    assert result["effects"]["competing_gain"]["estimate"] == 0.5
    saved = json.loads((tmp_path / "report" / "metrics.json").read_text(encoding="utf-8"))
    assert render_markdown(saved) == (tmp_path / "report" / "report.md").read_text(encoding="utf-8")
    from ecqa.cli import main
    manifest_path, prediction_path = tmp_path / "manifest.jsonl", tmp_path / "predictions.jsonl"
    manifest_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    prediction_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["ecqa", "report", "--protocol", str(protocol_path),
                                    "--manifest", str(manifest_path), "--predictions", str(prediction_path),
                                    "--output", str(tmp_path / "cli-report")])
    main()
    cli_saved = json.loads((tmp_path / "cli-report" / "metrics.json").read_text(encoding="utf-8"))
    assert cli_saved == saved
    assert cli_saved["bootstrap"]["requested_samples"] == 37 and cli_saved["bootstrap"]["seed"] == 17
    changed = deepcopy(records)
    changed[-1]["answer"] = "B"
    with pytest.raises(ValueError, match="manifest differs"):
        verify_report_inputs(protocol_path, changed, rows)
    changed_rows = deepcopy(rows)
    changed_rows[0]["input_digest"] = "different-prepared-input"
    with pytest.raises(ValueError, match="input differs"):
        verify_report_inputs(protocol_path, records, changed_rows)


def test_prediction_resume_reuses_only_matching_cells_and_recomputes_missing(synthetic_cohort, tmp_path):
    root, records, _ = synthetic_cohort
    runtime = FakeRuntime(root)
    output = tmp_path / "predictions"
    first = run_mock(runtime, records, output)
    assert len(runtime.calls) == 8
    assert len(saved_rows(output)) == 8
    assert not list(output.glob("*.tmp"))
    assert run_mock(runtime, records, output) == first
    assert len(runtime.calls) == 8
    missing_path, missing_row = saved_rows(output)[0]
    missing_path.unlink()
    resumed = run_mock(runtime, records, output)
    assert len(resumed) == 8
    assert len(runtime.calls) == 9
    assert runtime.calls[-1] == (missing_row["id"], missing_row["condition"])
    assert not list(output.glob("*.tmp"))
    new_state = run_mock(runtime, records, output, model_digest="different-model-state")
    assert len(runtime.calls) == 17
    assert len(saved_rows(output)) == 16
    assert all(row["model_digest"] == "different-model-state" for row in new_state)


def test_cached_identity_tampering_is_rejected(synthetic_cohort, tmp_path):
    root, records, _ = synthetic_cohort
    runtime = FakeRuntime(root)
    output = tmp_path / "predictions"
    run_mock(runtime, records, output)
    path, row = saved_rows(output)[0]
    row["protocol_digest"] = "wrong-cached-protocol"
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="Prediction identity mismatch"):
        run_mock(runtime, records, output)
    assert len(runtime.calls) == 8
    row["protocol_digest"] = "fixture-protocol"
    row["prediction"] = "D"
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="Cached prediction disagrees"):
        run_mock(runtime, records, output)
    assert len(runtime.calls) == 8


def test_inference_failure_is_persisted_retried_and_cannot_be_a_final_report(synthetic_cohort, tmp_path):
    root, records, _ = synthetic_cohort
    runtime = FakeRuntime(root, fail_once=("fixture-2", "neutral"))
    output = tmp_path / "predictions"
    with pytest.raises(RuntimeError, match="synthetic inference interruption"):
        run_mock(runtime, records, output)
    partial = [row for _, row in saved_rows(output)]
    assert len(partial) == 2
    assert sorted(row["status"] for row in partial) == ["error", "ok"]
    error = next(row for row in partial if row["status"] == "error")
    assert error["condition"] == "neutral" and error["error"] == "RuntimeError"
    assert not list(output.glob("*.tmp"))
    with pytest.raises(AnalysisError):
        write_report(records, partial, tmp_path / "invalid-report", bootstrap_samples=50)
    assert not (tmp_path / "invalid-report").exists()
    completed = run_mock(runtime, records, output)
    assert len(completed) == 8
    assert all(row["status"] == "ok" for row in completed)
    assert len(runtime.calls) == 9  # One cached success; the failed cell is retried.
    with pytest.raises(AnalysisError, match="incomplete prediction matrix"):
        write_report(records, completed, tmp_path / "base-only-report", bootstrap_samples=50)
    assert not (tmp_path / "base-only-report").exists()


def test_freeze_detects_modified_pixels_and_known_donor_leakage(synthetic_cohort, tmp_path):
    root, records, config = synthetic_cohort
    protocol_path = tmp_path / "protocol.json"
    freeze(config, records, root, protocol_path)
    frame = root / records[-1]["conditions"]["original"]["frames"][0]
    original_bytes = frame.read_bytes()
    Image.new("RGB", (128, 128), (255, 255, 255)).save(frame)
    with pytest.raises(ValueError, match="Prepared input changed"):
        verify_freeze(protocol_path, config, records, root)
    frame.write_bytes(original_bytes)
    verify_freeze(protocol_path, config, records, root)
    leaked = deepcopy(records)
    leaked[-1]["audit"]["donor_source_groups"] = [records[0]["source_group"]]
    with pytest.raises(ValueError, match="overlap"):
        freeze(config, leaked, root, tmp_path / "leaked-protocol.json")
    assert not (tmp_path / "leaked-protocol.json").exists()


@pytest.mark.parametrize("role", ["target", "donor"])
def test_freeze_catches_constituent_leakage_despite_different_group_labels(synthetic_cohort, tmp_path, role):
    root, records, config = synthetic_cohort
    shared = records[0]["provenance"]["source_clip_fingerprints"][0]
    if role == "target":
        records[-1]["provenance"]["source_clip_fingerprints"][0] = shared
    else:
        records[-1]["audit"]["donor_source_clip_fingerprints"][0] = shared
    assert records[0]["source_group"] != records[-1]["source_group"]
    with pytest.raises(ValueError, match="overlap across splits"):
        freeze(config, records, root, tmp_path / "protocol.json")


@pytest.mark.parametrize("role", ["id", "target_clip", "donor_clip"])
def test_freeze_excludes_previously_exposed_test_targets_and_donors(synthetic_cohort, tmp_path, role):
    root, records, config = synthetic_cohort
    if role == "id":
        config["exposure_exclusions"]["ids"] = [records[-1]["id"]]
    else:
        clip = (records[-1]["provenance"]["source_clip_fingerprints"][0] if role == "target_clip"
                else records[-1]["audit"]["donor_source_clip_fingerprints"][0])
        config["exposure_exclusions"]["source_clip_fingerprints"] = [clip]
    with pytest.raises(ValueError, match="previously exposed"):
        freeze(config, records, root, tmp_path / "protocol.json")


def test_freeze_refuses_perceptual_identity_and_identical_control_pixels(synthetic_cohort, tmp_path):
    root, records, config = synthetic_cohort
    records[-1]["provenance"]["source_clip_fingerprint_method"] = "perceptual-thumbnail-hash"
    with pytest.raises(ValueError, match="exact all-frame constituent fingerprint"):
        freeze(config, records, root, tmp_path / "protocol.json")
    records[-1]["provenance"]["source_clip_fingerprint_method"] = "rgb24-all-frames-sha256-v1"
    records[-1]["conditions"]["competing"] = deepcopy(records[-1]["conditions"]["neutral"])
    with pytest.raises(ValueError, match="pixel-identical"):
        freeze(config, records, root, tmp_path / "protocol.json")


def test_new_protocol_cannot_reuse_another_runs_selection_directory(synthetic_cohort, tmp_path):
    root, records, config = synthetic_cohort
    (tmp_path / "selected-model.json").write_text(json.dumps({"protocol_digest": "another-run"}))
    with pytest.raises(ValueError, match="selection for another protocol"):
        freeze(config, records, root, tmp_path / "protocol.json")


@pytest.mark.parametrize("scores", [{"A": float("nan"), "B": -2, "C": -3, "D": -4}, {"A": -1, "B": -2}])
def test_invalid_scorer_outputs_are_persisted_as_errors(synthetic_cohort, tmp_path, scores, monkeypatch):
    root, records, _ = synthetic_cohort
    runtime = FakeRuntime(root)
    monkeypatch.setattr(runtime, "scores", lambda record, condition: scores)
    output = tmp_path / "predictions"
    with pytest.raises(ValueError, match="Fixed-choice scores"):
        run_mock(runtime, records, output)
    rows = saved_rows(output)
    assert len(rows) == 1 and rows[0][1]["status"] == "error"
    assert not list(output.glob("*.tmp"))


def visual_decision(record):
    from scripts.review_full_study import review_identity
    return {
        "id": record["id"], "review_identity": review_identity(record), "decision": "accept",
        "reviewer": "synthetic fixture reviewer", "reviewer_type": "ai",
        "reason": "Software fixture pixels match the stated synthetic answer.",
        "viewpoint": "synthetic image coordinates", "boundaries_verified": True,
        "evidence_verified": True, "answer_verified": True, "viewpoint_verified": True,
    }


def test_visual_review_binds_temporal_facts_but_allows_later_grouping(synthetic_cohort):
    from scripts.review_full_study import apply_reviews, review_identity
    _, records, _ = synthetic_cohort
    record = deepcopy(records[0])
    decision = visual_decision(record)
    record.update(source_group="later-confirmed-component", split="val")
    assert review_identity(record) == decision["review_identity"]
    accepted, rejected = apply_reviews([record], [decision])
    assert len(accepted) == 1 and not rejected
    assert accepted[0]["audit"]["human_validated"] is False
    assert review_identity(accepted[0]) == decision["review_identity"]
    record["audit"]["source_clip_ranges"] = [[0, 19], [19, 40]]
    with pytest.raises(ValueError, match="Reviewed candidate changed"):
        apply_reviews([record], [decision])


def test_edited_visual_review_requires_fresh_condition_findings_and_binds_donors(synthetic_cohort):
    from scripts.review_full_study import apply_reviews
    _, records, _ = synthetic_cohort
    record = deepcopy(records[-1])
    decision = visual_decision(record)
    # Existing audit flags cannot substitute for the current reviewer's decision.
    with pytest.raises(ValueError, match="neutral condition_review"):
        apply_reviews([record], [decision])
    decision["condition_review"] = {
        name: {"answer_preserved": True, "role_verified": True,
               "reason": f"Synthetic {name} fixture has the declared role and unchanged target."}
        for name in ("neutral", "competing")
    }
    accepted, rejected = apply_reviews([record], [decision])
    assert len(accepted) == 1 and not rejected
    assert accepted[0]["audit"]["condition_review"] == decision["condition_review"]
    record["audit"]["donor_source_clip_fingerprints"][0] = digest("different-donor")
    with pytest.raises(ValueError, match="Reviewed candidate changed"):
        apply_reviews([record], [decision])
