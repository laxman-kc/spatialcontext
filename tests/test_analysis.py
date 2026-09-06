"""Hand-computed checks for the paired analysis and its refusal conditions."""

from copy import deepcopy
import json

import pytest

from ecqa.analysis import AnalysisError, CONDITIONS, analyze, render_markdown, write_report


def make_records(groups=("g1", "g2", "g3", "g4")):
    return [
        {
            "id": f"q{index}",
            "split": "test",
            "source_group": group,
            "provenance_status": "verified",
            "answer": "A",
            "conditions": {
                condition: {
                    "input_digest": f"input-q{index}-{condition}",
                    **({"donor_source_groups": [f"donor-{group}-{condition}"]}
                       if condition in ("neutral", "competing") else {}),
                }
                for condition in CONDITIONS
            },
        }
        for index, group in enumerate(groups)
    ]


def make_predictions(records, correct=None):
    if correct is None:
        correct = {
            "base": {condition: {"q0"} for condition in CONDITIONS},
            "tuned": {condition: {"q0", "q1"} for condition in CONDITIONS},
        }
    rows = []
    for record in records:
        for model in ("base", "tuned"):
            for condition in CONDITIONS:
                prediction = "A" if record["id"] in correct[model][condition] else "B"
                rows.append({
                    "id": record["id"],
                    "condition": condition,
                    "model": model,
                    "input_digest": record["conditions"][condition]["input_digest"],
                    "model_digest": f"model-{model}",
                    "protocol_digest": "protocol-v1",
                    "scores": {label: -0.1 if label == prediction else -2.0 for label in "ABCD"},
                    "prediction": prediction,
                    "status": "ok",
                    "seconds": 0.25,
                })
    return rows


def test_hand_computed_accuracy_effects_and_paired_changes():
    records = make_records()
    correct = {
        "base": {
            "original": {"q0", "q1"}, "neutral": {"q0", "q1", "q2"},
            "competing": {"q0"}, "text_only": {"q0", "q1"},
        },
        "tuned": {
            "original": {"q0", "q1", "q2"}, "neutral": {"q0", "q1", "q2"},
            "competing": {"q1", "q2"}, "text_only": {"q0"},
        },
    }
    result = analyze(records, make_predictions(records, correct), bootstrap_samples=100)
    assert result["cohort"]["prediction_cells"] == 32
    assert result["accuracies"]["base"]["original"]["correct"] == 2
    assert result["accuracies"]["tuned"]["competing"]["accuracy"] == 0.5
    expected = {
        "original_gain": 0.25, "neutral_gain": 0.0, "competing_gain": 0.25,
        "text_only_gain": -0.25, "base_semantic_distraction_penalty": 0.5,
        "tuned_semantic_distraction_penalty": 0.25, "semantic_penalty_reduction": 0.25,
        "base_original_to_competing_drop": 0.25, "tuned_original_to_competing_drop": 0.25,
        "base_visual_benefit": 0.0, "tuned_visual_benefit": 0.5,
    }
    assert {name: effect["estimate"] for name, effect in result["effects"].items()} == expected
    changes = result["paired_changes"]["competing"]
    assert changes["corrected_ids"] == ["q1", "q2"]
    assert changes["regressed_ids"] == ["q0"]
    assert changes["unchanged_incorrect"] == 1
    assert result["interpretation"]["descriptive_promising_pattern"] is True
    assert "does not establish" in result["interpretation"]["statement"]


def test_question_weighted_group_bootstrap_preserves_all_pairs():
    records = make_records(("large", "large", "large", "small"))
    correct = {
        "base": {condition: {"q3"} for condition in CONDITIONS},
        "tuned": {condition: {"q0", "q1", "q2"} for condition in CONDITIONS},
    }
    result = analyze(records, make_predictions(records, correct), bootstrap_samples=1000, seed=42)
    assert result["cohort"]["source_donor_clusters"] == 2
    # Question-weighted difference is .75 - .25 = .5, not group-weighted zero.
    assert result["effects"]["competing_gain"]["estimate"] == 0.5
    # Two cluster draws yield effects -1, .5, or 1; both tails are well represented.
    assert result["effects"]["competing_gain"]["ci95"] == [-1.0, 1.0]
    # Conditions have identical outcomes, so paired penalty contrasts stay zero.
    assert result["effects"]["semantic_penalty_reduction"]["ci95"] == [0.0, 0.0]
    assert result["effects"]["base_semantic_distraction_penalty"]["ci95"] == [0.0, 0.0]
    assert result == analyze(records, make_predictions(records, correct), bootstrap_samples=1000, seed=42)


def test_shared_donor_merges_source_groups_and_suppresses_one_cluster_ci():
    records = make_records(("g1", "g2"))
    for record in records:
        record["conditions"]["neutral"]["donor_source_groups"] = ["shared-donor"]
    result = analyze(records, make_predictions(records), bootstrap_samples=50)
    assert result["cohort"]["source_groups"] == 2
    assert result["cohort"]["source_donor_clusters"] == 1
    assert result["effects"]["competing_gain"]["ci95"] is None
    assert result["bootstrap"]["completed_samples"] == 0
    assert "unavailable" in render_markdown(result)


def test_source_donor_overlap_and_transitive_overlap_merge():
    records = make_records(("g1", "g2", "g3"))
    records[0]["conditions"]["neutral"]["donor_source_groups"] = ["g2"]
    records[1]["conditions"]["competing"]["donor_source_groups"] = ["g3"]
    result = analyze(records, make_predictions(records), bootstrap_samples=50)
    assert result["cohort"]["source_donor_clusters"] == 1


def test_canonical_nested_audit_donor_groups_and_hashes_are_merged():
    records = make_records(("g1", "g2", "g3"))
    for record in records:
        record.pop("provenance_status")
        record["audit"] = {"status": "approved", "provenance_status": "verified"}
        for condition in ("neutral", "competing"):
            record["conditions"][condition].pop("donor_source_groups")
    records[0]["audit"]["donor_source_groups"] = ["g2"]
    records[1]["audit"]["donor_sha256"] = {"neutral": {"2": ["same-frame-hash"]}}
    records[2]["audit"]["donor_sha256"] = {"competing": {"1": ["same-frame-hash"]}}
    result = analyze(records, make_predictions(records), bootstrap_samples=50)
    assert result["cohort"]["source_donor_clusters"] == 1
    assert not any("source provenance is unverified" in item for item in result["bootstrap"]["caveats"])


def test_unapproved_audit_cannot_produce_final_report():
    records = make_records()
    records[0]["audit"] = {"status": "needs_review"}
    with pytest.raises(AnalysisError, match="approved audit"):
        analyze(records, make_predictions(records), bootstrap_samples=10)


def test_exact_constituent_overlap_merges_original_and_donor_clusters():
    records = make_records(("g1", "g2", "g3"))
    records[0]["provenance"] = {"source_clip_fingerprints": ["clip-a"]}
    records[1]["provenance"] = {"source_clip_fingerprints": ["clip-b"]}
    records[1]["audit"] = {
        "status": "approved", "donor_source_clip_fingerprints": ["clip-a"]
    }
    records[2]["provenance"] = {"source_clip_fingerprints": ["clip-b"]}
    result = analyze(records, make_predictions(records), bootstrap_samples=50)
    assert result["cohort"]["source_donor_clusters"] == 1
    assert result["effects"]["competing_gain"]["ci95"] is None


def test_unknown_provenance_and_repeated_donor_ids_are_explicit():
    records = make_records(("g1", "g2"))
    for record in records:
        record.pop("provenance_status")
        for condition in ("neutral", "competing"):
            record["conditions"][condition].pop("donor_source_groups")
        record["conditions"]["neutral"]["donor_id"] = "shared-unresolved-donor"
    result = analyze(records, make_predictions(records), bootstrap_samples=50)
    assert result["cohort"]["source_donor_clusters"] == 1
    assert result["bootstrap"]["conditional_on_available_grouping"] is True
    assert any("unverified" in caveat for caveat in result["bootstrap"]["caveats"])
    assert any("hidden overlap" in caveat for caveat in result["bootstrap"]["caveats"])


def test_neutral_regression_prevents_promising_pattern():
    records = make_records()
    correct = {
        "base": {condition: {"q0"} for condition in CONDITIONS},
        "tuned": {condition: {"q0", "q1"} for condition in CONDITIONS},
    }
    correct["tuned"]["neutral"] = set()
    result = analyze(records, make_predictions(records, correct), bootstrap_samples=100)
    assert result["effects"]["competing_gain"]["estimate"] > 0
    assert result["interpretation"]["descriptive_promising_pattern"] is False


@pytest.mark.parametrize("mutation,match", [
    (lambda rows: rows.pop(), "incomplete prediction matrix"),
    (lambda rows: rows.append(deepcopy(rows[0])), "duplicate prediction"),
    (lambda rows: rows[0].update(status="failed"), "did not succeed"),
    (lambda rows: rows[0]["scores"].update(A=float("nan")), "finite number"),
    (lambda rows: rows[0]["scores"].update(A=float("inf")), "finite number"),
    (lambda rows: rows[0]["scores"].update(A=True), "finite number"),
    (lambda rows: rows[0]["scores"].pop("D"), "scores must contain"),
    (lambda rows: rows[0].update(prediction="D"), "prediction disagrees"),
    (lambda rows: rows[0].update(seconds=-1), "seconds cannot be negative"),
    (lambda rows: rows[0].update(protocol_digest="different"), "inconsistent protocol"),
    (lambda rows: rows[0].update(model_digest="different"), "inconsistent model identity"),
    (lambda rows: rows[0].update(input_digest="different"), "frozen manifest"),
    (lambda rows: rows[0].update(id="outside"), "outside the test cohort"),
    (lambda rows: rows[0].update(condition="unsupported"), "invalid model or condition"),
])
def test_invalid_prediction_matrix_is_rejected(mutation, match):
    records = make_records()
    predictions = make_predictions(records)
    mutation(predictions)
    with pytest.raises(AnalysisError, match=match):
        analyze(records, predictions, bootstrap_samples=10)


def test_base_tuned_input_mismatch_without_manifest_digest_is_rejected():
    records = make_records()
    predictions = make_predictions(records)
    for record in records:
        for condition in CONDITIONS:
            record["conditions"][condition].pop("input_digest")
    predictions[4]["input_digest"] = "changed-tuned-input"
    with pytest.raises(AnalysisError, match="input identity mismatch"):
        analyze(records, predictions, bootstrap_samples=10)


def test_same_model_state_for_both_models_is_rejected():
    records = make_records()
    predictions = make_predictions(records)
    for row in predictions:
        row["model_digest"] = "same-state"
    with pytest.raises(AnalysisError, match="must differ"):
        analyze(records, predictions, bootstrap_samples=10)


def test_duplicate_manifest_id_is_rejected_even_outside_test():
    records = make_records()
    predictions = make_predictions(records)
    records.append({"id": "q0", "split": "train"})
    with pytest.raises(AnalysisError, match="duplicate record id"):
        analyze(records, predictions, bootstrap_samples=10)


def test_full_manifest_selects_test_and_refuses_extra_predictions():
    records = make_records()
    predictions = make_predictions(records)
    records.append({"id": "train-example", "split": "train"})
    result = analyze(records, predictions, bootstrap_samples=10)
    assert result["cohort"]["test_questions"] == 4
    predictions.append({**predictions[0], "id": "train-example"})
    with pytest.raises(AnalysisError, match="outside the test cohort"):
        analyze(records, predictions, bootstrap_samples=10)


def test_saved_jsonl_report_can_be_regenerated_without_inference(tmp_path):
    records = make_records()
    predictions = make_predictions(records)
    records_path, predictions_path = tmp_path / "records.jsonl", tmp_path / "predictions.jsonl"
    records_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    predictions_path.write_text("\n".join(json.dumps(row) for row in predictions) + "\n", encoding="utf-8")
    destination = tmp_path / "report"
    result = write_report(records_path, predictions_path, destination, bootstrap_samples=100)
    saved = json.loads((destination / "metrics.json").read_text(encoding="utf-8"))
    assert saved == result
    assert (destination / "report.md").read_text(encoding="utf-8") == render_markdown(saved)
    first = (destination / "report.md").read_bytes()
    write_report(records_path, predictions_path, destination, bootstrap_samples=100)
    assert (destination / "report.md").read_bytes() == first


def test_incomplete_results_do_not_create_report_files(tmp_path):
    records = make_records()
    predictions = make_predictions(records)[:-1]
    output = tmp_path / "invalid-report"
    with pytest.raises(AnalysisError, match="incomplete prediction matrix"):
        write_report(records, predictions, output, bootstrap_samples=10)
    assert not output.exists()
