"""CPU-only integrity checks for frozen reporting and checkpoint files."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from ecqa.artifacts import digest, file_digest, load_config, verify_report_inputs
from ecqa.train import verify_checkpoint


def frozen_report_fixture(tmp_path):
    records = [
        {"id": "train", "split": "train", "answer": "C", "source_group": "training-group"},
        {"id": "test", "split": "test", "answer": "A", "source_group": "test-group"},
    ]
    body = {
        "schema_version": 1,
        "manifest_digest": digest(records),
        "inputs": {"train": {"original": "train-input"}, "test": {"original": "test-input"}},
        "config": {"model": {"id": "synthetic-fixture-model"}},
    }
    body["protocol_digest"] = digest(body)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    (tmp_path / "selected-model.json").write_text(json.dumps({
        "protocol_digest": body["protocol_digest"], "rule": "end_of_run", "model_digest": "fixture-adapter",
    }), encoding="utf-8")
    predictions = [{
        "id": "test", "condition": "original", "input_digest": "test-input",
        "protocol_digest": body["protocol_digest"],
        "model": "base", "model_digest": digest({"model": body["config"]["model"]}),
    }]
    return path, body, records, predictions


def test_report_gate_accepts_exact_frozen_artifacts_without_media(tmp_path):
    path, expected, records, predictions = frozen_report_fixture(tmp_path)
    assert verify_report_inputs(path, records, predictions) == expected


@pytest.mark.parametrize("field,value", [("answer", "B"), ("source_group", "other-group"), ("split", "val")])
def test_report_gate_rejects_changed_gold_group_or_split(tmp_path, field, value):
    path, _, records, predictions = frozen_report_fixture(tmp_path)
    records[1][field] = value
    with pytest.raises(ValueError, match="manifest differs"):
        verify_report_inputs(path, records, predictions)


def test_report_gate_rejects_changed_protocol_body(tmp_path):
    path, body, records, predictions = frozen_report_fixture(tmp_path)
    body["inputs"]["test"]["original"] = "modified-input"
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol integrity"):
        verify_report_inputs(path, records, predictions)


@pytest.mark.parametrize("field,value,match", [
    ("protocol_digest", "wrong-protocol", "different report protocol"),
    ("input_digest", "wrong-input", "input differs"),
    ("condition", "competing", "input differs"),
    ("id", "train", "outside the frozen test cohort"),
    ("id", "unknown", "outside the frozen test cohort"),
])
def test_report_gate_rejects_prediction_identity_changes(tmp_path, field, value, match):
    path, _, records, predictions = frozen_report_fixture(tmp_path)
    predictions[0][field] = value
    with pytest.raises(ValueError, match=match):
        verify_report_inputs(path, records, predictions)


def test_changed_answer_cannot_be_hidden_by_copying_old_protocol_digest(tmp_path):
    path, _, records, predictions = frozen_report_fixture(tmp_path)
    changed = deepcopy(records)
    changed[1]["answer"] = "B"
    with pytest.raises(ValueError, match="manifest differs"):
        verify_report_inputs(path, changed, predictions)


@pytest.mark.parametrize("model", ["base", "tuned"])
def test_report_gate_binds_both_model_states_to_frozen_selection(tmp_path, model):
    path, _, records, predictions = frozen_report_fixture(tmp_path)
    predictions[0].update(model=model, model_digest="wrong-model-state")
    with pytest.raises(ValueError, match="model differs"):
        verify_report_inputs(path, records, predictions)


@pytest.mark.parametrize("section,field,value", [
    ("model", "revision", "x" * 40),
    ("model", "dtype", "float32"),
    ("preprocessing", "frames", 7),
    ("training", "gradient_accumulation_steps", 0),
    ("training", "learning_rate", float("nan")),
    ("training", "checkpoint_selection", "best_test"),
    ("analysis", "bootstrap_samples", False),
])
def test_invalid_runtime_config_fails_before_gpu_work(tmp_path, section, field, value):
    import yaml
    path = Path(__file__).parents[1] / "configs" / "pilot.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config[section][field] = value
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(invalid)


def checkpoint_fixture(tmp_path):
    root = tmp_path / "checkpoint-000001"
    root.mkdir()
    for name in ("adapter_model.safetensors", "training.pt", "adapter_config.json"):
        (root / name).write_bytes(("fixture:" + name).encode())
    metadata = {
        "identity": "training-identity", "step": 1, "next_index": 4,
        "files": {name: file_digest(root / name) for name in (
            "adapter_model.safetensors", "training.pt", "adapter_config.json"
        )},
    }
    (root / "checkpoint.json").write_text(json.dumps(metadata), encoding="utf-8")
    return root, metadata


def test_checkpoint_file_integrity_requires_no_torch_or_real_weights(tmp_path):
    root, expected = checkpoint_fixture(tmp_path)
    assert verify_checkpoint(root) == expected


@pytest.mark.parametrize("name", ["adapter_model.safetensors", "training.pt", "adapter_config.json"])
def test_checkpoint_changed_payload_is_rejected(tmp_path, name):
    root, _ = checkpoint_fixture(tmp_path)
    (root / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity failure"):
        verify_checkpoint(root)


def test_checkpoint_missing_payload_is_rejected(tmp_path):
    root, _ = checkpoint_fixture(tmp_path)
    (root / "adapter_model.safetensors").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_checkpoint(root)


def test_checkpoint_missing_integrity_manifest_is_rejected(tmp_path):
    root, _ = checkpoint_fixture(tmp_path)
    (root / "checkpoint.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="no integrity manifest"):
        verify_checkpoint(root)


def test_checkpoint_integrity_paths_cannot_escape_checkpoint(tmp_path):
    root, metadata = checkpoint_fixture(tmp_path)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"unrelated")
    metadata["files"]["../outside.bin"] = file_digest(outside)
    (root / "checkpoint.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity failure"):
        verify_checkpoint(root)
