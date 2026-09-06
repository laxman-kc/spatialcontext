"""CPU protocol, evidence, and unchanged-loop routing checks; no model simulation run."""
from copy import deepcopy
import json
from pathlib import Path

from PIL import Image
import pytest

from ecqa.artifacts import atomic_json, digest
from ecqa.memory import MemoryStore
from ecqa.memory_model import MemoryRuntime
from ecqa import memory_train as mt


def config():
    return {"schema_version": 1, "model": {"id": "Qwen/Qwen3-VL-4B-Instruct", "revision": "a" * 40,
            "dtype": "bfloat16", "attention_backend": "sdpa_math", "deterministic": True},
            "preprocessing": {"frames": 8, "synthetic_fps": 2.0, "max_sequence_tokens": 4096},
            "training": {"epochs": 1, "seed": 42, "learning_rate": 1e-5, "warmup_ratio": 0.03,
            "weight_decay": 0.01, "max_grad_norm": 1.0, "gradient_accumulation_steps": 4,
            "max_cumulative_seconds": 7200, "checkpoint_every_optimizer_steps": 25,
            "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.05, "checkpoint_selection": "end_of_run"},
            "analysis": {"seed": 42, "bootstrap_samples": 10}}


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    # Small CPU fixture exercises the same exact-count gate without a GPU or benchmark scores.
    monkeypatch.setattr(mt, "EXPECTED_COUNTS", {"train": 1, "val": 1})
    study = tmp_path / "parent"
    rows, cases = [], []
    sources = []
    for index in range(8):
        path = tmp_path / f"image{index}.png"
        Image.new("RGB", (128, 128), (index * 20, 10, 20)).save(path)
        sources.append(path)
    for split in ("train", "val"):
        condition = {"frames": [str(p) for p in sources], "clip_ranges": [[0, 2], [2, 4], [4, 8]], "fps": 2.0}
        row = {"id": split, "split": split, "question": "In the second clip, which object is left?",
               "options": {label: label for label in "ABCD"}, "answer": "B", "source_group": split,
               "conditions": {"original": condition}, "audit": {"status": "approved", "reviewer": "AI fixture",
               "reviewer_type": "ai", "answer_verified": True, "evidence_verified": True,
               "boundaries_verified": True, "viewpoint_verified": True, "target_clip": 2}}
        root = study / "memory" / split
        with MemoryStore.create(root, capacity_pairs=4) as store:
            for index, clip in enumerate([1, 2, 3, 3]):
                store.observe(episode_id=split, scene_id=f"scene{clip}", source_id=f"source{clip}",
                              clip_index=clip, pair_index=index, timestamp=index + 0.25,
                              frames=sources[index * 2:index * 2 + 2])
            receipt = store.seal()
            retained = store.observations(split)
        reader = {"question": row["question"], "options": row["options"], "conditions": {
            "original": {**condition, "frames": [p for obs in retained for p in obs["frames"]]}}}
        cases.append({"id": split, "split": split, "condition": "full", "record": reader,
                      "gold": "B", "store_digest": receipt["store_digest"], "source_group": split,
                      "data_root": str(root), "evidence": {"target_episode_id": split,
                      "target_clip_index": 2, "target_pair_count": 1}})
        rows.append(row)
    rows.append({"id": "old-test", "split": "test", "never_read_pixels": True})
    cases.append({"id": "old-test", "split": "test", "condition": "full", "data_root": "/NEVER_OPEN_TEST"})
    parent = {"protocol_digest": "parent-digest", "cohort_digest": digest(rows), "config": config()}
    monkeypatch.setattr(mt, "_parent", lambda _: (deepcopy(parent), deepcopy(cases)))
    manifest = tmp_path / "approved.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return {"study": study, "manifest": manifest, "output": tmp_path / "candidate",
            "parent": parent, "cases": cases, "rows": rows}


def test_prepare_binds_train_validation_only_and_same_question_selected_evidence(prepared):
    result = mt.prepare_candidate(prepared["study"], prepared["manifest"], prepared["output"])
    protocol, cases = mt.load_candidate(prepared["study"], prepared["output"])
    assert result["counts"] == {"train": 1, "val": 1}
    assert {case["split"] for case in cases} == {"train", "val"}
    assert {case["condition"] for case in cases} == {"retrieval_all"}
    assert all(case["retrieval"]["requested_clip"] == 2 and not case["retrieval"]["diagnostic"] for case in cases)
    assert len(protocol["training_records"]) == 1 and protocol["candidate_count"] == 1
    assert all("answer" not in case["record"] and "audit" not in case["record"] for case in cases)
    assert protocol["config"]["training"] == prepared["parent"]["config"]["training"]
    assert protocol["historical_parent_code_check"] is False


def test_ordinal_selection_happens_before_audit_and_mismatch_fails(prepared, monkeypatch):
    row = prepared["rows"][0]
    row["audit"]["target_clip"] = 1
    prepared["parent"]["cohort_digest"] = digest(prepared["rows"])
    prepared["manifest"].write_text("".join(json.dumps(r) + "\n" for r in prepared["rows"]))
    original = mt.retrieve
    seen = []
    def checked(*args, **kwargs):
        seen.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(mt, "retrieve", checked)
    with pytest.raises(ValueError, match="audited target"):
        mt.prepare_candidate(prepared["study"], prepared["manifest"], prepared["output"])
    assert seen[0]["question"] == row["question"] and set(seen[0]) == {"episode_id", "question", "max_pairs"}


@pytest.mark.parametrize("failure", ["manifest", "approval", "count", "source_group", "learning_rate"])
def test_source_and_fixed_design_gates(prepared, failure):
    if failure == "manifest":
        prepared["parent"]["cohort_digest"] = "bad"
    else:
        if failure == "approval":
            prepared["rows"][0]["audit"]["evidence_verified"] = False
        elif failure == "count":
            prepared["rows"].pop(0)
        elif failure == "source_group":
            prepared["rows"][1]["source_group"] = "train"
        else:
            prepared["parent"]["config"]["training"]["learning_rate"] = 2e-5
        prepared["parent"]["cohort_digest"] = digest(prepared["rows"])
        prepared["manifest"].write_text("".join(json.dumps(r) + "\n" for r in prepared["rows"]))
    with pytest.raises(ValueError):
        mt.prepare_candidate(prepared["study"], prepared["manifest"], prepared["output"])


@pytest.mark.parametrize("failure", ["protocol", "code", "lineage", "pixels"])
def test_resume_load_reverifies_hashes_protocol_and_selected_store(prepared, monkeypatch, failure):
    mt.prepare_candidate(prepared["study"], prepared["manifest"], prepared["output"])
    if failure == "protocol":
        path = prepared["output"] / "protocol.json"
        body = json.loads(path.read_text())
        body["training_records"][0]["answer"] = "C"
        atomic_json(path, body)
    elif failure == "code":
        monkeypatch.setattr(mt, "code_identity", lambda: "changed")
    elif failure == "lineage":
        prepared["parent"]["protocol_digest"] = "changed"
    else:
        frame = next((prepared["study"] / "memory/train/frames").glob("*.png"))
        frame.write_bytes(b"altered")
    with pytest.raises(ValueError):
        mt.load_candidate(prepared["study"], prepared["output"])


def test_runtime_routes_stripped_reader_record_and_gold_only_to_loss_api(prepared, monkeypatch):
    mt.prepare_candidate(prepared["study"], prepared["manifest"], prepared["output"])
    protocol, cases = mt.load_candidate(prepared["study"], prepared["output"])
    records = protocol["training_records"]
    roots = {case["id"]: case["data_root"] for case in cases if case["split"] == "train"}
    def init(self, config, data_root, adapter=None):
        assert adapter is None
        self.data_root = Path(data_root)
    monkeypatch.setattr(MemoryRuntime, "__init__", init)
    calls = []
    def encode(self, record, condition, answer):
        calls.append((self.data_root, record, condition, answer))
        return "encoded-sentinel"
    monkeypatch.setattr(MemoryRuntime, "encode", encode)
    runtime = mt.TrainingRuntime(protocol["config"], prepared["study"], records, roots)
    assert runtime.encode(records[0], answer="B") == "encoded-sentinel"
    assert calls[0][0] == Path(roots["train"])
    assert set(calls[0][1]) == {"question", "options", "memory"}
    assert calls[0][2:] == ("original", "B")
    assert runtime.data_root == prepared["study"]
    for altered, answer in [({**records[0], "answer": "C"}, "C"), (records[0], "D")]:
        with pytest.raises(ValueError, match="frozen"):
            runtime.encode(altered, answer=answer)
    def fail(*args, **kwargs):
        raise RuntimeError("encoding failed")
    monkeypatch.setattr(MemoryRuntime, "encode", fail)
    with pytest.raises(RuntimeError):
        runtime.encode(records[0], answer="B")
    assert runtime.data_root == prepared["study"]


def test_training_delegates_once_to_existing_loop_with_new_protocol(prepared, monkeypatch):
    mt.prepare_candidate(prepared["study"], prepared["manifest"], prepared["output"])
    sentinel = object()
    monkeypatch.setattr(mt, "TrainingRuntime", lambda *args: sentinel)
    calls = []
    def loop(runtime, records, cfg, output, **kwargs):
        calls.append((runtime, records, cfg, output, kwargs))
        return {"delegation_checked": True}
    monkeypatch.setattr(mt, "train", loop)
    assert mt.run_training(prepared["study"], prepared["output"]) == {"delegation_checked": True}
    assert len(calls) == 1 and calls[0][0] is sentinel
    assert [record["split"] for record in calls[0][1]] == ["train"]
    assert calls[0][4]["protocol_digest"] != prepared["parent"]["protocol_digest"]
    assert calls[0][4]["resume"] is None


def test_new_candidate_cannot_overwrite_parent_or_existing_output(prepared):
    with pytest.raises(ValueError, match="new directory"):
        mt.prepare_candidate(prepared["study"], prepared["manifest"], prepared["study"] / "new")
    mt.prepare_candidate(prepared["study"], prepared["manifest"], prepared["output"])
    with pytest.raises(ValueError, match="new directory"):
        mt.prepare_candidate(prepared["study"], prepared["manifest"], prepared["output"])


def test_incomplete_final_checkpoint_is_not_evaluable(prepared, monkeypatch):
    mt.prepare_candidate(prepared["study"], prepared["manifest"], prepared["output"])
    protocol, _ = mt.load_candidate(prepared["study"], prepared["output"])
    records = protocol["training_records"]
    identity = digest({"protocol": protocol["protocol_digest"], "records": records,
                       "settings": protocol["config"]["training"], "diagnostic": False})
    metadata = {"identity": identity, "diagnostic": False, "config": protocol["config"],
                "protocol_digest": protocol["protocol_digest"], "next_index": len(records)}
    monkeypatch.setattr(mt, "verify_checkpoint", lambda path: metadata)
    summary = {"checkpoint": "checkpoint-000001", "identity": identity, "stop_reason": "epoch_complete"}
    path = prepared["output"] / "train/summary.json"
    atomic_json(path, summary)
    assert mt._candidate_adapter(prepared["output"], protocol).name == "checkpoint-000001"
    summary["stop_reason"] = "time_cap"
    atomic_json(path, summary)
    with pytest.raises(ValueError, match="complete"):
        mt._candidate_adapter(prepared["output"], protocol)


@pytest.mark.parametrize("base_correct,tuned_correct,winner", [(1, 1, "base"), (1, 0, "base"), (0, 1, "tuned")])
def test_report_rule_uses_validation_only_and_keeps_base_on_tie(prepared, monkeypatch,
                                                              base_correct, tuned_correct, winner):
    """Pure report-decision fixture, not measured GPU accuracy."""
    mt.prepare_candidate(prepared["study"], prepared["manifest"], prepared["output"])
    protocol, _ = mt.load_candidate(prepared["study"], prepared["output"])
    adapter = prepared["output"] / "train/checkpoint-000001"
    monkeypatch.setattr(mt, "_candidate_adapter", lambda *args: adapter)
    monkeypatch.setattr(mt, "model_identity", lambda cfg, path=None: "tuned-id" if path else "base-id")
    for name in ("base", "tuned"):
        atomic_json(prepared["output"] / f"{name}-predictions.json", [])
    seen = []
    def summarize(cases, rows, **kwargs):
        seen.append(kwargs)
        return {"identities": {"protocol_digest": protocol["protocol_digest"],
                "code_digest": protocol["code_digest"], "config_digest": digest(protocol["config"]),
                "model:base": "base-id", "model:tuned": "tuned-id"}, "limitations": [],
                "splits": {"train": {"irrelevant_to_selection": "any training result"},
                           "val": {"conditions": {"retrieval_all": {"models": {
                               "base": {"correct": base_correct}, "tuned": {"correct": tuned_correct}}}}}}}
    monkeypatch.setattr(mt, "summarize_cases", summarize)
    selected = mt.report_candidate(prepared["study"], prepared["output"])
    assert selected["selected"] == winner
    assert seen[0]["expected_conditions"] == {"train": ["retrieval_all"], "val": ["retrieval_all"]}
