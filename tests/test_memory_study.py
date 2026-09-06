"""End-to-end CPU protocol checks; fixture pixels do not measure model quality."""
from copy import deepcopy
import json

from PIL import Image
import pytest

from ecqa.artifacts import digest
from ecqa.memory import MemoryStore
from ecqa.memory_study import (
    _write_store, answer_file, load_study, observation_projection, observe_file, prepare_study,
)


@pytest.fixture
def study(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    frames = []
    for i in range(8):
        name = f"frame{i}.png"
        Image.new("RGB", (128, 128), (i * 20, 10, 100)).save(source / name)
        frames.append(name)
    condition = {"frames": frames, "fps": 2.0, "clip_ranges": [[0, 4], [4, 8]]}
    record = {"id": "fixture", "split": "val", "source_group": "fixture-group",
              "question": "What is in the first clip?", "options": dict(zip("ABCD", ["red", "blue", "green", "gray"])),
              "answer": "A", "conditions": {"original": condition},
              "audit": {"target_clip": 1, "reviewer_type": "synthetic_fixture"}}
    settings = {"schema_version": 1, "seed": 42, "reader_pairs": 1, "capacity_pairs": 4,
                "conditions": ["full", "retrieval", "oracle", "empty"],
                "capacity_conditions": [{"name": "recent_capacity2", "capacity_pairs": 2, "policy": "recent"}],
                "interpretation": "diagnostic_existing_cohort_not_fresh_test"}
    return source, record, settings


def test_observation_projection_excludes_question_and_future_identity(study):
    root, record, _ = study
    first = observation_projection(record["conditions"]["original"], root, episode_id="external-episode")
    changed = deepcopy(record)
    changed.update(question="Different question about third clip", answer="D", options={})
    changed["audit"]["target_clip"] = 2
    second = observation_projection(changed["conditions"]["original"], root, episode_id="external-episode")
    assert first == second
    assert all(not ({"question", "answer", "options", "target_clip", "audit"} & set(row))
               for row in first["observations"])
    # An externally assigned identity is stable when later pixels change.
    Image.new("RGB", (128, 128), "orange").save(root / "frame7.png")
    third = observation_projection(record["conditions"]["original"], root, episode_id="external-episode")
    assert third["identity"] != first["identity"]
    assert third["episode_id"] == first["episode_id"]
    assert third["observations"][0] == first["observations"][0]


def test_prepare_seal_relocate_and_integrity(study, tmp_path):
    root, record, settings = study
    output = tmp_path / "run"
    result = prepare_study(settings, {}, [record], root, output)
    assert result["cases"] == 5 and result["stores"] == 2
    protocol, cases = load_study(output)
    assert protocol["cohort_counts"] == {"train": 0, "val": 1, "test": 0}
    selected = next(case for case in cases if case["condition"] == "retrieval")
    assert set(selected["record"]) == {"question", "options", "memory"}
    assert selected["retrieval"]["clip_indices"] == [1]
    assert next(case for case in cases if case["condition"] == "recent_capacity2")["record"]["memory"][
        "observation_ids"] == []
    # Reader evidence stays usable when the original input directory disappears.
    for path in root.iterdir():
        path.unlink()
    root.rmdir()
    with MemoryStore.open(selected["data_root"]) as store:
        assert store.verify()["store_digest"] == selected["store_digest"]
    contents = json.loads((output / "cases.json").read_text())
    contents[0]["gold"] = "D"
    (output / "cases.json").write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="Case manifest changed"):
        load_study(output)


def test_changed_questions_do_not_change_memory(study, tmp_path):
    root, record, settings = study
    prepare_study(settings, {}, [record], root, tmp_path / "one")
    changed = deepcopy(record)
    changed.update(question="What is in the second clip?", answer="B")
    changed["audit"]["target_clip"] = 2
    prepare_study(settings, {}, [changed], root, tmp_path / "two")
    commits = [json.loads((tmp_path / name / "observation-commit.json").read_text()) for name in ("one", "two")]
    assert commits[0]["stores_digest"] == commits[1]["stores_digest"]
    assert digest(json.loads((tmp_path / "one" / "cases.json").read_text())) != digest(
        json.loads((tmp_path / "two" / "cases.json").read_text()))


def test_observe_command_rejects_question_fields(study, tmp_path):
    root, record, _ = study
    projection = observation_projection(record["conditions"]["original"], root, episode_id="user-episode")
    path = root / "observations.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in projection["observations"]))
    result = observe_file(path, tmp_path / "store", capacity_pairs=1, policy="recent")
    assert result["retained_pairs"] == 1 and result["write_count"] == 4
    bad = {**projection["observations"][0], "question": "No labels in writer"}
    path.write_text(json.dumps(bad))
    with pytest.raises(TypeError):
        observe_file(path, tmp_path / "invalid")


def test_prepare_refuses_internally_valid_preexisting_store(study, tmp_path):
    root, record, _ = study
    projection = observation_projection(record["conditions"]["original"], root, episode_id="episode")
    path = tmp_path / "preexisting"
    _write_store(projection, path, 4, "recent", 42)
    with pytest.raises(FileExistsError, match="requires new stores"):
        _write_store(projection, path, 2, "episode", 41)


def test_public_answer_rejects_store_output_before_loading_model(tmp_path):
    with pytest.raises(ValueError, match="outside the sealed"):
        answer_file("missing-config", tmp_path / "store", "missing-questions", tmp_path / "store" / "answer.json")
    with pytest.raises(ValueError, match="one through four"):
        answer_file("missing-config", tmp_path / "store", "missing-questions", tmp_path / "answer.json", max_pairs=0)


def test_public_answer_refuses_unavailable_evidence_before_loading_model(study, tmp_path):
    root, record, _ = study
    projection = observation_projection(record["conditions"]["original"], root, episode_id="episode")
    _write_store(projection, tmp_path / "store", 1, "recent", 42)
    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps({"episode_id": "episode", "question": "What was in the first clip?",
                                     "options": record["options"]}))
    with pytest.raises(ValueError, match="No retained evidence"):
        answer_file("configs/memory.yaml", tmp_path / "store", questions, tmp_path / "answer.json")
    assert not (tmp_path / "answer.json").exists()
    with MemoryStore.open(tmp_path / "store") as store:
        store.verify()
