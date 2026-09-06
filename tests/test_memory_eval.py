"""Real sealed stores, fake likelihood runtimes: no model downloads or GPU."""
from copy import deepcopy
import json
from pathlib import Path

from PIL import Image
import pytest

from ecqa.artifacts import digest
from ecqa.memory import MemoryStore
from ecqa.memory_eval import evaluate_cases, summarize_cases, write_memory_report


class Runtime:
    def __init__(self, root, answers=None, failure=False):
        self.data_root = Path(root)
        self.config = {"model": {"id": "cpu-test", "revision": "pinned"}}
        self.answers = answers or {}
        self.failure = failure
        self.calls = []

    def scores(self, record, condition):
        self.calls.append((deepcopy(record), self.data_root))
        assert set(record).issubset({"question", "options", "memory", "conditions"})
        assert condition == "original"
        if self.failure:
            raise RuntimeError("Intentional interrupted evaluation")
        answer = self.answers.get(record["question"], "A")
        return {label: -0.1 if label == answer else -2.0 for label in "ABCD"}


@pytest.fixture
def store(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    paths = []
    for i in range(8):
        path = source / f"{i}.png"
        Image.new("RGB", (64, 64), (i * 20, 30, 40)).save(path)
        paths.append(path)
    with MemoryStore.create(tmp_path / "store", 4) as memory:
        for i in range(4):
            memory.observe(episode_id="episode", scene_id=f"scene-{i // 2}", clip_index=i // 2 + 1,
                           pair_index=i, timestamp=i + 0.25, frames=paths[2 * i:2 * i + 2],
                           source_id=f"source-{i // 2}")
        manifest = memory.seal()
    return tmp_path / "store", manifest


def case(store, identifier="q1", condition="full", gold="A", group="g1", split="val"):
    root, manifest = store
    record = {"question": identifier, "options": {label: f"Option {label}" for label in "ABCD"}}
    if condition == "full":
        record["conditions"] = {"original": {"frames": [frame for obs in manifest["observations"]
                                                          for frame in obs["frames"]],
                                               "clip_ranges": [[0, 4], [4, 8]], "fps": 2.0}}
    else:
        selections = {"memory_full": [1, 2, 3, 4], "retrieval": [1], "oracle": [1],
                      "oracle_all": [1, 2], "recent": [4], "uniform": [2], "wrong": [3], "empty": []}
        record["memory"] = {"episode_id": "episode", "store_digest": manifest["store_digest"],
                            "observation_ids": selections[condition]}
    return {"id": identifier, "split": split, "condition": condition, "record": record,
            "gold": gold, "data_root": str(root), "store_digest": manifest["store_digest"],
            "source_group": group, "question_class": "image_plane",
            "evidence": {"target_episode_id": "episode", "target_clip_index": 1,
                         "target_pair_count": 2, "delay_clips": 1, "retrieval_seconds": 0.05}}


def evaluate(runtime, cases, output, model="base"):
    return evaluate_cases(runtime, cases, output, model_name=model,
                          model_digest="weights-" + model, protocol_digest="frozen-protocol")


def prediction_path(output, model="base"):
    return next(path for path in (output / model).glob("*.json") if path.name != "manifest.json")


def rewrite_row(path, **changes):
    row = json.loads(path.read_text())
    row.update(changes)
    row["row_digest"] = digest({key: value for key, value in row.items() if key != "row_digest"})
    path.write_text(json.dumps(row))
    return row


def test_real_store_evidence_coverage_is_post_selection_and_partial(store, tmp_path):
    cases = [case(store, condition=name) for name in ("full", "retrieval", "wrong", "empty")]
    runtime = Runtime(tmp_path)
    rows = evaluate(runtime, cases, tmp_path / "scores")
    assert runtime.data_root == tmp_path
    assert all(root == store[0] for _, root in runtime.calls)
    metrics = {row["condition"]: row["evidence_metrics"] for row in rows}
    assert metrics["full"]["target_pair_recall"] == 1
    assert metrics["retrieval"]["target_pair_recall"] == 0.5
    assert metrics["wrong"]["target_clip_hit"] is False
    assert metrics["empty"]["selected_pairs"] == 0
    assert metrics["empty"]["target_pair_recall"] == 0
    assert metrics["retrieval"]["store_bytes"] > 0
    assert runtime.calls[1][0]["memory"]["observation_ids"] == [1]
    assert all("evidence" not in record and "gold" not in record for record, _ in runtime.calls)


def test_metadata_cannot_enter_reader_and_cached_success_skips_runtime(store, tmp_path):
    item = case(store)
    item["record"].update(answer="D", audit={"target_clip": 999}, provenance={"secret": "audit"})
    runtime = Runtime(tmp_path)
    first = evaluate(runtime, [item], tmp_path / "scores")
    second = evaluate(runtime, [item], tmp_path / "scores")
    assert len(runtime.calls) == 1
    assert first == second
    assert set(runtime.calls[0][0]) == {"question", "options", "conditions"}


@pytest.mark.parametrize("field", ["question", "gold", "source_group", "store_digest", "audit"])
def test_resume_refuses_changed_frozen_case(store, tmp_path, field):
    item = case(store)
    runtime = Runtime(tmp_path)
    evaluate(runtime, [item], tmp_path / "scores")
    changed = deepcopy(item)
    if field == "question":
        changed["record"]["question"] = "Different question"
    elif field == "audit":
        changed["evidence"]["target_clip_index"] = 2
    else:
        changed[field] = {"gold": "B", "source_group": "different", "store_digest": "stale"}[field]
    with pytest.raises(ValueError, match="mismatch|identity differs"):
        evaluate(runtime, [changed], tmp_path / "scores")
    assert len(runtime.calls) == 1


def test_model_config_and_code_mismatches_refuse_resume(store, tmp_path, monkeypatch):
    runtime, cases = Runtime(tmp_path), [case(store)]
    output = tmp_path / "scores"
    evaluate(runtime, cases, output)
    runtime.config["changed"] = True
    with pytest.raises(ValueError, match="manifest mismatch"):
        evaluate(runtime, cases, output)
    runtime.config.pop("changed")
    monkeypatch.setattr("ecqa.memory_eval.code_identity", lambda: "different-code")
    with pytest.raises(ValueError, match="manifest mismatch"):
        evaluate(runtime, cases, output)


@pytest.mark.parametrize("changes", [
    {"seconds": -0.1}, {"seconds": True}, {"prediction": "D"}, {"margin": 100},
    {"scores": {"A": 1, "B": 2}}, {"status": "skipped"}, {"input_digest": "changed"},
])
def test_even_rechecksummed_semantically_invalid_cache_is_rejected(store, tmp_path, changes):
    runtime, cases, output = Runtime(tmp_path), [case(store)], tmp_path / "scores"
    evaluate(runtime, cases, output)
    rewrite_row(prediction_path(output), **changes)
    with pytest.raises(ValueError):
        evaluate(runtime, cases, output)
    assert len(runtime.calls) == 1


def test_corrupt_json_and_row_digest_fail_closed(store, tmp_path):
    runtime, cases, output = Runtime(tmp_path), [case(store)], tmp_path / "scores"
    evaluate(runtime, cases, output)
    path = prediction_path(output)
    original = path.read_text()
    path.write_text("{")
    with pytest.raises(ValueError):
        evaluate(runtime, cases, output)
    row = json.loads(original)
    row["scores"]["A"] = -9
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="integrity"):
        evaluate(runtime, cases, output)


def test_failure_is_atomic_restores_root_and_can_retry_same_identity(store, tmp_path):
    runtime, cases, output = Runtime(tmp_path, failure=True), [case(store)], tmp_path / "scores"
    with pytest.raises(RuntimeError, match="Intentional"):
        evaluate(runtime, cases, output)
    assert runtime.data_root == tmp_path
    failed = json.loads(prediction_path(output).read_text())
    assert failed["status"] == "error"
    with pytest.raises(ValueError, match="Failed"):
        summarize_cases(cases, [failed], expected_models=("base",), expected_conditions={"val": ["full"]})
    runtime.failure = False
    assert evaluate(runtime, cases, output)[0]["status"] == "ok"
    assert len(runtime.calls) == 2


def test_store_tampering_fails_before_runtime(store, tmp_path):
    root, manifest = store
    image_path = root / manifest["observations"][0]["frames"][0]
    image_path.write_bytes(b"corrupt pixels")
    runtime = Runtime(tmp_path)
    with pytest.raises(ValueError, match="integrity|differs"):
        evaluate(runtime, [case(store, condition="retrieval")], tmp_path / "scores")
    assert not runtime.calls


def test_two_model_paired_report_hand_computed_and_diagnostic_labels(store, tmp_path):
    cases = [case(store, "q1", gold="A", group="g1"), case(store, "q2", gold="B", group="g2")]
    base = evaluate(Runtime(tmp_path, {"q1": "B", "q2": "B"}), cases, tmp_path / "scores")
    tuned = evaluate(Runtime(tmp_path, {"q1": "A", "q2": "A"}), cases, tmp_path / "scores", "tuned")
    kwargs = {"expected_conditions": {"val": ["full"]}, "bootstrap_samples": 1000, "seed": 42}
    summary = write_memory_report(cases, base + tuned, tmp_path / "report", **kwargs)
    detail = summary["splits"]["val"]["conditions"]["full"]
    assert detail["models"]["base"]["accuracy"] == 0.5
    assert detail["models"]["base"]["mean_gold_negative_logscore"] == pytest.approx(1.05)
    assert detail["models"]["tuned"]["by_gold_class"]["A"]["accuracy"] == 1
    assert detail["models"]["tuned"]["by_gold_class"]["B"]["accuracy"] == 0
    assert detail["tuned_minus_base"]["accuracy_change"] == 0
    assert detail["tuned_minus_base"]["corrected"] == 1
    assert detail["tuned_minus_base"]["regressed"] == 1
    assert detail["tuned_minus_base"]["ci95"] == [-1, 1]
    assert detail["evidence"]["target_clip_hit_rate"] == 1
    assert detail["delay_clips_counts"] == {"1": 2}
    assert (tmp_path / "report" / "memory-report.md").exists()
    assert summary == summarize_cases(cases, base + tuned, **kwargs)


@pytest.mark.parametrize("problem", ["missing", "duplicate", "unexpected", "case_changed", "condition_missing"])
def test_report_requires_exact_complete_pairing(store, tmp_path, problem):
    cases = [case(store)]
    rows = evaluate(Runtime(tmp_path), cases, tmp_path / "scores")
    rows += evaluate(Runtime(tmp_path), cases, tmp_path / "scores", "tuned")
    expected = {"val": ["full"]}
    if problem == "missing":
        rows.pop()
    elif problem == "duplicate":
        rows.append(rows[0])
    elif problem == "unexpected":
        rows[0]["model"] = "unexpected"
        rows[0]["row_digest"] = digest({k: v for k, v in rows[0].items() if k != "row_digest"})
    elif problem == "case_changed":
        cases[0]["record"]["question"] = "Changed after evaluation"
    else:
        expected["val"].append("retrieval")
    with pytest.raises(ValueError):
        summarize_cases(cases, rows, expected_conditions=expected)


def test_expanded_conditions_within_model_contrasts_and_single_group_ci(store, tmp_path):
    conditions = ["full", "memory_full", "retrieval", "uniform", "recent", "oracle", "oracle_all"]
    cases = [case(store, condition=name) for name in conditions]
    rows = evaluate(Runtime(tmp_path), cases, tmp_path / "scores")
    rows += evaluate(Runtime(tmp_path), cases, tmp_path / "scores", "tuned")
    result = summarize_cases(cases, rows, expected_conditions={"val": conditions})
    contrasts = result["splits"]["val"]["within_model_contrasts"]
    assert contrasts["memory_full_minus_full"]["base"]["accuracy_change"] == 0
    assert contrasts["retrieval_minus_uniform"]["tuned"]["ci95"] is None
    assert contrasts["oracle_all_minus_oracle"]["base"]["source_groups"] == 1


def test_empty_cohort_and_duplicate_cases_fail(store, tmp_path):
    with pytest.raises(ValueError, match="Empty"):
        evaluate(Runtime(tmp_path), [], tmp_path / "scores")
    with pytest.raises(ValueError, match="Empty"):
        summarize_cases([], [])
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate(Runtime(tmp_path), [case(store), case(store)], tmp_path / "scores")


def test_target_audit_denominator_and_selected_id_validations(store, tmp_path):
    item = case(store, condition="memory_full")
    item["evidence"]["target_pair_count"] = 1
    with pytest.raises(ValueError, match="denominator"):
        evaluate(Runtime(tmp_path), [item], tmp_path / "scores")
    item = case(store, condition="retrieval")
    item["record"]["memory"]["observation_ids"] = [99]
    with pytest.raises(ValueError, match="absent"):
        evaluate(Runtime(tmp_path), [item], tmp_path / "scores")
