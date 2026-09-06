"""CPU evidence selection and exact retained-pixel materialization contracts."""

from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from ecqa.memory import MemoryStore
from ecqa.memory_model import MemoryRuntime, build_memory_prompt
from ecqa.model import ModelRuntime
from ecqa.retrieval import (
    OrdinalResolutionError, make_memory_record, question_clip, retrieve, retrieve_oracle,
)

OPTIONS = {"A": "Tower", "B": "Road", "C": "House", "D": "River"}
QUESTION = "In the second clip, what is left of the bridge?"


def _store(tmp_path, *, capacity=8, sizes=None, clips=(1, 1, 2, 2), policy="recent"):
    sources = tmp_path / "source"
    sources.mkdir()
    store = MemoryStore.create(tmp_path / "sealed", capacity_pairs=capacity, policy=policy)
    for index, clip in enumerate(clips):
        files = []
        for offset in range(2):
            path = sources / f"{index}_{offset}.png"
            Image.new("RGB", (sizes or {}).get(index, (128, 128)),
                      (index * 25, offset * 40, 80)).save(path)
            files.append(path)
        store.observe(episode_id="episode", scene_id=f"scene{clip}", clip_index=clip,
                      pair_index=index, timestamp=index + 0.25, frames=files, source_id=f"source{clip}")
    store.seal()
    return store


def _runtime(store):
    # Loading weights is unnecessary for the materialization contract.
    runtime = MemoryRuntime.__new__(MemoryRuntime)
    runtime.data_root = store.root
    runtime.config = {"preprocessing": {"pixel_budget_per_frame_min": 128 * 128,
                                       "pixel_budget_per_frame_max": 512 * 512}}
    runtime.processor = SimpleNamespace(video_processor=SimpleNamespace(patch_size=16, merge_size=2))
    return runtime


@pytest.mark.parametrize("question, expected", [
    ("In the first clip, which object?", 1), (QUESTION, 2),
    ("What is in the THIRD scene?", 3), ("In clip 4, which?", 4),
    ("What is in the 2nd video clip?", 2),
    ("In the second clip, was the second clip a city?", 2),
    ("In clip 5, which object?", 5), ("In clip 10, which object?", 10),
    ("In clip10, which object?", 10), ("In clip 100, which object?", 100),
])
def test_explicit_question_ordinal(question, expected):
    assert question_clip(question) == expected


@pytest.mark.parametrize("question", [
    "Which object is left of the bridge?", "Compare the first and second clips.",
    "In the fifth clip, which object?", "The first object in this video is what?",
    "In the second clip, was the first floor red?",
    "In clip 0, which object?", "Compare clip 5 and clip 10.",
    "Compare the first clip and clip 0.", "In clip 10a, which object?",
])
def test_missing_or_ambiguous_ordinal_refused(question):
    with pytest.raises(OrdinalResolutionError):
        question_clip(question)


def test_episode_selection_preserves_identity_and_has_no_target_argument(tmp_path):
    with _store(tmp_path) as store:
        result = retrieve(store, episode_id="episode", question=QUESTION, max_pairs=1)
        assert [item["clip_index"] for item in result.observations] == [2]
        assert result.provenance["observation_ids"] == [3]
        assert not result.provenance["evidence_usage_proven"]
        with pytest.raises(TypeError):
            retrieve(store, episode_id="episode", question=QUESTION, target_clip=1)
        with pytest.raises(TypeError):
            retrieve(store, episode_id="episode", question=QUESTION, audit={"target_clip": 1})
        row = make_memory_record(QUESTION, OPTIONS, result)
        assert set(row) == {"question", "options", "memory"}
        changed = deepcopy(row)
        changed.update(answer="D", audit={"target_clip": 1}, id="gold-secret")
        runtime = _runtime(store)
        actual, video, _ = runtime._materialize(row)
        altered, altered_video, _ = runtime._materialize(changed)
        assert actual == altered
        np.testing.assert_array_equal(video, altered_video)
        text = actual[0]["content"][-1]["text"]
        assert "original Clip 2" in text and "original Clip 1" not in text
        assert "local reference frames 1-2" in text and "<0.2 seconds>" in text
        assert "observation timestamp 2.250000 seconds" in text
        assert "synthetic prepared-frame timeline, not real elapsed flight time" in text
        assert "gold-secret" not in text
        assert video.shape == (2, 128, 128, 3)


def test_even_recent_empty_wrong_and_oracle_controls(tmp_path):
    with _store(tmp_path) as store:
        uniform = retrieve(store, episode_id="episode", question=QUESTION, policy="uniform", max_pairs=2)
        recent = retrieve(store, episode_id="episode", question=QUESTION, policy="recent", max_pairs=2)
        wrong = retrieve(store, episode_id="episode", question=QUESTION, policy="wrong_episode")
        assert uniform.provenance["observation_ids"] == [1, 4]
        assert recent.provenance["observation_ids"] == [3, 4]
        assert wrong.provenance["clip_indices"] == [1, 1]
        assert retrieve(store, episode_id="episode", question=QUESTION, policy="uniform", max_pairs=2) == uniform
        oracle = retrieve_oracle(store, episode_id="episode", target_clip=1)
        assert oracle.provenance["diagnostic"] and oracle.provenance["policy"] == "oracle"
        assert oracle.provenance["clip_indices"] == [1, 1]
        empty = retrieve(store, episode_id="episode", question=QUESTION, policy="empty")
        messages, video, _ = _runtime(store)._materialize(make_memory_record(QUESTION, OPTIONS, empty))
        assert video is None
        assert [item["type"] for item in messages[0]["content"]] == ["text"]
        assert "No visual observations" in messages[0]["content"][0]["text"]


def test_ordinal_fallback_is_explicit_and_recorded(tmp_path):
    with _store(tmp_path) as store:
        ambiguous = "Compare the first and second clips."
        with pytest.raises(OrdinalResolutionError):
            retrieve(store, episode_id="episode", question=ambiguous)
        result = retrieve(store, episode_id="episode", question=ambiguous,
                          ordinal_fallback="recent", max_pairs=1)
        assert result.provenance["observation_ids"] == [4]
        assert result.provenance["fallback"] == "recent"
        assert result.provenance["requested_clip"] is None
        assert "multiple" in result.provenance["reason"]


def test_evicted_pixels_are_not_reopened_and_missing_target_returns_empty(tmp_path):
    with _store(tmp_path, capacity=1) as store:
        result = retrieve(store, episode_id="episode", question="In the first clip, which object?")
        assert not result.observations
        assert "No retained observations" in result.provenance["reason"]
        row = make_memory_record(QUESTION, OPTIONS, retrieve(store, episode_id="episode", question=QUESTION))
        row["memory"]["observation_ids"] = [1]
        with pytest.raises(ValueError, match="unavailable"):
            _runtime(store)._materialize(row)


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_variable_even_frame_counts_and_original_pixels(tmp_path, count):
    with _store(tmp_path) as store:
        result = retrieve(store, episode_id="episode", question=QUESTION, policy="uniform", max_pairs=count)
        _, video, fps = _runtime(store)._materialize(make_memory_record(QUESTION, OPTIONS, result))
        assert video.shape == (count * 2, 128, 128, 3) and fps == 2.0
        for index, observation in enumerate(result.observations):
            with Image.open(store.root / observation["frames"][0]) as frame:
                np.testing.assert_array_equal(video[2 * index], np.array(frame))


def test_runtime_refuses_stale_digest_cross_episode_duplicate_and_reversed_ids(tmp_path):
    with _store(tmp_path) as store:
        result = retrieve(store, episode_id="episode", question=QUESTION, policy="uniform")
        row = make_memory_record(QUESTION, OPTIONS, result)
        for field, value in [("store_digest", "0" * 64), ("episode_id", "another-episode"),
                             ("observation_ids", [1, 1]), ("observation_ids", [4, 1])]:
            changed = deepcopy(row)
            changed["memory"][field] = value
            with pytest.raises(ValueError):
                _runtime(store)._materialize(changed)


def test_runtime_refuses_image_tampering_and_added_raw_path(tmp_path):
    with _store(tmp_path) as store:
        result = retrieve(store, episode_id="episode", question=QUESTION)
        row = make_memory_record(QUESTION, OPTIONS, result)
        bad = deepcopy(row)
        bad["memory"]["frames"] = ["../source/0_0.png"]
        with pytest.raises(ValueError, match="allowlist"):
            _runtime(store)._materialize(bad)
        path = store.root / result.observations[0]["frames"][0]
        Image.new("RGB", (128, 128), "red").save(path)
        with pytest.raises(ValueError):
            _runtime(store)._materialize(row)


@pytest.mark.parametrize("size", [(130, 128), (64, 64)])
def test_runtime_geometry_is_checked_without_resize(tmp_path, size):
    with _store(tmp_path, sizes={2: size, 3: size}) as store:
        result = retrieve(store, episode_id="episode", question=QUESTION)
        with pytest.raises(ValueError):
            _runtime(store)._materialize(make_memory_record(QUESTION, OPTIONS, result))


def test_baseline_delegation_and_per_case_store_root_are_uncached(tmp_path, monkeypatch):
    calls = []
    sentinel = object()
    monkeypatch.setattr(ModelRuntime, "_materialize", lambda self, row, condition:
                        (calls.append((self.data_root, row, condition)) or sentinel))
    runtime = MemoryRuntime.__new__(MemoryRuntime)
    runtime.data_root = Path("baseline-data")
    baseline = {"question": QUESTION, "options": OPTIONS, "conditions": {}}
    assert runtime._materialize(baseline) is sentinel
    assert calls == [(Path("baseline-data"), baseline, "original")]
    with _store(tmp_path) as store:
        runtime = _runtime(store)
        result = retrieve(store, episode_id="episode", question=QUESTION)
        row = make_memory_record(QUESTION, OPTIONS, result)
        runtime._materialize(row)
        runtime.data_root = tmp_path / "source"
        with pytest.raises(ValueError, match="sealed"):
            runtime._materialize(row)


@pytest.mark.parametrize("budget", [-1, 5, True, 1.5])
def test_frame_budget_cannot_exceed_eight_or_be_noninteger(tmp_path, budget):
    with _store(tmp_path) as store, pytest.raises(ValueError, match="budget"):
        retrieve(store, episode_id="episode", question=QUESTION, max_pairs=budget)


def test_empty_prompt_does_not_invent_clip_map():
    prompt = build_memory_prompt(QUESTION, OPTIONS, [])
    assert "No visual observations" in prompt and "Visual group" not in prompt


def test_numeric_clip_retrieval_does_not_match_a_digit_prefix(tmp_path):
    with _store(tmp_path, clips=(5, 5, 10, 10)) as store:
        result = retrieve(store, episode_id="episode", question="In clip10, which object?")
        assert result.provenance["requested_clip"] == 10
        assert result.provenance["clip_indices"] == [10, 10]


def test_default_timeline_prompt_is_byte_identical_and_provided_is_explicit():
    observations = [{"timestamp": 2.25, "clip_index": 2, "pair_index": 2}]
    prompt = build_memory_prompt(QUESTION, OPTIONS, observations)
    # Recorded from the unchanged synthetic prompt before adding the public mode.
    assert hashlib.sha256(prompt.encode()).hexdigest() == (
        "f4cf30c21739e0edd86dbc19924935c424eac0add642c6ecb1d8f3c1410b0517")
    assert prompt == build_memory_prompt(QUESTION, OPTIONS, observations, observation_timeline="synthetic")
    provided = build_memory_prompt(QUESTION, OPTIONS, observations, observation_timeline="provided")
    assert "Prepared timestamps are synthetic identifiers" in provided
    assert "Observation timestamps are observer-supplied values" in provided
    assert "time origin, units, and semantics are unknown" in provided
    assert "observation timestamp 2.250000 (observer-supplied)" in provided
    assert "synthetic prepared-frame timeline, not real elapsed flight time" not in provided
    with pytest.raises(ValueError, match="observation_timeline"):
        build_memory_prompt(QUESTION, OPTIONS, observations, observation_timeline="real")


def test_runtime_timeline_configuration_changes_labels_only(tmp_path):
    with _store(tmp_path) as store:
        result = retrieve(store, episode_id="episode", question=QUESTION)
        row = make_memory_record(QUESTION, OPTIONS, result)
        runtime = _runtime(store)
        original_messages, original_video, original_fps = runtime._materialize(row)
        runtime.config["memory"] = {"observation_timeline": "provided"}
        provided_messages, provided_video, provided_fps = runtime._materialize(row)
        assert "observer-supplied values" in provided_messages[0]["content"][-1]["text"]
        assert "observer-supplied values" not in original_messages[0]["content"][-1]["text"]
        np.testing.assert_array_equal(original_video, provided_video)
        assert original_fps == provided_fps
