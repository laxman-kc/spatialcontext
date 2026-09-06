"""Lightweight model-contract tests; real processor/backward checks are gpu_smoke."""

from copy import deepcopy

import pytest

from ecqa.model import build_prompt, clip_map, continuation_start, last_valid_position


def record():
    condition = {"frames": [f"{i}.png" for i in range(8)],
                 "clip_ranges": [[0, 4], [4, 8]], "fps": 2.0}
    return {"id": "secret-id", "question": "What is left of the bridge?",
            "options": {"A": "Tower", "B": "Road", "C": "House", "D": "River"},
            "answer": "A", "audit": {"notes": "secret object evidence"},
            "conditions": {"original": condition,
                           "text_only": {**condition, "frames": []}}}


def test_prompt_allowlist_and_text_only_map():
    first = record()
    changed = deepcopy(first)
    changed.update(answer="D", id="other", audit={"notes": "arbitrary private data"})
    assert build_prompt(first) == build_prompt(changed)
    assert build_prompt(first) == build_prompt(first, "text_only")
    assert "secret" not in build_prompt(first)


def test_temporal_pair_mapping_matches_qwen_rounding():
    result = clip_map(record()["conditions"]["original"])
    assert "groups 1-2; timestamps <0.2 seconds>, <1.2 seconds>" in result
    assert "groups 3-4; timestamps <2.2 seconds>, <3.2 seconds>" in result


@pytest.mark.parametrize("ranges", [[[0, 3], [3, 8]], [[0, 4], [6, 8]], [[0, 10]], []])
def test_reject_invalid_temporal_ranges(ranges):
    selected = record()["conditions"]["original"]
    selected["clip_ranges"] = ranges
    with pytest.raises(ValueError):
        clip_map(selected)


def test_token_prefix_is_exact():
    assert continuation_start([1, 2, 3], [1, 2, 3, 9, 10]) == 3
    with pytest.raises(ValueError):
        continuation_start([1, 2, 3], [1, 2, 4, 9])
    with pytest.raises(ValueError):
        continuation_start([1, 2, 3], [1, 2, 3])


def test_last_valid_position_handles_left_and_right_padding():
    assert last_valid_position([0, 0, 1, 1, 1]) == 4
    assert last_valid_position([1, 1, 1, 0, 0]) == 2
    with pytest.raises(ValueError):
        last_valid_position([0, 0])
