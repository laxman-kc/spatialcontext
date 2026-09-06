"""Provided annotation validation and conservative two-dimensional relations."""

from copy import deepcopy
import json
from types import SimpleNamespace

from PIL import Image
import pytest

from ecqa.memory import MemoryStore
from ecqa.memory_model import MemoryRuntime
from ecqa.retrieval import make_memory_record, retrieve
from ecqa.spatial import format_spatial_evidence, normalize_spatial


def obj(identifier="a", box=None, offset=0):
    return {"object_id": identifier, "label": "provided object", "frame_offset": offset,
            "bbox": box or [0.1, 0.1, 0.3, 0.3], "status": "ai_inferred"}


def test_axes_are_frame_local_and_symmetric():
    objects = [obj(), obj("b", [0.5, 0.5, 0.9, 0.9]), obj("c", offset=1)]
    value = normalize_spatial({"objects": objects})
    assert value["relations"] == [
        {"frame_offset": 0, "subject_id": "a", "object_id": "b", "horizontal": "left", "vertical": "above"},
        {"frame_offset": 0, "subject_id": "b", "object_id": "a", "horizontal": "right", "vertical": "below"},
    ]
    assert "not depth or world geometry" in value["annotation_notice"]
    assert json.loads(json.dumps(value, allow_nan=False)) == value
    assert normalize_spatial({"objects": list(reversed(objects))}) == value


@pytest.mark.parametrize("box", [
    [0.2, 0.2, 0.4, 0.4], [0.3, 0.3, 0.5, 0.5], [0.32, 0.32, 0.5, 0.5],
    [0.0, 0.0, 0.9, 0.9],
])
def test_overlap_touching_containment_and_near_gaps_are_unknown(box):
    result = normalize_spatial({"objects": [obj(), obj("b", box)]})
    assert all(row["horizontal"] == row["vertical"] == "unknown" for row in result["relations"])


def test_horizontal_relation_does_not_invent_vertical_order():
    result = normalize_spatial({"objects": [obj(), obj("b", [0.6, 0.15, 0.9, 0.4])]})
    assert result["relations"][0]["horizontal"] == "left"
    assert result["relations"][0]["vertical"] == "unknown"


@pytest.mark.parametrize("field,value", [
    ("bbox", [0, 0, float("nan"), 1]), ("bbox", [0, 0, float("inf"), 1]),
    ("bbox", [-0.1, 0, 0.3, 1]), ("bbox", [0.3, 0.1, 0.1, 0.4]),
    ("bbox", [0, 0, 0, 1]), ("bbox", [False, 0, 1, 1]), ("bbox", [0, 0, 1]),
    ("confidence", float("nan")), ("confidence", 1.1), ("confidence", True),
    ("status", "automatically_verified"), ("frame_offset", 2), ("frame_offset", True),
    ("object_id", ""), ("label", "\nspoofed"),
])
def test_invalid_annotation_rejected(field, value):
    annotation = obj()
    annotation[field] = value
    with pytest.raises(ValueError):
        normalize_spatial({"objects": [annotation]})


def test_no_questions_answers_or_supplied_relations_allowed():
    for field in ["question", "answer", "target_clip", "relations"]:
        with pytest.raises(ValueError):
            normalize_spatial({"objects": [obj()], field: "secret"})
        annotation = obj()
        annotation[field] = "secret"
        with pytest.raises(ValueError):
            normalize_spatial({"objects": [annotation]})


def test_ids_are_unique_per_frame_without_assuming_tracking():
    with pytest.raises(ValueError, match="unique"):
        normalize_spatial({"objects": [obj(), obj()]})
    result = normalize_spatial({"objects": [obj(), obj(offset=1)]})
    assert len(result["objects"]) == 2 and not result["relations"]


def test_confidence_status_and_tampering_are_explicit():
    annotation = {**obj(), "confidence": 0.75, "status": "human_verified"}
    result = normalize_spatial({"objects": [annotation, obj("b", [0.5, 0.1, 0.8, 0.3])]})
    text = format_spatial_evidence(result)
    assert "supplied status human_verified" in text
    assert "uncalibrated confidence 0.75" in text
    assert "not automatically extracted or verified" in text
    changed = deepcopy(result)
    changed["relations"][0]["horizontal"] = "right"
    with pytest.raises(ValueError, match="regenerated"):
        format_spatial_evidence(changed)


def test_empty_annotations_are_valid_and_not_invented():
    result = normalize_spatial({"objects": []})
    assert not result["objects"] and not result["relations"]
    assert "No supplied spatial annotations" in format_spatial_evidence(result)


def test_reader_requires_explicit_opt_in_and_uses_only_selected_frame_annotations(tmp_path):
    frames = []
    for index in range(2):
        path = tmp_path / f"source{index}.png"
        Image.new("RGB", (128, 128), "blue").save(path)
        frames.append(path)
    with MemoryStore.create(tmp_path / "memory", capacity_pairs=2) as store:
        for clip in [1, 2]:
            annotation = {**obj(), "label": f"supplied-landmark-clip{clip}"}
            store.observe(episode_id="video", scene_id=f"scene{clip}", clip_index=clip,
                          pair_index=clip - 1, timestamp=clip - 0.75, source_id=f"source{clip}",
                          frames=frames, spatial={"objects": [annotation]})
        store.seal()
        question = "In the second clip, which object is left of the road?"
        result = retrieve(store, episode_id="video", question=question)
        record = make_memory_record(question, {label: label for label in "ABCD"}, result)
        runtime = MemoryRuntime.__new__(MemoryRuntime)
        runtime.data_root = store.root
        runtime.config = {}
        runtime.processor = SimpleNamespace(video_processor=SimpleNamespace(patch_size=16, merge_size=2))
        messages, _, _ = runtime._materialize(record)
        text = messages[0]["content"][-1]["text"]
        assert "supplied-landmark" not in text
        runtime.config = {"memory": {"use_spatial": True}}
        messages, _, _ = runtime._materialize(record)
        text = messages[0]["content"][-1]["text"]
        assert "supplied-landmark-clip2" in text and "supplied-landmark-clip1" not in text
        assert "pair frame_offset 0 is local reference frame 1" in text
        assert "x increases rightward and y downward" in text
        assert "not automatically extracted or verified" in text
        empty = make_memory_record(question, record["options"],
                                   retrieve(store, episode_id="video", question=question, policy="empty"))
        assert "supplied-landmark" not in runtime._materialize(empty)[0][0]["content"][-1]["text"]
        runtime.config = {"memory": {"use_spatial": "true"}}
        with pytest.raises(ValueError, match="boolean"):
            runtime._materialize(record)
