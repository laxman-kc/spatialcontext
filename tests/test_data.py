import copy
import pytest
from PIL import Image

from ecqa.data import (assert_condition_invariants, assert_split_separation, canvas_size,
                       choose_candidates, create_conditions, download_file, load_manifest,
                       index_video, media_url, normalize_annotations, prepare_record, prepared_clip_ranges, safe_path,
                       sample_frame_indices, source_groups, validate_record, write_jsonl)


def annotation(identifier="sample", target="first", count=3):
    return {"question_id": identifier, "video_path": "UAVideo/AirScape/sample.mp4",
            "task_type": "positional_relationship", "concat_num": count,
            "question": f"In the {target} clip, what is to the left of the bridge?",
            "options": {"A": "road", "B": "tree", "C": "water", "D": "building"}, "answer": "B"}


@pytest.fixture
def record(tmp_path):
    paths = []
    for index in range(8):
        path = tmp_path / f"frame_{index:04d}.png"
        Image.new("RGB", (128, 128), (index * 20, 10, 20)).save(path)
        paths.append(path.name)
    clip_ranges = [[0, 4], [4, 6], [6, 8]]
    return {"id": "example", "split": "test", "source_group": "unresolved:example",
            "question": "In the first clip what is to the left of the bridge?",
            "options": {"A": "road", "B": "tree", "C": "water", "D": "building"}, "answer": "B",
            "audit": {"status": "needs_review", "reviewer": None, "target_clip": 1},
            "conditions": {"original": {"frames": paths, "clip_ranges": clip_ranges, "fps": 2.0},
                           "text_only": {"frames": [], "clip_ranges": clip_ranges, "fps": 2.0}}}


def donors(tmp_path, name, color, count=2):
    paths = []
    for index in range(count):
        path = tmp_path / f"{name}_{index}.png"
        Image.new("RGB", (192, 96), color).save(path)
        paths.append(path)
    return paths


def test_normalization_preserves_raw_and_unknowns():
    raw = annotation()
    row = normalize_annotations([raw], "test")[0]
    assert row["is_earlier_candidate"] is True
    assert row["target_clip"] == 1
    assert row["audit"]["status"] == "needs_review"
    assert row["provenance_status"] == "unknown"
    row["raw"]["options"]["B"] = "changed"
    assert raw["options"]["B"] == "tree"


def test_training_missing_boundaries_not_invented():
    row = annotation()
    row["id"] = row.pop("question_id")
    row.pop("concat_num")
    row["video_path"] = "AirScape_dataset/sample.mp4"
    normalized = normalize_annotations([row], "train")[0]
    assert normalized["clip_count"] is None
    assert normalized["is_earlier_candidate"] is None
    assert "Revision=edc34" in media_url(normalized)


def test_invalid_annotation_duplicate_and_final_filter():
    with pytest.raises(ValueError, match="duplicate"):
        normalize_annotations([annotation(), annotation()], "test")
    rows = normalize_annotations([annotation("earlier"), annotation("final", "third")], "test")
    assert [r["id"] for r in choose_candidates(rows, 5)] == ["test:earlier"]
    assert "/video/AirScape/sample.mp4" in media_url(rows[0])


def test_candidate_sampling_order_independent_seeded():
    rows = normalize_annotations([annotation(str(i)) for i in range(10)], "test")
    assert choose_candidates(rows, 4) == choose_candidates(list(reversed(rows)), 4)


@pytest.mark.parametrize("bounds,expected", [
    ([[0, 100], [100, 200]], [20, 40, 60, 80, 120, 140, 160, 180]),
    ([[0, 129], [129, 258], [258, 387]], [25, 51, 77, 103, 172, 215, 301, 344]),
    ([[0, 12], [12, 24], [24, 36], [36, 48]], [4, 8, 16, 20, 28, 32, 40, 44]),
])
def test_sampling_exact_order_and_no_temporal_pair_crossing(bounds, expected):
    assert sample_frame_indices(bounds) == expected
    for left, right in zip(expected[::2], expected[1::2]):
        assert any(start <= left <= right < end for start, end in bounds)


@pytest.mark.parametrize("bounds", [[[0, 1], [1, 10]], [[0, 10], [11, 20]], [[0, 10]], [[-1, 10], [10, 20]]])
def test_bad_clip_boundaries_rejected(bounds):
    with pytest.raises(ValueError):
        sample_frame_indices(bounds)


def test_source_component_transitivity_and_known_leakage(record):
    groups = source_groups(["a", "b", "c", "d"], [("a", "b"), ("b", "c")])
    assert groups["a"] == groups["c"] != groups["d"]
    first = copy.deepcopy(record)
    second = copy.deepcopy(record)
    first.update(id="a", split="train", source_group=groups["a"])
    second.update(id="c", split="test", source_group=groups["c"])
    with pytest.raises(ValueError, match="overlap"):
        assert_split_separation([first, second])
    with pytest.raises(ValueError, match="unresolved"):
        assert_split_separation([record], require_known=True)


def test_requires_review_by_default(record, tmp_path):
    path = tmp_path / "records.jsonl"
    write_jsonl(path, [record])
    with pytest.raises(ValueError, match="requires review"):
        load_manifest(path, data_root=tmp_path)
    assert len(load_manifest(path, data_root=tmp_path, require_approved=False)) == 1
    record["audit"]["status"] = "approved"
    with pytest.raises(ValueError, match="reviewer"):
        validate_record(record)


def test_approval_requires_real_review_fields(record):
    record["audit"].update({"status": "approved", "reviewer": "reviewer-name", "reviewer_type": "ai"})
    with pytest.raises(ValueError, match="boundaries_verified"):
        validate_record(record)
    record["audit"].update({"boundaries_verified": True, "evidence_verified": True,
                            "answer_verified": True, "viewpoint_verified": True,
                            "source_clip_ranges": [[0, 129], [129, 258], [258, 387]],
                            "target_clip": 1, "evidence_frame_range": [25, 104], "viewpoint": "image coordinates"})
    validate_record(record, require_approved=True)
    record["audit"]["target_clip"] = 3
    with pytest.raises(ValueError, match="nonfinal"):
        validate_record(record)


def test_missing_frame_and_path_escape_rejected(record, tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        safe_path(tmp_path, "../outside.png")
    with pytest.raises(ValueError, match="relative"):
        safe_path(tmp_path, "/outside.png")
    record["conditions"]["original"]["frames"][0] = "missing.png"
    with pytest.raises(ValueError, match="missing"):
        validate_record(record, data_root=tmp_path)


def test_later_donors_preserve_all_unselected_pixels(record, tmp_path):
    neutral = donors(tmp_path, "neutral", (200, 10, 10))
    competing = donors(tmp_path, "competing", (10, 200, 10))
    edited = create_conditions(record, {2: neutral}, {2: competing}, tmp_path)
    assert edited["audit"]["status"] == "needs_review"
    assert edited["conditions"]["neutral"]["frames"][:6] == record["conditions"]["original"]["frames"][:6]
    assert_condition_invariants(edited, tmp_path)
    changed = copy.deepcopy(edited)
    changed["conditions"]["neutral"]["frames"][0] = edited["conditions"]["neutral"]["frames"][6]
    with pytest.raises(ValueError):
        assert_condition_invariants(changed, tmp_path)


def test_target_or_unmatched_replacement_rejected(record, tmp_path):
    donor = donors(tmp_path, "donor", (200, 10, 10))
    with pytest.raises(ValueError, match="strictly later"):
        create_conditions(record, {0: donor}, {0: donor}, tmp_path)
    with pytest.raises(ValueError, match="same nonempty"):
        create_conditions(record, {1: donor}, {2: donor}, tmp_path)
    with pytest.raises(ValueError, match="exactly"):
        create_conditions(record, {2: donor[:1]}, {2: donor[:1]}, tmp_path)


def test_canvas_dimensions_are_patch_compatible():
    width, height = canvas_size(1920, 1080)
    assert width % 32 == height % 32 == 0
    assert 128 * 128 <= width * height <= 512 * 512
    assert prepared_clip_ranges(3) == [[0, 4], [4, 6], [6, 8]]


def test_download_single_file_hash_and_size_gate(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("selected file")
    destination = tmp_path / "out.txt"
    metadata = download_file(source.as_uri(), destination, max_bytes=100)
    assert metadata["bytes"] == len("selected file")
    with pytest.raises(ValueError, match="hash mismatch"):
        download_file(source.as_uri(), destination, expected_sha256="wrong", max_bytes=100)
    with pytest.raises(ValueError, match="byte limit"):
        download_file(source.as_uri(), tmp_path / "too_small.txt", max_bytes=1)
    assert not (tmp_path / "too_small.txt").exists()


def test_gold_audit_mutations_do_not_change_sampling():
    # Sampler's function signature cannot accept gold/question/evidence metadata.
    record = {"gold": "A", "evidence": [20, 40], "bounds": [[0, 129], [129, 258], [258, 387]]}
    first = sample_frame_indices(record["bounds"])
    record.update(gold="D", evidence=[70, 80])
    assert sample_frame_indices(record["bounds"]) == first


def test_real_codec_decode_prepare_and_corrupt_source(tmp_path):
    import av
    video = tmp_path / "source.mp4"
    with av.open(str(video), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=24)
        stream.width = stream.height = 128
        stream.pix_fmt = "yuv420p"
        for index in range(24):
            frame = av.VideoFrame.from_image(Image.new("RGB", (128, 128), (index * 9, 30, 90)))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    candidate = normalize_annotations([annotation()], "test")[0]
    result = prepare_record(candidate, video, [[0, 8], [8, 16], [16, 24]], tmp_path, split="test")
    assert result["audit"]["status"] == "needs_review"
    assert result["provenance"]["decoded_frame_count"] == 24
    assert result["provenance"]["frame_indices"] == [1, 3, 4, 6, 10, 13, 18, 21]
    assert result["provenance"]["source_pts"][0] == pytest.approx(1 / 24)
    for path in result["conditions"]["original"]["frames"]:
        with Image.open(tmp_path / path) as image:
            assert image.mode == "RGB" and image.size == (128, 128) and image.format == "PNG"
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"not a video")
    with pytest.raises(Exception):
        index_video(corrupt)
