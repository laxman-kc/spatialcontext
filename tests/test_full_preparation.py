import copy
import json
from pathlib import Path

import av
import pytest
from PIL import Image

from ecqa.data import prepare_record, sha256_file
from scripts.prepare_full_dataset import (
    deterministic_queue,
    durable_artifact_manifest,
    record_from_donor_bank,
    scan_clip_sources,
    supplemental_exposure_screen,
    verify_129_boundaries,
    verify_completed,
)


def candidate(identifier, dataset="test", target=1, earlier=True):
    return {
        "id": identifier,
        "dataset": dataset,
        "target_clip": target,
        "is_earlier_candidate": earlier,
        "answer": "A",
    }


def test_queue_uses_all_possible_nonfinal_refs_and_preserves_manual_resolution():
    rows = [
        candidate("test-old"),
        candidate("test-first"),
        candidate("test-second", target=2),
        candidate("test-final", target=3, earlier=False),
        candidate("train-third", "train", 3, None),
        candidate("train-fourth", "train", 4, None),
        candidate("train-unresolved", "train", None, None),
    ]
    queue, excluded = deterministic_queue(rows, 42, {"test-old"})
    assert {row["id"] for row in queue} == {"test-first", "test-second", "train-third", "train-unresolved"}
    assert {row["id"] for row in excluded} == {"test-old", "test-final", "train-fourth"}
    changed = copy.deepcopy(rows)
    for row in changed:
        row["answer"] = "D"
    another, _ = deterministic_queue(list(reversed(changed)), 42, {"test-old"})
    assert [row["id"] for row in queue] == [row["id"] for row in another]


def test_frame_divisibility_does_not_approve_boundaries():
    assert not verify_129_boundaries(258, {}, 2)["supported"]
    assert verify_129_boundaries(258, {129: 50.0}, 2)["supported"]
    assert not verify_129_boundaries(258, {129: 50.0, 60: 60.0}, 2)["supported"]
    assert not verify_129_boundaries(258, {129: 50.0}, 3)["supported"]
    assert not verify_129_boundaries(257, {129: 50.0}, 2)["supported"]


def make_source(path: Path):
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=24)
        stream.width = stream.height = 128
        stream.pix_fmt = "yuv420p"
        for index in range(258):
            color = (220, 20, 10) if index < 129 else (10, 220, 20)
            frame = av.VideoFrame.from_image(Image.new("RGB", (128, 128), color))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_exact_full_clip_hashes_donor_bank_and_review_status(tmp_path):
    video = tmp_path / "source.mp4"
    make_source(video)
    scan = scan_clip_sources(video, tmp_path / "scan", tmp_path, 2, "example")
    duplicate = scan_clip_sources(video, tmp_path / "duplicate", tmp_path, 2, "other-name", False)
    assert scan["source_clip_fingerprints"] == duplicate["source_clip_fingerprints"]
    assert len(scan["source_clip_fingerprints"]) == 2
    assert scan["source_clip_fingerprints"][0] != scan["source_clip_fingerprints"][1]
    assert scan["status"] == "needs_review" and scan["reviewer"] is None
    assert scan["source_media_sha256"] == sha256_file(video)
    assert all(len(clip) == 6 for clip in scan["donor_bank"])
    assert [frame["clip_local_frame"] for frame in scan["donor_bank"][0]] == [25, 43, 51, 77, 86, 103]
    assert (tmp_path / "scan/boundary_pairs.jpg").is_file()
    row = {
        "id": "test:synthetic",
        "dataset": "test",
        "dataset_revision": "fixture",
        "source_group": "fixture-source",
        "source_path": "source.mp4",
        "target_clip": 1,
        "question": "Which color is visible in the first clip?",
        "options": {"A": "red", "B": "green", "C": "blue", "D": "yellow"},
        "answer": "A",
    }
    original = prepare_record(row, video, [[0, 129], [129, 258]], tmp_path, split="test")
    review = {
        "boundaries_verified": True,
        "reviewer_type": "human",
        "reviewer": "fixture reviewer",
        "reason": "Controlled fixture has a single deliberate color transition at129.",
        "evidence_artifacts": ["scan/boundary_pairs.jpg"],
    }
    reconstructed = record_from_donor_bank(row, scan, tmp_path, review)
    assert reconstructed["provenance"]["prepared_sha256"] == original["provenance"]["prepared_sha256"]
    assert reconstructed["provenance"]["frame_indices"] == original["provenance"]["frame_indices"]
    assert reconstructed["audit"]["status"] == "needs_review"
    with pytest.raises(ValueError, match="explicit evidenced"):
        record_from_donor_bank(row, scan, tmp_path, {})


def test_completed_artifact_integrity_is_required_for_resume(tmp_path):
    folder = tmp_path / "item"
    folder.mkdir()
    file = folder / "evidence.json"
    file.write_text('{"status":"needs_review"}')
    record = {"artifacts": durable_artifact_manifest(folder, tmp_path)}
    verify_completed(record, tmp_path)
    file.write_text("changed")
    with pytest.raises(ValueError, match="changed or missing"):
        verify_completed(record, tmp_path)


def test_supplemental_prior_training_source_quarantines_test_alias(tmp_path):
    raw = tmp_path / "raw/media/train_source.mp4"
    raw.parent.mkdir(parents=True)
    make_source(raw)
    normalized = tmp_path / "normalized"
    normalized.mkdir()
    prior = {"dataset": "train", "id": "train:prior", "source_path": "public/source.mp4", "clip_count": 2}
    (normalized / "train.jsonl").write_text(json.dumps(prior) + "\n")
    (normalized / "test.jsonl").write_text("")
    output = tmp_path / "fullstudy/preparation"
    scan_clip_sources(raw, output / "review/new-test", tmp_path, 2, "test:different-name", False)
    primary = output / "exposure_index.json"
    primary.write_text('{"original":"preserve"}')
    summary = supplemental_exposure_screen(tmp_path, output)
    decision = json.loads((output / "supplemental_test_eligibility.jsonl").read_text())
    assert summary["prior_videos"] == 1 and summary["prior_chunks"] == 2
    assert summary["test_videos_quarantined"] == 1
    assert decision["id"] == "test:different-name"
    assert not decision["eligible_after_supplemental_exposure_screen"]
    assert len(decision["exact_chunk_links"]) == 2
    assert primary.read_text() == '{"original":"preserve"}'
