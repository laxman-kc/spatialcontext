"""Source partition checks independent of model predictions."""
from copy import deepcopy

import pytest

from scripts.assemble_full_study import assemble, held_training_audit, record_roots, secondary_vetoes
from scripts.build_full_conditions import donor_frames


def record(identifier, clips, split="train", donors=()):
    return {"id": identifier, "split": split, "provenance": {"source_clip_fingerprints": clips},
            "audit": {"donor_source_clip_fingerprints": list(donors)}}


def state(records):
    clips = {value for row in records for value in
             [*row["provenance"]["source_clip_fingerprints"], *row["audit"]["donor_source_clip_fingerprints"]]}
    return {"approved": records, "clip_groups": {value: value for value in clips},
            "exposed_roots": set(), "quarantined_ids": set()}


def test_source_lookup_does_not_mutate_frozen_provenance():
    row = record("t", ["a"], "test", ["b"])
    before = deepcopy(row)
    assert record_roots(row, state([row])) == {"a", "b"}
    assert record_roots(row, state([row])) == {"a", "b"}
    assert row == before


def test_test_donor_overlap_excludes_whole_training_record():
    test = record("t", ["test-site"], "test", ["donor-site"])
    rows = [record("conflict", ["donor-site", "unrelated"]),
            record("safe1", ["a"]), record("safe2", ["b"]), record("safe3", ["c"])]
    result, summary = assemble(state(rows + [test]), [test], train_cap=2, val_cap=1)
    assert "conflict" not in {row["id"] for row in result}
    assert summary["counts"] == {"train": 2, "val": 1, "test": 1}
    sources = {}
    for row in result:
        for value in record_roots(row, state(rows + [test])):
            assert sources.setdefault(value, row["split"]) == row["split"]


def test_component_cap_never_moves_unused_members_to_other_split():
    rows = [record("shared1", ["same"]), record("shared2", ["same"]),
            record("separate", ["other"])]
    result, _ = assemble(state(rows), [], train_cap=1, val_cap=1)
    assert len({row["split"] for row in result if row["id"].startswith("shared")}) <= 1


def test_exposed_donor_cannot_enter_test_even_with_clean_target():
    test = record("t", ["new-target"], "test", ["prior"])
    current = state([test])
    current["exposed_roots"] = {"prior"}
    with pytest.raises(ValueError, match="Exposed test target/donor"):
        assemble(current, [test])


def test_large_shared_component_is_reserved_for_training():
    rows = [record(f"shared{i}", ["same"]) for i in range(8)]
    rows += [record(f"single{i}", [f"other{i}"]) for i in range(2)]
    result, summary = assemble(state(rows), [], train_cap=8, val_cap=2)
    assert summary["counts"] == {"train": 8, "val": 2, "test": 0}
    assert all(row["split"] == "train" for row in result if row["id"].startswith("shared"))


def test_donor_exposure_and_boundary_checks_precede_materialization(tmp_path):
    donor = {"id": "donor", "clip_index": 1, "source_clip_fingerprint": "abc",
             "boundaries_verified": True}
    current = {"prior_ids": set(), "scans": {"donor": {"source_clip_fingerprints": ["abc"]}},
               "clip_groups": {"abc": "shared"}, "exposed_roots": {"shared"}}
    with pytest.raises(ValueError, match="known prior exposure"):
        donor_frames(tmp_path, current, donor, 2)
    current["exposed_roots"] = set()
    with pytest.raises(ValueError, match="explicit boundary review"):
        donor_frames(tmp_path, current, {**donor, "boundaries_verified": False}, 2)


def test_exact_donor_bank_corruption_is_rejected(tmp_path):
    from ecqa.data import sha256_file
    bank = []
    for index in (43, 86):
        path = tmp_path / f"{index}.png"
        path.write_bytes(b"original image contents")
        bank.append({"frame": index, "path": path.name, "sha256": sha256_file(path)})
    current = {"prior_ids": set(), "scans": {"donor": {
        "source_clip_fingerprints": ["abc"], "donor_bank": [bank], "source_media_sha256": "video"}},
        "clip_groups": {"abc": "shared"}, "exposed_roots": set()}
    donor = {"id": "donor", "clip_index": 1, "source_clip_fingerprint": "abc", "boundaries_verified": True}
    paths, receipt = donor_frames(tmp_path, current, donor, 2)
    assert receipt["source_frame_indices"] == [43, 86]
    paths[1].write_bytes(b"substituted image contents")
    with pytest.raises(ValueError, match="changed exact donor"):
        donor_frames(tmp_path, current, donor, 2)


def test_secondary_label_review_is_content_bound_and_veto_only(monkeypatch):
    monkeypatch.setattr("scripts.assemble_full_study.review_identity", lambda row: row["identity"])
    prepared = {"example": {"identity": "original"}}
    decision = {"id": "example", "review_identity": "original", "decision": "accept",
                "reviewer": "second reviewer", "reviewer_type": "ai", "scope": "labels only",
                "reason": "The alternatives were inspected on exact frames."}
    assert secondary_vetoes(prepared, [decision]) == set()
    rejected = {**decision, "decision": "reject"}
    assert secondary_vetoes(prepared, [decision, rejected]) == {"example"}
    with pytest.raises(ValueError, match="changed or unknown"):
        secondary_vetoes(prepared, [{**rejected, "review_identity": "stale"}])


def test_held_audit_distinguishes_missing_review_metadata_and_changed_source(tmp_path):
    from ecqa.artifacts import file_digest
    from ecqa.data import write_jsonl
    folder = tmp_path / "source"
    folder.mkdir()
    scan = folder / "source_scan.json"
    scan.write_text('{"source": "original"}')
    candidate = {"id": "held", "candidate_folder": str(folder),
                 "source_scan_sha256": file_digest(scan), "target_preview": "review.png"}
    metadata = {"id": "final", "source_scan_sha256": "metadata-bound",
                "triage": "decoded_final_target"}
    write_jsonl(tmp_path / "fullstudy/held-training/queue.jsonl", [candidate, metadata])
    audited, pending = held_training_audit(tmp_path)
    assert pending == ["held"]
    assert audited[0]["decision"] == "metadata_exclusion"
    assert audited[0]["visually_reviewed"] is False
    decision = {"id": "held", "source_scan_sha256": file_digest(scan), "decision": "reject",
                "reviewer": "visual reviewer", "reviewer_type": "ai", "reason": "Two listed objects match."}
    write_jsonl(tmp_path / "fullstudy/reviews/test-held-training-audit.jsonl", [decision])
    audited, pending = held_training_audit(tmp_path)
    assert pending == []
    assert audited[0]["decision"] == "reject"
    scan.write_text('{"source": "changed after review"}')
    with pytest.raises(ValueError, match="source scan changed"):
        held_training_audit(tmp_path)
