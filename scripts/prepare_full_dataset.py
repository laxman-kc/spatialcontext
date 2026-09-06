#!/usr/bin/env python3
"""Resumable bounded preparation; outputs require independent visual review.

Only raw files newly downloaded into this run's scratch directory are removed,
and only after provenance, PNGs/review panels, hashes and completion are durable.
Prior feasibility artifacts are read-only. No model predictions are consumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

from ecqa.data import (
    MODELSCOPE_REVISION,
    canvas_size,
    download_file,
    fit_canvas,
    json_digest,
    media_url,
    prepared_clip_ranges,
    prepare_record,
    read_jsonl,
    safe_path,
    sample_frame_indices,
    sha256_file,
    validate_record,
    write_jsonl,
)
from scripts.acquire_data import prepared_review, review_media

POLICY = "full-study-preparation-v1"
FINGERPRINT_METHOD = "rgb24-all-frames-sha256-v1"
PRIOR_TEST_IDS = {
    "test:positional_relationship_0103",
    "test:positional_relationship_0188",
    "test:positional_relationship_0055",
    "test:positional_relationship_0088",
    "test:positional_relationship_0002",
}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with part.open("w") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    part.replace(path)


def deterministic_queue(
    records: list[dict], seed: int, excluded_ids: set[str]
) -> tuple[list[dict], list[dict]]:
    retained, excluded = [], []
    for record in records:
        reason = None
        if record["dataset"] == "test" and record["id"] in excluded_ids:
            reason = "previously_exposed_test_id"
        elif record["dataset"] == "test" and record["is_earlier_candidate"] is not True:
            reason = "metadata_final_clip"
        elif record["dataset"] == "train" and record["target_clip"] == 4:
            reason = "fourth_clip_cannot_be_nonfinal_under_two_to_four_clip_contract"
        if reason:
            excluded.append({"id": record["id"], "reason": reason})
        else:
            retained.append(record)
    retained.sort(
        key=lambda row: (
            row["dataset"] != "test",
            row["target_clip"] is None,
            hashlib.sha256(f"{seed}:{row['dataset']}:{row['id']}".encode()).hexdigest(),
        )
    )
    return [dict(row, queue_index=index) for index, row in enumerate(retained)], excluded


def verify_129_boundaries(
    frame_count: int, differences: dict[int, float], annotated_count: int | None
) -> dict:
    count = frame_count // 129
    if frame_count % 129 or count not in (2, 3, 4):
        return {"supported": False, "reason": "frame_count_not_two_to_four_129_frame_clips", "ranges": []}
    if annotated_count is not None and count != annotated_count:
        return {"supported": False, "reason": "annotation_clip_count_mismatch", "ranges": []}
    expected = {129 * index for index in range(1, count)}
    observed = {index for index, value in differences.items() if value > 30.0}
    missing, extra = sorted(expected - observed), sorted(observed - expected)
    ranges = [[129 * index, 129 * (index + 1)] for index in range(count)]
    return {
        "supported": not missing and not extra,
        "reason": "all_expected_cuts_supported" if not missing and not extra else "boundary_review_required",
        "ranges": ranges,
        "missing_expected_cuts": missing,
        "unexpected_cuts": extra,
        "method": "consecutive RGB mean absolute difference>30 at64x36; visual reviewer still required",
    }


def scan_clip_sources(
    video: Path,
    output: Path,
    root: Path,
    annotated_count: int | None,
    source_id: str,
    save_donor_bank: bool = True,
) -> dict:
    """Exact all-RGB-frame hashes are distinct from review-only coarse descriptors."""
    output.mkdir(parents=True, exist_ok=True)
    hashes, descriptors, donors, differences, timestamps = [], [], [], {}, []
    digest = None
    small_frames, donor_frames = [], []
    previous = None
    count = 0
    size = None
    for_boundary = []
    last_image = None
    with av.open(str(video)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("missing source PTS")
            image = frame.to_image().convert("RGB")
            if size is None:
                size = canvas_size(image.width, image.height)
            clip_index, local_index = divmod(index, 129)
            if local_index == 0:
                if digest is not None:
                    hashes.append(digest.hexdigest())
                    descriptors.append(small_frames)
                    donors.append(donor_frames)
                header = {
                    "method": FINGERPRINT_METHOD,
                    "width": image.width,
                    "height": image.height,
                    "expected_frame_count": 129,
                }
                digest = hashlib.sha256(
                    json.dumps(header, sort_keys=True, separators=(",", ":")).encode() + b"\n"
                )
                small_frames, donor_frames = [], []
            digest.update(image.tobytes())
            small = np.asarray(image.resize((64, 36))).astype(np.int16)
            difference = float(np.abs(small - previous).mean()) if previous is not None else 0.0
            if difference > 30 or local_index == 0:
                differences[index] = difference
            if local_index == 0 and index and last_image is not None:
                for_boundary.extend([(index - 1, last_image), (index, image.resize((384, 216)))])
            previous = small
            last_image = image.resize((384, 216))
            timestamps.append(float(frame.pts * frame.time_base))
            small_frames.append(np.asarray(image.resize((32, 18))).astype(np.uint8))
            # Four interior donor frames plus the two-frame sampling positions.
            if save_donor_bank and local_index in (25, 43, 51, 77, 86, 103):
                path = output / "donor_bank" / f"clip_{clip_index + 1:02d}" / f"source_{index:05d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                fit_canvas(image, size).save(path, format="PNG")
                donor_frames.append(
                    {
                        "frame": index,
                        "clip_local_frame": local_index,
                        "pts": timestamps[-1],
                        "path": path.relative_to(root).as_posix(),
                        "sha256": sha256_file(path),
                    }
                )
            count = index + 1
    if digest is None:
        raise ValueError("no decoded frames")
    hashes.append(digest.hexdigest())
    descriptors.append(small_frames)
    donors.append(donor_frames)
    verification = verify_129_boundaries(count, differences, annotated_count)
    if for_boundary:
        panel = Image.new("RGB", (768, len(for_boundary) // 2 * 242), "white")
        draw = ImageDraw.Draw(panel)
        for position, (index, image) in enumerate(for_boundary):
            x, y = position % 2 * 384, position // 2 * 242
            panel.paste(image, (x, y))
            draw.text((x + 3, y + 218), f"source frame {index}", fill="black")
        panel.save(output / "boundary_pairs.jpg", quality=88)
    descriptor_path = output / "clip_review_descriptors.npz"
    np.savez_compressed(
        descriptor_path, **{f"clip_{i + 1}": np.stack(values) for i, values in enumerate(descriptors)}
    )
    result = {
        "source_id": source_id,
        "decoded_frame_count": count,
        "source_pts": timestamps,
        "boundary_verification": verification,
        "source_clip_fingerprint_method": FINGERPRINT_METHOD,
        "source_clip_fingerprints": hashes if verification["supported"] else [],
        "tentative_129_chunk_fingerprints": hashes,
        "source_media_sha256": sha256_file(video),
        "review_descriptors": descriptor_path.relative_to(root).as_posix(),
        "review_descriptor_semantics": "perceptual review aids, never exact identity",
        "donor_bank": donors,
        "canvas_wh": list(size),
        "status": "needs_review",
        "reviewer": None,
    }
    atomic_json(output / "source_scan.json", result)
    return result


def find_prior_test_records(root: Path) -> list[dict]:
    found = {}
    for name in ("candidates.jsonl", "sample_needs_review.jsonl", "approved.jsonl"):
        path = root / "manifests" / name
        if path.exists():
            for record in read_jsonl(path):
                if record["id"] in PRIOR_TEST_IDS:
                    found.setdefault(record["id"], record)
    return list(found.values())


def prior_exposure_index(root: Path, output: Path) -> dict:
    scans = []
    for record in find_prior_test_records(root):
        name = record["id"].replace(":", "_")
        location = output / "prior_exposure" / name
        if (location / "source_scan.json").exists():
            scans.append(json.loads((location / "source_scan.json").read_text()))
            continue
        source = record.get("source_path", record.get("provenance", {}).get("source_path"))
        if not source:
            raise ValueError(f"missing prior exposed source locator: {record['id']}")
        path = root / "raw" / "media" / ("test_" + Path(source).name)
        if not path.exists():
            raise ValueError(f"prior exposed raw source is unavailable: {path}")
        annotated = record.get("clip_count", record.get("raw", {}).get("concat_num"))
        scans.append(scan_clip_sources(path, location, root, annotated, record["id"], save_donor_bank=False))
    index = {
        fingerprint: scan["source_id"] for scan in scans for fingerprint in scan["source_clip_fingerprints"]
    }
    return {"fingerprints": index, "scans": scans, "digest": json_digest(index)}


def near_exposure_links(scan: dict, exposure: dict, root: Path) -> list[dict]:
    """Conservative review quarantine using aligned sequence color AND contrast/texture.

    Dark-only mean-color matches are insufficient. These links never establish
    confirmed source identity; pending links prevent final-test eligibility.
    """
    links = []
    with np.load(safe_path(root, scan["review_descriptors"])) as current:
        for prior in exposure["scans"]:
            with np.load(safe_path(root, prior["review_descriptors"])) as reference:
                for left_name in current.files:
                    left = current[left_name].astype(np.float32)
                    for right_name in reference.files:
                        right = reference[right_name].astype(np.float32)
                        if left.shape != right.shape:
                            continue
                        mad = float(np.abs(left - right).mean())
                        if mad >= 6:
                            continue
                        a, b = left.ravel(), right.ravel()
                        std_a, std_b = float(a.std()), float(b.std())
                        correlation = float(np.corrcoef(a, b)[0, 1]) if min(std_a, std_b) > 3 else 0.0
                        if correlation >= 0.95:
                            links.append(
                                {
                                    "prior_id": prior["source_id"],
                                    "new_clip": left_name,
                                    "prior_clip": right_name,
                                    "mean_abs_rgb": mad,
                                    "pixel_correlation": correlation,
                                    "status": "needs_source_review",
                                }
                            )
    return links


def supplemental_exposure_screen(root: Path, output: Path) -> dict:
    """Preserve the original five-source index; separately quarantine all prior footage.

    This adds no approvals and changes no prepared tensors or immutable completion
    records. The final cohort assembler must also consume this eligibility ledger.
    """
    records = [
        row for dataset in ("train", "test") for row in read_jsonl(root / "normalized" / f"{dataset}.jsonl")
    ]
    by_filename = {row["dataset"] + "_" + Path(row["source_path"]).name: row for row in records}
    prior_scans = []
    for raw in sorted((root / "raw" / "media").glob("*.mp4")):
        if raw.name not in by_filename:
            raise ValueError(f"prior public raw source missing annotation locator: {raw.name}")
        row = by_filename[raw.name]
        folder = output / "supplemental_exposure" / row["id"].replace(":", "_")
        path = folder / "source_scan.json"
        if path.exists():
            scan = json.loads(path.read_text())
            if scan["source_media_sha256"] != sha256_file(raw):
                raise ValueError("prior raw exposure media changed")
        else:
            scan = scan_clip_sources(raw, folder, root, row.get("clip_count"), row["id"], False)
        prior_scans.append(scan)
    # An exact match to a129-frame chunk proves shared pixels even when a cut is
    # unresolved; label it as a chunk match rather than inventing clip boundaries.
    fingerprints = {}
    descriptor_items = []
    for scan in prior_scans:
        for index, digest in enumerate(scan["tentative_129_chunk_fingerprints"]):
            fingerprints.setdefault(digest, []).append(
                {
                    "id": scan["source_id"],
                    "chunk_index": index + 1,
                    "boundary_supported": scan["boundary_verification"]["supported"],
                }
            )
        with np.load(safe_path(root, scan["review_descriptors"])) as arrays:
            for name in arrays.files:
                descriptor_items.append((scan["source_id"], name, arrays[name].astype(np.float32)))
    index = {
        "version": "all-prior-local-public-footage-v1",
        "prior_ids": sorted(scan["source_id"] for scan in prior_scans),
        "source_media_sha256": sorted(scan["source_media_sha256"] for scan in prior_scans),
        "exact_129_frame_chunk_matches": fingerprints,
        "source_clip_fingerprint_method": FINGERPRINT_METHOD,
        "primary_five_source_index_preserved": True,
        "note": "Expanded before final cohort selection to all previously inspected or used local footage, including training/validation and intervention donors.",
    }
    index["digest"] = json_digest(index)
    atomic_json(output / "supplemental_exposure_index.json", index)
    decisions = []
    for path in sorted((output / "review").glob("*/source_scan.json")):
        scan = json.loads(path.read_text())
        if not scan["source_id"].startswith("test:"):
            continue
        exact = [
            {"new_chunk_index": position + 1, "sha256": digest, "prior": fingerprints[digest]}
            for position, digest in enumerate(scan["tentative_129_chunk_fingerprints"])
            if digest in fingerprints
        ]
        near = []
        with np.load(safe_path(root, scan["review_descriptors"])) as arrays:
            for name in arrays.files:
                left = arrays[name].astype(np.float32)
                for prior_id, prior_name, right in descriptor_items:
                    if left.shape != right.shape or len(left) != 129:
                        continue
                    if float(np.abs(left[[25, 64, 103]] - right[[25, 64, 103]]).mean()) >= 8:
                        continue
                    mad = float(np.abs(left - right).mean())
                    if mad >= 6 or min(float(left.std()), float(right.std())) <= 3:
                        continue
                    correlation = float(np.corrcoef(left.ravel(), right.ravel())[0, 1])
                    if correlation >= 0.95:
                        near.append(
                            {
                                "new_chunk": name,
                                "prior_id": prior_id,
                                "prior_chunk": prior_name,
                                "mean_abs_rgb": mad,
                                "pixel_correlation": correlation,
                                "status": "requires_visual_source_review",
                            }
                        )
        media_hit = scan["source_media_sha256"] in index["source_media_sha256"]
        decisions.append(
            {
                "id": scan["source_id"],
                "exposure_digest": index["digest"],
                "eligible_after_supplemental_exposure_screen": not (exact or near or media_hit),
                "known_media_overlap": media_hit,
                "exact_chunk_links": exact,
                "possible_source_links": near,
                "status": "quarantine" if exact or near or media_hit else "no_match_found",
                "claim_limit": "No detected overlap does not establish source independence; shifted shots and common sites require separate visual grouping.",
            }
        )
    write_jsonl(output / "supplemental_test_eligibility.jsonl", decisions)
    summary = {
        "prior_videos": len(prior_scans),
        "prior_chunks": sum(len(scan["tentative_129_chunk_fingerprints"]) for scan in prior_scans),
        "test_videos_screened": len(decisions),
        "test_videos_quarantined": sum(
            not row["eligible_after_supplemental_exposure_screen"] for row in decisions
        ),
        "exposure_digest": index["digest"],
    }
    atomic_json(output / "supplemental_exposure_summary.json", summary)
    return summary


def record_from_donor_bank(candidate: dict, scan: dict, root: Path, boundary_review: dict) -> dict:
    """Reconstruct exact sampler pixels after an explicit visual boundary review.

    The original automatic hold is preserved. This returns a new needs-review
    record; answer/evidence/viewpoint acceptance remains a separate decision.
    """
    if (
        boundary_review.get("boundaries_verified") is not True
        or boundary_review.get("reviewer_type") not in {"ai", "human"}
        or not boundary_review.get("reviewer")
        or not boundary_review.get("reason")
        or not boundary_review.get("evidence_artifacts")
    ):
        raise ValueError("explicit evidenced visual boundary review required")
    boundaries = scan["boundary_verification"]["ranges"]
    selected = sample_frame_indices(boundaries)
    bank = {frame["frame"]: frame for clip in scan["donor_bank"] for frame in clip}
    frames = [bank[index] for index in selected]
    for frame in frames:
        if sha256_file(safe_path(root, frame["path"])) != frame["sha256"]:
            raise ValueError("durable donor bank pixel artifact changed")
    ranges = prepared_clip_ranges(len(boundaries))
    record = {
        "id": candidate["id"],
        "split": candidate["dataset"],
        "source_group": candidate["source_group"],
        "question": candidate["question"],
        "options": candidate["options"],
        "answer": candidate["answer"],
        "option_permutation": {label: label for label in "ABCD"},
        "audit": {
            "status": "needs_review",
            "reviewer": None,
            "target_clip": candidate["target_clip"],
            "source_clip_ranges": boundaries,
            "boundary_status": "explicit_visual_boundary_review",
            "boundary_review": boundary_review,
            "automatic_boundary_hold": scan["boundary_verification"],
            "provenance_status": candidate.get("provenance_status", "unknown"),
        },
        "conditions": {
            "original": {"frames": [frame["path"] for frame in frames], "clip_ranges": ranges, "fps": 2.0},
            "text_only": {"frames": [], "clip_ranges": ranges, "fps": 2.0},
        },
        "provenance": {
            "dataset": candidate["dataset"],
            "revision": candidate["dataset_revision"],
            "source_media_sha256": scan["source_media_sha256"],
            "source_path": candidate["source_path"],
            "frame_indices": selected,
            "source_pts": [scan["source_pts"][i] for i in selected],
            "prepared_sha256": [frame["sha256"] for frame in frames],
            "decoder": f"PyAV {av.__version__}",
            "decoded_frame_count": scan["decoded_frame_count"],
            "canvas_hw": scan["canvas_wh"][::-1],
            "timeline": "synthetic prepared-frame order; not flight time",
            "provenance_status": candidate.get("provenance_status", "unknown"),
            "source_clip_fingerprints": scan["tentative_129_chunk_fingerprints"],
            "source_clip_fingerprint_method": FINGERPRINT_METHOD,
            "source_clip_review_descriptors": scan["review_descriptors"],
            "prepared_from": "durable lossless sampler-position donor-bank PNGs",
        },
    }
    validate_record(record, data_root=root)
    return record


def durable_artifact_manifest(folder: Path, root: Path) -> list[dict]:
    artifacts = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.name not in ("completion.json", "completion.json.part"):
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
            artifacts.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return artifacts


def verify_completed(record: dict, root: Path) -> None:
    for artifact in record["artifacts"]:
        path = safe_path(root, artifact["path"])
        if (
            not path.is_file()
            or path.stat().st_size != artifact["bytes"]
            or sha256_file(path) != artifact["sha256"]
        ):
            raise ValueError(f"completed artifact changed or missing: {path}")


def prepare_one(
    candidate: dict, folder: Path, root: Path, output: Path, exposure: dict, policy: dict, args
) -> None:
    complete_path = folder / "completion.json"
    reserve = int(args.minimum_free_gb * 1024**3) + args.workers * args.max_media_mb * 1024**2
    if shutil.disk_usage(root).free < reserve:
        print(
            json.dumps({"stage": "stopped", "id": candidate["id"], "reason": "free_space_reserve"}),
            flush=True,
        )
        return
    started = time.monotonic()
    folder.mkdir(parents=True, exist_ok=True)
    scratch = output / "scratch" / (json_digest(candidate["id"])[:20] + ".mp4")
    try:
        acquisition = download_file(media_url(candidate), scratch, max_bytes=args.max_media_mb * 1024**2)
        print(
            json.dumps({"stage": "downloaded", "id": candidate["id"], "bytes": acquisition["bytes"]}),
            flush=True,
        )
        scan = scan_clip_sources(scratch, folder, root, candidate.get("clip_count"), candidate["id"])
        review_media(scratch, folder, candidate.get("clip_count"))
        exact_hits = sorted(set(scan["source_clip_fingerprints"]) & set(exposure["fingerprints"]))
        near_hits = near_exposure_links(scan, exposure, root) if candidate["dataset"] == "test" else []
        target, boundaries = candidate.get("target_clip"), scan["boundary_verification"]["ranges"]
        reason = None
        if candidate["dataset"] == "test" and exact_hits:
            reason = "known_prior_test_constituent_overlap"
        elif near_hits:
            reason = "possible_prior_test_source_overlap_requires_review"
        elif not scan["boundary_verification"]["supported"]:
            reason = "boundaries_require_review"
        elif target is None:
            reason = "content_reference_requires_target_review"
        elif target >= len(boundaries):
            reason = "decoded_target_is_final_clip"
        prepared = None
        if reason is None:
            prepared = prepare_record(candidate, scratch, boundaries, root, split=candidate["dataset"])
            prepared["provenance"].update(
                source_clip_fingerprints=scan["source_clip_fingerprints"],
                source_clip_fingerprint_method=FINGERPRINT_METHOD,
                source_clip_review_descriptors=scan["review_descriptors"],
            )
            prepared["audit"].update(
                boundary_status="automatic_cut_supported_pending_visual_review",
                exposure_index_digest=exposure["digest"],
                source_overlap_review="unknown_remaining_provenance",
            )
            write_jsonl(folder / "prepared.jsonl", [prepared])
            prepared_review(prepared, root, folder)
        review = {
            "id": candidate["id"],
            "queue_index": candidate["queue_index"],
            "status": "needs_review",
            "question": candidate["question"],
            "options": candidate["options"],
            "answer": candidate["answer"],
            "target_clip": target,
            "candidate": candidate,
            "automatic_exclusion_or_hold": reason,
            "known_exposure_fingerprint_hits": exact_hits,
            "possible_exposure_links": near_hits,
            "source_scan": "source_scan.json",
            "source_locator": media_url(candidate),
            "source_media_sha256": acquisition["sha256"],
            "source_bytes": acquisition["bytes"],
            "prepared_record": "prepared.jsonl" if prepared else None,
            "raw_redownloadable": True,
            "reviewer": None,
            "human_approved": False,
        }
        atomic_json(folder / "review.json", review)
        artifacts = durable_artifact_manifest(folder, root)
        if prepared:
            for relative in prepared["conditions"]["original"]["frames"]:
                path = safe_path(root, relative)
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
                artifacts.append(
                    {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                )
        completion = {
            "id": candidate["id"],
            "policy_digest": json_digest(policy),
            "exposure_digest": exposure["digest"],
            "status": "complete_needs_review",
            "reason": reason,
            "artifacts": artifacts,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "temporary_raw_deleted": not args.keep_temporary_raw,
        }
        atomic_json(complete_path, completion)
        verify_completed(completion, root)
        if not args.keep_temporary_raw:
            scratch.unlink()
        print(
            json.dumps(
                {
                    "stage": "prepared",
                    "id": candidate["id"],
                    "queue_index": candidate["queue_index"],
                    "reason": reason,
                    "seconds": completion["elapsed_seconds"],
                    "review": str(folder),
                }
            ),
            flush=True,
        )
    except Exception as error:
        atomic_json(
            folder / "failure.json",
            {
                "id": candidate["id"],
                "error": f"{type(error).__name__}: {error}",
                "policy_digest": json_digest(policy),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        print(json.dumps({"stage": "failed", "id": candidate["id"], "error": str(error)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("fullstudy/preparation"))
    parser.add_argument("--dataset", choices=("train", "test", "all"), default="test")
    parser.add_argument("--max-items", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--priority-file", type=Path)
    parser.add_argument("--adopt-prior-policy", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-media-mb", type=int, default=256)
    parser.add_argument("--minimum-free-gb", type=float, default=3.0)
    parser.add_argument("--keep-temporary-raw", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--exposure-screen-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("workers must be between1 and4")
    if not 1 <= args.max_items <= 50:
        parser.error("each invocation must attempt between1 and50 files")
    root = args.data_root.resolve()
    output = safe_path(root, args.output.as_posix())
    output.mkdir(parents=True, exist_ok=True)
    if args.exposure_screen_only:
        print(json.dumps(supplemental_exposure_screen(root, output)), flush=True)
        return
    normalized_paths = [root / "normalized" / f"{dataset}.jsonl" for dataset in ("train", "test")]
    records = [row for path in normalized_paths for row in read_jsonl(path)]
    queue, exclusions = deterministic_queue(records, args.seed, PRIOR_TEST_IDS)
    if args.priority_file:
        priority = read_jsonl(args.priority_file)
        priority_ids = [row["id"] for row in priority]
        expected_ids = {row["id"] for row in queue if row["dataset"] == "test"}
        if len(set(priority_ids)) != len(priority_ids) or set(priority_ids) != expected_ids:
            raise ValueError("priority queue must contain exactly the eligible test IDs")
        rank = {identifier: index for index, identifier in enumerate(priority_ids)}
        queue.sort(key=lambda row: (row["dataset"] != "test", rank.get(row["id"], row["queue_index"])))
        queue = [dict(row, queue_index=index) for index, row in enumerate(queue)]
    policy = {
        "version": POLICY,
        "script_sha256": sha256_file(Path(__file__)),
        "seed": args.seed,
        "normalized_sha256": [sha256_file(path) for path in normalized_paths],
        "queue_digest": json_digest(queue),
        "prior_test_ids": sorted(PRIOR_TEST_IDS),
        "modelscope_revision": MODELSCOPE_REVISION,
        "fingerprint_method": FINGERPRINT_METHOD,
        "decoder": av.__version__,
        "frames": 8,
        "cut_threshold": 30.0,
        "donor_positions": [25, 43, 51, 77, 86, 103],
        "priority_queue_sha256": sha256_file(args.priority_file) if args.priority_file else None,
    }
    policy_path = output / "policy.json"
    if policy_path.exists():
        previous_policy = json.loads(policy_path.read_text())
        if previous_policy != policy:
            if not args.adopt_prior_policy:
                raise ValueError(
                    "resume identity changed; explicit --adopt-prior-policy or new output required"
                )
            compatible = (
                "version",
                "normalized_sha256",
                "prior_test_ids",
                "modelscope_revision",
                "fingerprint_method",
                "decoder",
                "frames",
                "cut_threshold",
                "donor_positions",
            )
            if any(previous_policy.get(key) != policy.get(key) for key in compatible):
                raise ValueError("preparation semantics changed; prior artifacts cannot be adopted")
            previous_digest = json_digest(previous_policy)
            atomic_json(output / "policies" / f"{previous_digest}.json", previous_policy)
            if (output / "queue.jsonl").exists():
                shutil.copyfile(
                    output / "queue.jsonl", output / "policies" / f"{previous_digest}.queue.jsonl"
                )
    atomic_json(output / "policies" / f"{json_digest(policy)}.json", policy)
    atomic_json(policy_path, policy)
    write_jsonl(output / "queue.jsonl", queue)
    write_jsonl(output / "metadata_exclusions.jsonl", exclusions)
    print(
        json.dumps(
            {
                "stage": "queue",
                "test": sum(r["dataset"] == "test" for r in queue),
                "train": sum(r["dataset"] == "train" for r in queue),
                "policy_digest": json_digest(policy),
            }
        ),
        flush=True,
    )
    if args.plan_only:
        return
    exposure = prior_exposure_index(root, output)
    atomic_json(
        output / "exposure_index.json",
        {
            "digest": exposure["digest"],
            "fingerprints": exposure["fingerprints"],
            "prior_ids": [scan["source_id"] for scan in exposure["scans"]],
        },
    )
    completed = {}
    for path in sorted((output / "review").glob("*/completion.json")):
        entry = json.loads(path.read_text())
        verify_completed(entry, root)
        if entry["id"] in completed:
            raise ValueError("duplicate completed ID")
        completed[entry["id"]] = (entry, path.parent)
    pending = [
        candidate
        for candidate in queue
        if candidate["id"] not in completed
        and (args.dataset == "all" or candidate["dataset"] == args.dataset)
    ][: args.max_items]
    reuses = [
        {
            "id": candidate["id"],
            "historical_policy": completed[candidate["id"]][0]["policy_digest"],
            "current_policy": json_digest(policy),
            "current_queue_index": candidate["queue_index"],
            "folder": completed[candidate["id"]][1].relative_to(root).as_posix(),
        }
        for candidate in queue
        if candidate["id"] in completed
    ]
    write_jsonl(output / "verified_reuse.jsonl", reuses)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for candidate in pending:
            folder = output / "review" / f"{candidate['queue_index']:04d}_{candidate['id'].replace(':', '_')}"
            futures.append(pool.submit(prepare_one, candidate, folder, root, output, exposure, policy, args))
        for future in as_completed(futures):
            future.result()
    attempted = len(pending)
    completions = [
        json.loads(path.read_text()) for path in sorted((output / "review").glob("*/completion.json"))
    ]
    write_jsonl(output / "ledger.jsonl", completions)
    prepared_records = [
        row for path in sorted((output / "review").glob("*/prepared.jsonl")) for row in read_jsonl(path)
    ]
    write_jsonl(output / "prepared_needs_review.jsonl", prepared_records)
    print(
        json.dumps({"stage": "supplemental_exposure", **supplemental_exposure_screen(root, output)}),
        flush=True,
    )
    print(
        json.dumps(
            {
                "stage": "batch_complete",
                "attempted": attempted,
                "total_complete": len(completions),
                "total_prepared_needs_review": len(prepared_records),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
