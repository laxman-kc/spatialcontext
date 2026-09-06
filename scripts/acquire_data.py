#!/usr/bin/env python3
"""Retrieve a bounded, seeded candidate pool and produce honest review artifacts.

No predictions are used for selection. Suggested cuts are never approved by this
script. Training defaults to refusing these records until review is complete.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecqa.data import (DATASETS, annotation_url, choose_candidates, download_file,
                       media_url, normalize_annotations, prepare_record, read_jsonl,
                       write_jsonl)


def prepared_review(record: dict, data_root: Path, output: Path) -> None:
    """Show actual final target-frame pixels at full resolution for review."""
    from PIL import Image, ImageDraw
    original = record["conditions"]["original"]
    start, end = original["clip_ranges"][record["audit"]["target_clip"] - 1]
    with Image.open(data_root / original["frames"][start]) as first:
        width, height = first.size
    panel = Image.new("RGB", (2 * width, ((end - start + 1) // 2) * (height + 26)), "white")
    draw = ImageDraw.Draw(panel)
    for offset, position in enumerate(range(start, end)):
        x, y = offset % 2 * width, offset // 2 * (height + 26)
        with Image.open(data_root / original["frames"][position]) as image:
            panel.paste(image, (x, y))
        source_index = record["provenance"]["frame_indices"][position]
        draw.text((x + 4, y + height + 3), f"prepared frame {position} | source frame {source_index}", fill="black")
    output.mkdir(parents=True, exist_ok=True)
    panel.save(output / "target_prepared.png")


def review_media(path: Path, output: Path, clip_count: int | None) -> dict:
    import av
    import numpy as np
    from PIL import Image, ImageDraw

    previous, frames, suggestions, thumbs, next_time = None, [], [], [], 0.0
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if frame.pts is None:
                raise ValueError("missing source timestamps")
            pts = float(frame.pts * frame.time_base)
            image = frame.to_image()
            small = np.asarray(image.resize((64, 36))).astype("float32")
            difference = float(np.abs(small - previous).mean()) if previous is not None else 0
            if difference > 30:
                suggestions.append({"frame": index, "pts": pts, "difference": difference})
            previous = small
            frames.append({"frame": index, "pts": pts})
            if pts >= next_time:
                thumbs.append((index, pts, image.resize((288, 162))))
                next_time += 2.0
    if not frames:
        raise ValueError("empty video")
    # These are visual difference proposals, never known ground-truth boundaries.
    candidates = []
    for suggestion in sorted(suggestions, key=lambda item: -item["difference"]):
        if all(abs(suggestion["frame"] - item["frame"]) >= 12 for item in candidates):
            candidates.append(suggestion)
    if clip_count is not None:
        candidates = candidates[:clip_count - 1]
        valid_count = len(candidates) == clip_count - 1
    else:
        valid_count = 1 <= len(candidates) <= 3
    cuts = [0, *sorted(item["frame"] for item in candidates), len(frames)] if valid_count else []
    boundaries = [[start, end] for start, end in zip(cuts, cuts[1:])]
    output.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (4 * 288, max(1, (len(thumbs) + 3) // 4) * 190), "white")
    draw = ImageDraw.Draw(sheet)
    for position, (index, pts, image) in enumerate(thumbs):
        x, y = (position % 4) * 288, (position // 4) * 190
        sheet.paste(image, (x, y))
        draw.text((x + 3, y + 164), f"frame {index} | {pts:.2f}s", fill="black")
    sheet.save(output / "contact.jpg", quality=86)
    report = {"status": "needs_review", "reviewer": None, "frame_count": len(frames),
              "first_pts": frames[0]["pts"], "last_pts": frames[-1]["pts"],
              "cut_suggestions": suggestions, "suggested_clip_ranges": boundaries,
              "boundary_method": "unapproved consecutive-frame mean RGB difference >30 at64x36",
              "source_provenance": "unknown"}
    (output / "probe.json").write_text(json.dumps(report, indent=2))
    (output / "frame_index.json").write_text(json.dumps(frames))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=10, help="Candidate total, including provisional validation")
    parser.add_argument("--val-count", type=int, default=2)
    parser.add_argument("--test-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-media-mb", type=int, default=256)
    parser.add_argument("--annotations-only", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.val_count <= args.train_count <= 20 or not 0 <= args.test_count <= 10:
        parser.error("bounded access check supports at most20 train and10 test candidates")
    root = args.data_root.resolve()
    rows, prepared, ledger = [], [], []
    for dataset in ("train", "test"):
        config = DATASETS[dataset]
        annotation = root / "raw" / config["annotation"]
        download = download_file(annotation_url(dataset), annotation, max_bytes=40 * 1024 * 1024)
        normalized = normalize_annotations(read_jsonl(annotation), dataset)
        write_jsonl(root / "normalized" / f"{dataset}.jsonl", normalized)
        print(json.dumps({"stage": "annotations", "dataset": dataset, "count": len(normalized), "download": download}), flush=True)
        if args.annotations_only:
            continue
        count = args.train_count if dataset == "train" else args.test_count
        selected = choose_candidates(normalized, count, args.seed, first_clip_only=True)
        for position, candidate in enumerate(selected):
            # This membership is for inspection/plumbing only. Unknown source
            # identities prevent claiming a verified independent final split.
            split = "test" if dataset == "test" else ("val" if position < args.val_count else "train")
            candidate["proposed_split"] = split
            name = candidate["id"].replace(":", "_")
            destination = root / "raw" / "media" / f"{dataset}_{Path(candidate['source_path']).name}"
            try:
                acquisition = download_file(media_url(candidate), destination, max_bytes=args.max_media_mb * 1024 * 1024)
                candidate["acquisition"] = acquisition
                report = review_media(destination, root / "audits" / name, candidate["clip_count"])
                candidate["review_probe"] = report
                if report["suggested_clip_ranges"]:
                    record = prepare_record(candidate, destination, report["suggested_clip_ranges"], root, split=split)
                    record["audit"].update({"boundary_status": "suggested", "evidence_status": "unverified",
                                            "split_status": "provisional_unknown_source_identity"})
                    prepared_review(record, root, root / "audits" / name)
                    prepared.append(record)
                candidate["status"] = "needs_review"
                print(json.dumps({"id": candidate["id"], "status": candidate["status"], "bytes": acquisition["bytes"],
                                  "question": candidate["question"], "options": candidate["options"], "answer": candidate["answer"],
                                  "suggested_boundaries": report["suggested_clip_ranges"]}), flush=True)
            except Exception as error:
                candidate["status"] = "failed"
                ledger.append({"id": candidate["id"], "stage": "acquisition_or_preparation", "error": f"{type(error).__name__}: {error}"})
                print(json.dumps(ledger[-1]), flush=True)
            rows.append(candidate)
            # Persist progress even if a later network transfer is interrupted.
            write_jsonl(root / "manifests" / "candidates.jsonl", rows)
            write_jsonl(root / "manifests" / "prepared_needs_review.jsonl", prepared)
            write_jsonl(root / "manifests" / "acquisition_failures.jsonl", ledger)
    print(json.dumps({"stage": "complete", "candidates": len(rows), "prepared_needs_review": len(prepared),
                      "failures": len(ledger), "human_approved": 0}), flush=True)


if __name__ == "__main__":
    main()
