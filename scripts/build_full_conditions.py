"""Materialize explicit, reviewed donor plans; never select or approve donors.

Run as ``python -m scripts.build_full_conditions --plan PATH``. The resulting
candidates require a separate content-bound review before cohort assembly.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from ecqa.artifacts import atomic_json, digest, read_jsonl
from ecqa.data import create_conditions, safe_path, sha256_file, write_jsonl
from scripts.assemble_full_study import collect, record_roots
from scripts.review_full_study import review_identity


def donor_frames(data_root, state, donor, count):
    """Resolve precisely the same interior sampling used for original clips."""
    identifier, clip_index = donor["id"], donor["clip_index"]
    if identifier in state["prior_ids"]:
        raise ValueError("Previously used video cannot supply a new test donor")
    scan = state["scans"][identifier]
    fingerprint = scan["source_clip_fingerprints"][clip_index - 1]
    if fingerprint != donor["source_clip_fingerprint"]:
        raise ValueError("Donor source identity changed")
    if state["clip_groups"][fingerprint] in state["exposed_roots"]:
        raise ValueError("Donor has known prior exposure")
    if count not in (2, 4) or donor.get("boundaries_verified") is not True:
        raise ValueError("Donor requires an explicit boundary review and 2/4 frames")
    indices = [(clip_index - 1) * 129 + j * 129 // (count + 1)
               for j in range(1, count + 1)]
    bank = {row["frame"]: row for row in scan["donor_bank"][clip_index - 1]}
    paths = [safe_path(Path(data_root).resolve(), bank[index]["path"]) for index in indices]
    if not all(path.is_file() and sha256_file(path) == bank[index]["sha256"]
               for path, index in zip(paths, indices)):
        raise ValueError("Missing or changed exact donor interior frames")
    return paths, {"id": identifier, "clip_index": clip_index,
                   "source_clip_fingerprint": fingerprint,
                   "source_media_sha256": scan["source_media_sha256"],
                   "source_frame_indices": indices,
                   "bank_frame_sha256": [sha256_file(path) for path in paths]}


def build(data_root, plan, output):
    state = collect(data_root)
    originals = {row["id"]: row for row in state["approved"]}
    records, receipts, seen = [], [], set()
    for assignment in plan:
        identifier = assignment["id"]
        if identifier in seen:
            raise ValueError("Duplicate control plan")
        seen.add(identifier)
        original = originals[identifier]
        roots = record_roots(original, state)
        if original["split"] != "test" or roots & state["exposed_roots"]:
            raise ValueError("Target is not an eligible reviewed test original")
        if identifier in state["quarantined_ids"]:
            raise ValueError("Target has unresolved prior overlap")
        if not assignment.get("reviewer") or assignment.get("reviewer_type") not in {"ai", "human"}:
            raise ValueError("Donor plan requires an actual named reviewer")
        slot = assignment["replacement_clip_index"]  # zero-based
        if not original["audit"]["target_clip"] <= slot < len(original["conditions"]["original"]["clip_ranges"]):
            raise ValueError("Only a whole clip strictly after the target may change")
        start, end = original["conditions"]["original"]["clip_ranges"][slot]
        frames, details, donor_roots = {}, {}, set()
        for kind in ("neutral", "competing"):
            if not assignment[kind].get("reason"):
                raise ValueError("Donor role needs a substantive visual review reason")
            paths, receipt = donor_frames(data_root, state, assignment[kind], end - start)
            node = state["clip_groups"][receipt["source_clip_fingerprint"]]
            if node in roots:
                raise ValueError("Donor shares known target-source material")
            donor_roots.add(node)
            frames[kind], details[kind] = {slot: paths}, receipt
        if len(donor_roots) != 2:
            raise ValueError("Neutral and competing must use distinct known donor sources")
        result = create_conditions(original, frames["neutral"], frames["competing"], data_root,
                                   donor_source_groups=sorted(donor_roots))
        result["audit"]["donor_source_clip_fingerprints"] = [details[kind]["source_clip_fingerprint"]
                                                              for kind in ("neutral", "competing")]
        result["audit"]["control_donor_details"] = details
        result["audit"]["control_plan_digest"] = digest(assignment)
        result["audit"].pop("condition_review", None)
        records.append(result)
        receipts.append({"id": identifier, "review_identity": review_identity(result),
                         "plan": copy.deepcopy(assignment), "donor_details": details})
    output = Path(output)
    write_jsonl(output / "candidates.jsonl", records)
    write_jsonl(output / "control-receipts.jsonl", receipts)
    atomic_json(output / "build.json", {"records": len(records), "plan_digest": digest(plan),
                                       "source_evidence_digest": state["source_evidence_digest"],
                                       "status": "awaiting fresh condition review"})
    render_panels(data_root, records, output)
    return records


def render_panels(data_root, records, output):
    from PIL import Image, ImageDraw, ImageOps
    for record in records:
        panel = Image.new("RGB", (1100, 530), "white")
        draw = ImageDraw.Draw(panel)
        draw.text((3, 2), f"{record['id']} target={record['audit']['target_clip']} gold={record['answer']}", fill="black")
        for row, kind in enumerate(("original", "neutral", "competing")):
            draw.text((3, 27 + row * 166), kind, fill="black")
            for column, frame in enumerate(record["conditions"][kind]["frames"]):
                image = Image.open(Path(data_root) / frame)
                image = ImageOps.contain(image, (135, 135))
                panel.paste(image, (column * 137, 46 + row * 166))
                draw.text((column * 137 + 2, 46 + row * 166), str(column + 1), fill="red")
        panel.save(output / f"{record['id'].replace(':', '_')}.jpg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", default="data/fullstudy/controlled")
    args = parser.parse_args()
    records = build(args.data_root, read_jsonl(args.plan), args.output)
    print(json.dumps({"controlled_candidates_requiring_review": len(records)}))


if __name__ == "__main__":
    main()
