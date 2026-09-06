"""Bind explicit visual-review decisions to prepared candidate contents.

This tool records decisions; it never produces or approves a visual review.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from ecqa.artifacts import digest, read_jsonl
from ecqa.data import validate_record, write_jsonl


def review_identity(record):
    body = {key: record[key] for key in ("id", "question", "options", "answer", "conditions", "provenance")}
    # These are facts the reviewer sees, not later administrative group/split
    # assignments or the approval fields produced by apply_reviews.
    audit = record["audit"]
    body["reviewed_temporal_and_donor_facts"] = {
        key: audit.get(key) for key in (
            "target_clip", "source_clip_ranges", "replacement_clip_indices", "replacement_frame_indices",
            "donor_source_clip_fingerprints", "donor_sha256",
        )
    }
    return digest(body)


def apply_reviews(records, decisions):
    indexed = {record["id"]: record for record in records}
    if len(indexed) != len(records):
        raise ValueError("Duplicate candidate IDs")
    reviewed, rejected, seen = [], [], set()
    for decision in decisions:
        identifier = decision["id"]
        if identifier in seen or identifier not in indexed:
            raise ValueError("Duplicate decision or unknown candidate")
        seen.add(identifier)
        record = indexed[identifier]
        if decision["review_identity"] != review_identity(record):
            raise ValueError(f"Reviewed candidate changed: {identifier}")
        if decision["decision"] not in {"accept", "reject", "needs_review"}:
            raise ValueError("Unknown review decision")
        if any(not isinstance(decision.get(key), str) or not decision[key].strip()
               for key in ("reason", "reviewer")):
            raise ValueError("A named reviewer and substantive reason are required")
        if decision["decision"] != "accept":
            rejected.append(decision)
            continue
        if not all(decision.get(key) is True for key in (
            "boundaries_verified", "evidence_verified", "answer_verified", "viewpoint_verified"
        )):
            raise ValueError("Accepted visual review has incomplete checks")
        if decision.get("reviewer_type") not in {"ai", "human"}:
            raise ValueError("Actual reviewer type must be explicit")
        if not isinstance(decision.get("viewpoint"), str) or not decision["viewpoint"].strip():
            raise ValueError("Accepted visual review requires an explicit viewpoint")
        result = copy.deepcopy(record)
        edited_reviews = {}
        for condition in ("neutral", "competing"):
            if condition not in result["conditions"]:
                continue
            reviews = decision.get("condition_review", {})
            review = reviews.get(condition, {}) if isinstance(reviews, dict) else {}
            if (not isinstance(review, dict) or review.get("answer_preserved") is not True
                    or review.get("role_verified") is not True
                    or not isinstance(review.get("reason"), str) or not review["reason"].strip()):
                raise ValueError(f"Accepted edited example requires a substantive {condition} condition_review")
            edited_reviews[condition] = copy.deepcopy(review)
        if edited_reviews:
            result["audit"]["condition_review"] = edited_reviews
        target = result["audit"]["target_clip"]
        start, end = result["conditions"]["original"]["clip_ranges"][target - 1]
        source_frames = result["provenance"]["frame_indices"][start:end]
        result["audit"].update({
            "status": "approved", "reviewer": decision["reviewer"],
            "reviewer_type": decision["reviewer_type"],
            "human_validated": decision["reviewer_type"] == "human",
            "boundaries_verified": True, "evidence_verified": True,
            "answer_verified": True, "viewpoint_verified": True,
            "viewpoint": decision["viewpoint"],
            "evidence_frame_range": [min(source_frames), max(source_frames) + 1],
            "reviewed_source_frames": source_frames,
            "evidence_scope": "reviewed deterministic sampled frames; no exhaustive continuous-time claim",
            "review_identity": decision["review_identity"],
            "review_reason": decision["reason"],
            "source_review_notes": decision.get("source_review_notes", "Source identities remain unresolved"),
        })
        validate_record(result, require_approved=True)
        reviewed.append(result)
    return reviewed, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = read_jsonl(args.candidates)
    if args.decisions is None:
        for record in records:
            print(json.dumps({"id": record["id"], "review_identity": review_identity(record),
                              "question": record["question"], "options": record["options"],
                              "answer": record["answer"]}, ensure_ascii=False))
        return
    if args.output is None:
        parser.error("--output is required when applying decisions")
    accepted, rejected = apply_reviews(records, read_jsonl(args.decisions))
    write_jsonl(args.output, accepted)
    print(json.dumps({"accepted_originals": len(accepted), "rejected_or_unresolved": len(rejected),
                      "unreviewed": len(records) - len(accepted) - len(rejected)}))


if __name__ == "__main__":
    main()
