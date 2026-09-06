"""Collect explicit reviews and conservative source links before a study freeze.

This performs no visual review and never reads model predictions. Site groups
are review-backed conservative exclusions, not recovered camera-file metadata.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path

from ecqa.artifacts import atomic_json, digest, file_digest, load_config, read_jsonl
from ecqa.data import write_jsonl
from scripts.review_full_study import apply_reviews, review_identity

SITE_ALIASES = {
    # Reviewers explicitly compared the same labeled towers, road layout and
    # projecting yellow/white frontage; these are spelling conventions only.
    "yangguang-chengtanfu-residential": "yangguangcheng-tanfu-yellow-white-frontage",
    "yangguang-labeled-tower-junction": "yangguangcheng-tanfu-yellow-white-frontage",
    "blue-dome-red-roof-tower-district": "red-spire-towers-blue-dome-construction",
}


def site_node(tag):
    return "site:" + SITE_ALIASES.get(tag, tag)


def secondary_vetoes(prepared, decisions):
    """A label-only second review can exclude, never approve an original."""
    rejected = set()
    for decision in decisions:
        identifier = decision["id"]
        if identifier not in prepared or decision["review_identity"] != review_identity(prepared[identifier]):
            raise ValueError("Secondary review refers to a changed or unknown original")
        if decision.get("decision") not in {"accept", "reject"}:
            raise ValueError("Secondary decision must explicitly accept or reject")
        if (decision.get("reviewer_type") not in {"ai", "human"}
                or not decision.get("reviewer") or not decision.get("reason") or not decision.get("scope")):
            raise ValueError("Secondary review requires named reviewer, scope and reason")
        if decision["decision"] == "reject":
            rejected.add(identifier)
    return rejected


class Groups:
    def __init__(self):
        self.parent = {}

    def root(self, item):
        self.parent.setdefault(item, item)
        while item != self.parent[item]:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def join(self, first, second):
        a, b = self.root(first), self.root(second)
        self.parent[max(a, b)] = min(a, b)


def collect(data_root):
    root = Path(data_root)
    study = root / "fullstudy"
    scans, prior = {}, set()
    for pattern in ("preparation/review/*/source_scan.json",
                    "preparation/supplemental_exposure/*/source_scan.json"):
        for path in sorted(study.glob(pattern)):
            scan = json.loads(path.read_text())
            # Unsupported cuts keep tentative chunk identities. These still
            # support conservative overlap exclusions; they do not approve a
            # boundary or qualify a clip for use as a donor.
            scan["source_clip_fingerprints"] = (scan["source_clip_fingerprints"]
                                                or scan.get("tentative_129_chunk_fingerprints", []))
            scans[scan["source_id"]] = scan
            if "supplemental_exposure" in path.parts:
                prior.add(scan["source_id"])
    prepared = {}
    for pattern in ("preparation/review/*/prepared.jsonl", "recovered/**/*.jsonl"):
        for path in sorted(study.glob(pattern)):
            for record in read_jsonl(path):
                if "conditions" not in record or "provenance" not in record:
                    continue
                previous = prepared.setdefault(record["id"], record)
                if previous != record:
                    raise ValueError("Conflicting prepared identities: " + record["id"])
    decisions = []
    for reviewer in ("root", "model", "engineering", "data"):
        path = study / "reviews" / f"{reviewer}-decisions.jsonl"
        if path.exists():
            decisions.extend(read_jsonl(path))
    approved, rejected = apply_reviews(list(prepared.values()), decisions)
    secondary = []
    for path in sorted((study / "reviews").glob("*-secondary-decisions.jsonl")):
        secondary.extend(read_jsonl(path))
    vetoed = secondary_vetoes(prepared, secondary)
    approved = [row for row in approved if row["id"] not in vetoed]
    groups, evidence = Groups(), [{"review_confirmed_site_aliases": SITE_ALIASES}]
    for scan in scans.values():
        for value in scan["source_clip_fingerprints"]:
            groups.root("clip:" + value)

    def member_node(member):
        identifier, index = member["id"], member["clip_index"] - 1
        scan = scans[identifier]
        value = scan["source_clip_fingerprints"][index]
        if member.get("source_clip_fingerprint", value) != value:
            raise ValueError("Stale reviewed source identity: " + identifier)
        return "clip:" + value

    for path in sorted((study / "reviews").glob("*-site-links.jsonl")):
        for link in read_jsonl(path):
            if not link.get("reason") or link.get("reviewer_type") not in {"ai", "human"}:
                raise ValueError("Source link requires actual review provenance")
            tag = site_node(link["site_tag"])
            for member in link["members"]:
                groups.join(tag, member_node(member))
            evidence.append(link)
    for path in sorted((study / "reviews").glob("*-scenes.jsonl")):
        for scene in read_jsonl(path):
            if scene.get("conservative_site_group"):
                if not scene.get("site_group_basis"):
                    raise ValueError("Scene group lacks review basis")
                groups.join(site_node(scene["conservative_site_group"]), member_node(scene))
                evidence.append(scene)
    # Treat strong unresolved prior matches as exclusions, never as proof of identity.
    quarantine = set()
    path = study / "preparation/supplemental_test_eligibility.jsonl"
    if path.exists():
        for row in read_jsonl(path):
            if not row["eligible_after_supplemental_exposure_screen"]:
                quarantine.add(row["id"])
                for link in row["possible_source_links"]:
                    first = {"id": row["id"], "clip_index": int(link["new_chunk"].split("_")[-1])}
                    second = {"id": link["prior_id"], "clip_index": int(link["prior_chunk"].split("_")[-1])}
                    groups.join(member_node(first), member_node(second))
    exposure = {groups.root("clip:" + value) for identifier in prior
                for value in scans[identifier]["source_clip_fingerprints"]}
    clip_groups = {value: groups.root("clip:" + value) for scan in scans.values()
                   for value in scan["source_clip_fingerprints"]}
    # Human-readable named site tags remain distinct from opaque union roots.
    group_sites = defaultdict(set)
    for node in groups.parent:
        if node.startswith("site:"):
            group_sites[groups.root(node)].add(node[5:])
    return {"approved": approved, "prepared": prepared, "decisions": decisions,
            "secondary_decisions": secondary, "secondary_veto_ids": vetoed,
            "rejected": rejected, "scans": scans, "prior_ids": prior,
            "clip_groups": clip_groups, "exposed_roots": exposure,
            "quarantined_ids": quarantine, "group_sites": group_sites,
            "source_evidence_digest": digest(evidence)}


def record_roots(record, state):
    values = [*record["provenance"]["source_clip_fingerprints"],
              *record["audit"].get("donor_source_clip_fingerprints", [])]
    return {state["clip_groups"][value] for value in values}


def held_training_audit(data_root):
    """Account for every held candidate without treating previews as reviews."""
    study = Path(data_root) / "fullstudy"
    queue = read_jsonl(study / "held-training/queue.jsonl")
    decisions = {}
    for path in sorted((study / "reviews").glob("*-held-training-audit.jsonl")):
        for row in read_jsonl(path):
            if row["id"] in decisions:
                raise ValueError("Duplicate held training audit: " + row["id"])
            decisions[row["id"]] = row
    result, pending = [], []
    for candidate in queue:
        row = decisions.get(candidate["id"])
        if "target_preview" not in candidate:
            result.append({"id": candidate["id"], "decision": "metadata_exclusion",
                           "reason": candidate["triage"], "visually_reviewed": False,
                           "source_scan_sha256": candidate["source_scan_sha256"]})
            continue
        if row is None:
            pending.append(candidate["id"])
            continue
        if (row.get("source_scan_sha256") != candidate["source_scan_sha256"]
                or not row.get("reviewer") or not row.get("reason")
                or row.get("reviewer_type") not in {"ai", "human"}
                or row.get("label_result", row.get("decision")) not in {"accept", "reject"}):
            raise ValueError("Incomplete or stale held training audit")
        scan_path = Path(candidate["candidate_folder"]) / "source_scan.json"
        if file_digest(scan_path) != candidate["source_scan_sha256"]:
            raise ValueError("Held source scan changed after review")
        result.append({**row, "decision": row.get("label_result", row.get("decision")), "visually_reviewed": True})
    unknown = set(decisions) - {row["id"] for row in queue}
    if unknown:
        raise ValueError("Held training audit refers to unknown candidates")
    return result, pending


def inventory(data_root, output):
    state = collect(data_root)
    rows = []
    decisions = {row["id"]: row for row in state["decisions"]}
    held_decisions = {}
    for path in sorted((Path(data_root) / "fullstudy/reviews").glob("*-held-*-audit.jsonl")):
        for row in read_jsonl(path):
            if row["id"] not in state["prepared"]:
                held_decisions[row["id"]] = row
    for identifier, scan in sorted(state["scans"].items()):
        if identifier in state["prior_ids"]:
            continue
        roots = {state["clip_groups"][value] for value in scan["source_clip_fingerprints"]}
        exposed = sorted(roots & state["exposed_roots"])
        decision = decisions.get(identifier)
        held = held_decisions.get(identifier)
        rows.append({"id": identifier, "dataset": identifier.split(":")[0],
                     "prepared": identifier in state["prepared"],
                     "review_decision": decision["decision"] if decision else held.get("label_result", held.get("decision")) if held else "not_reviewed",
                     "review_reason": decision.get("reason") if decision else held.get("reason") if held else None,
                     "secondary_review_veto": identifier in state["secondary_veto_ids"],
                     "source_roots": sorted(roots),
                     "site_groups": sorted({name for node in roots for name in state["group_sites"][node]}),
                     "prior_exposure_roots": exposed,
                     "test_exposure_eligible": not exposed and identifier not in state["quarantined_ids"]})
    eligible = []
    for record in state["approved"]:
        if record["split"] != "test" or (
            not record_roots(record, state) & state["exposed_roots"]
            and record["id"] not in state["quarantined_ids"]
        ):
            eligible.append(record)
    output = Path(output)
    write_jsonl(output / "candidate-audit.jsonl", rows)
    write_jsonl(output / "reviewed-originals.jsonl", eligible)
    summary = {"status": "provisional; reviews and donor controls may still be incomplete",
               "scanned_candidates": dict(Counter(row["dataset"] for row in rows)),
               "prepared_originals": len(state["prepared"]),
               "review_decisions": dict(Counter(row["decision"] for row in state["decisions"])),
               "secondary_review_vetoes": len(state["secondary_veto_ids"]),
               "eligible_reviewed_originals": dict(Counter(row["split"] for row in eligible)),
               "test_candidates_exposure_excluded": sum(row["dataset"] == "test" and not row["test_exposure_eligible"] for row in rows),
               "prior_videos": len(state["prior_ids"]),
               "source_evidence_digest": state["source_evidence_digest"],
               "limits": "AI reviews; conservative known-site grouping; incomplete original provenance. No matches do not establish independence."}
    atomic_json(output / "summary.json", summary)
    return summary


def assemble(state, controlled_tests, *, train_cap=300, val_cap=50, test_cap=50, seed=42):
    """Seeded source-component split; cap truncation never crosses partitions."""
    tests = copy.deepcopy(controlled_tests[:test_cap])
    test_roots = set()
    for record in tests:
        roots = record_roots(record, state)
        if roots & state["exposed_roots"] or record["id"] in state["quarantined_ids"]:
            raise ValueError("Exposed test target/donor: " + record["id"])
        test_roots.update(roots)
    train_pool = [copy.deepcopy(row) for row in state["approved"]
                  if row["split"] != "test" and not record_roots(row, state) & test_roots]
    groups = Groups()
    for record in train_pool:
        for node in record_roots(record, state):
            groups.join("record:" + record["id"], node)
    components = defaultdict(list)
    for record in train_pool:
        components[groups.root("record:" + record["id"])].append(record)
    ordered = sorted(components.values(), key=lambda values: digest([seed, sorted(row["id"] for row in values)]))
    train, val, unused = [], [], []
    # Reserve the largest connected component for training before filling the
    # smaller validation cap. Otherwise a common stock clip could put hundreds
    # of eligible questions into a 50-row validation component and discard the
    # rest. This choice uses source structure only, before any model scores.
    reserved = max(ordered, key=len) if ordered and train_cap else None
    if reserved is not None:
        ordered.remove(reserved)
        ordered.insert(0, reserved)
    for component in ordered:
        component.sort(key=lambda row: digest([seed, row["id"]]))
        destination, split, cap = ((val, "val", val_cap)
                                   if component is not reserved and len(val) < val_cap
                                   else (train, "train", train_cap))
        space = max(0, cap - len(destination))
        for record in component[:space]:
            record["split"] = split
            destination.append(record)
        unused.extend(row["id"] for row in component[space:])
    result = train + val + tests
    final_groups = Groups()
    for record in result:
        for node in record_roots(record, state):
            final_groups.join("record:" + record["id"], node)
    for record in result:
        record["source_group"] = "conservative:" + digest(final_groups.root("record:" + record["id"]))[:20]
        # Group donor reuse at the final test component level for the paired CI.
        if record["split"] == "test":
            record["audit"]["donor_source_groups"] = [record["source_group"]]
    return result, {"counts": {"train": len(train), "val": len(val), "test": len(tests)},
                    "train_components_before_caps": len(components), "unused_component_cap_ids": unused,
                    "source_split_rule": "Reserve largest source-connected component for training; seed42 hashes order other components and rows; fill validation then remaining training cap; unused members never enter the opposite split."}


def finalize(data_root, output, priority_path, controls_path, control_decisions_path, config_path):
    """Finalize only a reviewed cohort; never silently stop at an incomplete cap."""
    import yaml

    state = collect(data_root)
    held_audit, pending_held = held_training_audit(data_root)
    if pending_held:
        raise ValueError(f"Unfinished held training reviews: {len(pending_held)}")
    controlled, control_rejections = apply_reviews(read_jsonl(controls_path), read_jsonl(control_decisions_path))
    originals = {row["id"]: row for row in state["approved"] if row["split"] == "test"}
    eligible, exposed_controls, vetoed_controls = [], [], []
    for row in controlled:
        if row["id"] in state["secondary_veto_ids"]:
            vetoed_controls.append(row["id"])
            continue
        old = originals[row["id"]]
        if any(row[key] != old[key] for key in ("question", "options", "answer", "provenance")):
            raise ValueError("Controlled record changed its reviewed original facts")
        if row["conditions"]["original"] != old["conditions"]["original"]:
            raise ValueError("Controlled record changed the original frames")
        if record_roots(row, state) & state["exposed_roots"] or row["id"] in state["quarantined_ids"]:
            exposed_controls.append(row["id"])
        else:
            eligible.append(row)
    priority = {row["id"]: i for i, row in enumerate(read_jsonl(priority_path))}
    if any(row["id"] not in priority for row in eligible):
        raise ValueError("Test example is absent from the prespecified candidate priority")
    eligible.sort(key=lambda row: priority[row["id"]])
    records, summary = assemble(state, eligible)
    selected_tests = [row for row in records if row["split"] == "test"]
    selected_ids = {row["id"] for row in records}
    reviewed_ids = {row["id"] for row in state["decisions"]}
    test_roots = set().union(*(record_roots(row, state) for row in selected_tests)) if selected_tests else set()
    pending_test, pending_train, excluded_source_train = [], [], []
    last_priority = max((priority[row["id"]] for row in selected_tests), default=-1)
    controlled_ids = {row["id"] for row in controlled} | {
        row["id"] for row in control_rejections if row["decision"] == "reject"
    }
    for identifier, row in state["prepared"].items():
        roots = record_roots(row, state)
        if row["split"] == "test":
            if roots & state["exposed_roots"] or identifier in state["quarantined_ids"]:
                continue
            precedes_cap = len(selected_tests) < 50 or priority[identifier] <= last_priority
            if precedes_cap and (identifier not in reviewed_ids or
                                 identifier in originals and identifier not in controlled_ids):
                pending_test.append(identifier)
        elif roots & test_roots:
            excluded_source_train.append(identifier)
        elif identifier not in reviewed_ids:
            pending_train.append(identifier)
    if pending_test:
        raise ValueError(f"Unfinished test original/control reviews: {pending_test}")
    if (summary["counts"]["train"] < 300 or summary["counts"]["val"] < 50) and pending_train:
        raise ValueError(f"Cohort below caps with {len(pending_train)} eligible prepared training originals still unreviewed")
    if not all(summary["counts"].values()):
        raise ValueError("Training, validation and test each require usable examples")
    secondary_ids = {row["id"] for row in state["secondary_decisions"]}
    pending_secondary = [row["id"] for row in records if row["id"] not in secondary_ids]
    if pending_secondary:
        raise ValueError(f"Selected labels lack second review: {pending_secondary}")
    output = Path(output)
    config = load_config("configs/pilot.yaml")
    excluded_fingerprints = sorted(value for value, node in state["clip_groups"].items()
                                   if node in state["exposed_roots"])
    config["exposure_exclusions"] = {
        "ids": sorted(state["prior_ids"]), "source_groups": [],
        "source_clip_fingerprints": excluded_fingerprints,
        "media_sha256": sorted({state["scans"][identifier]["source_media_sha256"] for identifier in state["prior_ids"]}),
        "rule": "All16 previous feasibility/preflight videos; conservative reviewed matching clip/site roots expanded to all known fingerprints. No absent-match independence claim.",
    }
    config["cohort"] = {"caps": {"train": 300, "val": 50, "test": 50}, "actual": summary["counts"],
                        "priority_sha256": file_digest(priority_path),
                        "source_evidence_digest": state["source_evidence_digest"],
                        "control_decisions_sha256": file_digest(control_decisions_path),
                        "secondary_review_digest": digest(state["secondary_decisions"]),
                        "held_training_review_digest": digest(held_audit),
                        "source_split_rule": summary["source_split_rule"],
                        "reviewer_type": "ai", "human_validated": False}
    # Both files must be completed before the GPU freeze; they are not a freeze.
    write_jsonl(output / "approved.jsonl", records)
    write_jsonl(output / "held-training-audit.jsonl", held_audit)
    config_target = Path(config_path)
    config_target.parent.mkdir(parents=True, exist_ok=True)
    config_target.write_text(yaml.safe_dump(config, sort_keys=False))
    status_counts = defaultdict(Counter)
    for path in (Path(data_root) / "fullstudy/preparation/review").glob("*/review.json"):
        meta = json.loads(path.read_text())
        status_counts[meta["id"].split(":")[0]][meta.get("automatic_exclusion_or_hold") or "prepared"] += 1
    summary.update({
        "status": "curation complete; GPU protocol freeze still required",
        "preparation_status_before_manual_recovery": {key: dict(value) for key, value in status_counts.items()},
        "review_decisions": dict(Counter(row["decision"] for row in state["decisions"])),
        "secondary_review_decisions": dict(Counter(row["decision"] for row in state["secondary_decisions"])),
        "secondary_veto_ids": sorted(state["secondary_veto_ids"]),
        "held_training_audit": dict(Counter(row["decision"] for row in held_audit)),
        "held_training_metadata_exclusions": dict(Counter(row["reason"] for row in held_audit
                                                          if row["decision"] == "metadata_exclusion")),
        "reviewed_approved_originals": dict(Counter(row["split"] for row in state["approved"])),
        "exposed_control_ids": exposed_controls, "secondary_vetoed_control_ids": vetoed_controls,
        "rejected_control_ids": [row["id"] for row in control_rejections],
        "train_ids_excluded_by_selected_test_or_donor_source": sorted(excluded_source_train),
        "remaining_unreviewed_train_ids_after_caps": sorted(pending_train),
        "approved_but_not_selected_ids": sorted(row["id"] for row in state["approved"] if row["id"] not in selected_ids),
        "source_evidence_digest": state["source_evidence_digest"],
        "selected_label_counts": {split: dict(Counter(row["answer"] for row in records if row["split"] == split))
                                  for split in ("train", "val", "test")},
        "selected_test_source_components": len({row["source_group"] for row in selected_tests}),
        "limits": ["AI visual reviews only; no human validation.",
                   "Original constituent/camera identifiers are unavailable; known site and clip matches are grouped, residual overlap remains possible.",
                   "Class-level competing landmarks do not guarantee misleading answers; neutral/competing lighting, motion and scene complexity are imperfectly matched.",
                   "One training seed; test confidence intervals describe the reviewed conservative components and do not measure training-seed variability."],
    })
    atomic_json(output / "study-audit.json", summary)
    inventory(data_root, output / "curation")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="data/fullstudy/curation")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--priority", default="data/cohort_plan/test_priority.jsonl")
    parser.add_argument("--controls", default="data/fullstudy/controlled/candidates.jsonl")
    parser.add_argument("--control-decisions", default="data/fullstudy/reviews/control-decisions.jsonl")
    parser.add_argument("--config-output", default="configs/fullstudy.yaml")
    args = parser.parse_args()
    if args.finalize:
        result = finalize(args.data_root, args.output, args.priority, args.controls,
                          args.control_decisions, args.config_output)
    else:
        result = inventory(args.data_root, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
