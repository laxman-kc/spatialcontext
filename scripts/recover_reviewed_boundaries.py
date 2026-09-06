"""Recover sampler PNG records only from explicit source-bound visual cut reviews."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecqa.data import safe_path, sha256_file, write_jsonl
from scripts.acquire_data import prepared_review
from scripts.prepare_full_dataset import atomic_json, record_from_donor_bank


def recover(data_root: Path, candidate_folder: Path, decision_path: Path) -> Path:
    root = data_root.resolve()
    folder = candidate_folder.resolve()
    scan_path = folder / "source_scan.json"
    decision = json.loads(decision_path.read_text())
    if decision.get("source_scan_sha256") != sha256_file(scan_path):
        raise ValueError("Visual boundary decision is not bound to this exact source scan")
    meta = json.loads((folder / "review.json").read_text())
    if decision.get("id") != meta["id"]:
        raise ValueError("Boundary review belongs to another candidate")
    scan = json.loads(scan_path.read_text())
    record = record_from_donor_bank(meta["candidate"], scan, root, decision)
    record["provenance"]["recovered_source_scan_sha256"] = decision["source_scan_sha256"]
    output = safe_path(root, "fullstudy/recovered/" + meta["id"].replace(":", "_"))
    output.mkdir(parents=True, exist_ok=True)
    path = output / "prepared.jsonl"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != record:
            raise ValueError("Refusing to overwrite a different recovered record")
    write_jsonl(path, [record])
    atomic_json(output / "boundary_review.json", decision)
    prepared_review(record, root, output)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--candidate-folder", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    args = parser.parse_args()
    print(recover(args.data_root, args.candidate_folder, args.decision))


if __name__ == "__main__":
    main()
