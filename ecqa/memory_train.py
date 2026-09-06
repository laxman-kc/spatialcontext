"""One separately frozen memory-reader candidate; train/validation only.

Reuse the existing LoRA loop. Historical parent code is lineage, not the code
executing this candidate. No source-test cases are materialized or scored.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from .artifacts import atomic_json, code_identity, digest, read_jsonl, validate_config
from .evaluate import model_identity
from .memory import MemoryStore
from .memory_eval import _prepare_case, _reader_record, evaluate_cases, summarize_cases
from .memory_model import MemoryRuntime
from .memory_study import load_study
from .retrieval import make_memory_record, retrieve
from .train import train, verify_checkpoint

EXPECTED_COUNTS = {"train": 108, "val": 50}
SELECTION_RULE = "One fresh-base candidate; final checkpoint; strict validation accuracy improvement else base"


def _parent(study):
    # Explicitly allow historical source code; new executable code is frozen below.
    return load_study(study, check_code=False)


def prepare_candidate(study, manifest, output):
    study, output = Path(study).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(study):
        raise ValueError("Candidate output must be a new directory outside the historical study")
    parent, old_cases = _parent(study)
    source = read_jsonl(manifest)
    if digest(source) != parent["cohort_digest"]:
        raise ValueError("Source manifest differs from the historical parent cohort")
    records = [row for row in source if row["split"] in EXPECTED_COUNTS]
    if ({split: sum(row["split"] == split for row in records) for split in EXPECTED_COUNTS}
            != EXPECTED_COUNTS or len({row["id"] for row in records}) != len(records)):
        raise ValueError("Candidate requires the complete unique 108 train / 50 validation cohort")
    groups = {}
    for row in records:
        if (row["audit"].get("status") != "approved" or not row["audit"].get("reviewer")
                or any(row["audit"].get(key) is not True for key in
                       ("answer_verified", "evidence_verified", "boundaries_verified", "viewpoint_verified"))):
            raise ValueError("Candidate source labels/evidence must already be approved")
        if not row.get("source_group") or groups.setdefault(row["source_group"], row["split"]) != row["split"]:
            raise ValueError("Source groups must be present and separate training from validation")
    full = {(case["split"], case["id"]): case for case in old_cases
            if case["split"] in EXPECTED_COUNTS and case["condition"] == "full"}
    config = deepcopy(parent["config"])
    config["memory"] = {"use_spatial": False, "observation_timeline": "synthetic"}
    validate_config(config)
    if any(config["training"][key] != value for key, value in
           {"epochs": 1, "learning_rate": 1e-5, "seed": 42, "checkpoint_selection": "end_of_run"}.items()):
        raise ValueError("Candidate must retain the declared one-epoch, 1e-5, seed-42 settings")
    cases, training, bindings = [], [], []
    for row in records:
        original = full[(row["split"], row["id"])]
        root = Path(original["data_root"])
        if (original["gold"] != row["answer"] or original["record"]["question"] != row["question"]
                or original["record"]["options"] != row["options"]
                or any(original["record"]["conditions"]["original"][key]
                       != row["conditions"]["original"][key] for key in ("clip_ranges", "fps"))):
            raise ValueError("Parent full case differs from the approved source record")
        parent_input = _prepare_case(original, root)  # Only train/val retained images are opened.
        with MemoryStore.open(root) as store:
            observations = store.verify()["observations"]
            episodes = {item["episode_id"] for item in observations}
            if len(episodes) != 1:
                raise ValueError("Candidate requires one episode per sealed store")
            episode = episodes.pop()
            result = retrieve(store, episode_id=episode, question=row["question"], max_pairs=4)
        target = row["audit"]["target_clip"]  # Post-retrieval audit; never passed to retrieve.
        if type(target) is not int or not 1 <= target < len(row["conditions"]["original"]["clip_ranges"]):
            raise ValueError("Approved target must be a valid nonfinal source clip")
        start, end = row["conditions"]["original"]["clip_ranges"][target - 1]
        if (not result.observations or result.provenance["requested_clip"] != target
                or len(result.observations) != (end - start) // 2
                or any(item["clip_index"] != target for item in result.observations)):
            raise ValueError("Question-selected evidence does not cover all audited target pairs")
        reader = make_memory_record(row["question"], row["options"], result)
        relative = root.relative_to(study).as_posix()
        case = {"id": row["id"], "split": row["split"], "condition": "retrieval_all", "record": reader,
                "gold": row["answer"], "source_group": row["source_group"], "data_root": relative,
                "store_digest": result.provenance["store_digest"], "retrieval": result.provenance,
                "evidence": {"target_episode_id": episode, "target_clip_index": target,
                             "target_pair_count": (end - start) // 2}}
        identity = _prepare_case({**case, "data_root": str(root)}, root)["identity"]
        cases.append(case)
        bindings.append({"case": identity, "parent_input": parent_input["identity"]})
        if row["split"] == "train":
            training.append({**reader, "id": row["id"], "split": "train", "answer": row["answer"],
                             "audit": deepcopy(row["audit"])})
    body = {"schema": "memory-reader-candidate-v1", "parent_protocol_digest": parent["protocol_digest"],
            "source_cohort_digest": digest(source), "historical_parent_code_check": False,
            "config": config, "code_digest": code_identity(), "counts": EXPECTED_COUNTS,
            "cases": cases, "training_records": training, "input_bindings": bindings,
            "selection_rule": SELECTION_RULE, "candidate_count": 1,
            "interpretation": "Existing AI-reviewed development cohort; no fresh test or generalization claim"}
    body["protocol_digest"] = digest(body)
    atomic_json(output / "protocol.json", body)
    return {"protocol_digest": body["protocol_digest"], "counts": EXPECTED_COUNTS}


def load_candidate(study, output):
    study = Path(study).resolve()
    protocol = json.loads((Path(output) / "protocol.json").read_text())
    if digest({key: value for key, value in protocol.items() if key != "protocol_digest"}) != protocol["protocol_digest"]:
        raise ValueError("Candidate protocol integrity mismatch")
    if protocol["code_digest"] != code_identity():
        raise ValueError("Candidate executable code changed after freeze")
    parent, _ = _parent(study)
    if (parent["protocol_digest"] != protocol["parent_protocol_digest"]
            or parent["cohort_digest"] != protocol["source_cohort_digest"]):
        raise ValueError("Candidate historical lineage changed")
    if protocol["selection_rule"] != SELECTION_RULE or protocol["candidate_count"] != 1:
        raise ValueError("Candidate selection rule changed")
    validate_config(protocol["config"])
    if (protocol["config"]["training"] != parent["config"]["training"]
            or protocol["config"]["model"] != parent["config"]["model"]
            or protocol["config"]["memory"] != {"use_spatial": False, "observation_timeline": "synthetic"}
            or protocol["counts"] != EXPECTED_COUNTS):
        raise ValueError("Candidate declared configuration/counts changed")
    cases = deepcopy(protocol["cases"])
    for case, binding in zip(cases, protocol["input_bindings"], strict=True):
        relative = Path(case["data_root"])
        root = (study / relative).resolve()
        if relative.is_absolute() or not root.is_relative_to(study / "memory"):
            raise ValueError("Candidate store must be inside the historical sealed memory directory")
        case["data_root"] = str(root)
        if _prepare_case(case, root)["identity"] != binding["case"]:
            raise ValueError("Candidate selected input identity changed")
    return protocol, cases


class TrainingRuntime(MemoryRuntime):
    """Route approved envelopes to sealed roots; gold reaches only the loss API."""

    def __init__(self, config, data_root, records, roots):
        self._training_records = {row["id"]: deepcopy(row) for row in records}
        self._training_roots = {key: Path(value).resolve() for key, value in roots.items()}
        if len(self._training_records) != len(records) or set(self._training_records) != set(roots):
            raise ValueError("Training envelopes and sealed roots must match uniquely")
        super().__init__(config, data_root, adapter=None)

    def encode(self, record, condition="original", answer=None):
        if (self._training_records.get(record.get("id")) != record or condition != "original"
                or answer != record.get("answer")):
            raise ValueError("Training input/label differs from the frozen approved envelope")
        previous = self.data_root
        try:
            self.data_root = self._training_roots[record["id"]]
            return super().encode(_reader_record(record), "original", answer=answer)
        finally:
            self.data_root = previous


def run_training(study, output, resume=None):
    protocol, cases = load_candidate(study, output)
    records = protocol["training_records"]
    roots = {case["id"]: case["data_root"] for case in cases if case["split"] == "train"}
    runtime = TrainingRuntime(protocol["config"], study, records, roots)
    return train(runtime, records, protocol["config"], Path(output) / "train",
                 protocol_digest=protocol["protocol_digest"], resume=resume)


def _candidate_adapter(output, protocol):
    summary = json.loads((Path(output) / "train/summary.json").read_text())
    adapter = (Path(output) / "train" / summary["checkpoint"]).resolve()
    if not adapter.is_relative_to((Path(output) / "train").resolve()):
        raise ValueError("Candidate checkpoint escapes its training directory")
    metadata = verify_checkpoint(adapter)
    identity = digest({"protocol": protocol["protocol_digest"], "records": protocol["training_records"],
                       "settings": protocol["config"]["training"], "diagnostic": False})
    if (summary.get("stop_reason") != "epoch_complete" or summary.get("identity") != identity
            or metadata["identity"] != identity or metadata["diagnostic"]
            or metadata["config"] != protocol["config"]
            or metadata["protocol_digest"] != protocol["protocol_digest"]
            or metadata["next_index"] != len(protocol["training_records"])):
        raise ValueError("Candidate must be the complete verified final checkpoint from this run")
    return adapter


def run_scores(study, output, *, tuned=False):
    protocol, cases = load_candidate(study, output)
    adapter = _candidate_adapter(output, protocol) if tuned else None
    runtime = MemoryRuntime(protocol["config"], study, adapter=adapter)
    name = "tuned" if tuned else "base"
    rows = evaluate_cases(runtime, cases, Path(output) / "predictions", model_name=name,
                          model_digest=model_identity(protocol["config"], adapter),
                          protocol_digest=protocol["protocol_digest"])
    atomic_json(Path(output) / f"{name}-predictions.json", rows)
    return {"model": name, "cases": len(rows)}


def report_candidate(study, output):
    protocol, cases = load_candidate(study, output)
    adapter = _candidate_adapter(output, protocol)
    rows = [row for name in ("base", "tuned")
            for row in json.loads((Path(output) / f"{name}-predictions.json").read_text())]
    summary = summarize_cases(cases, rows, expected_conditions={split: ["retrieval_all"] for split in EXPECTED_COUNTS},
                              bootstrap_samples=protocol["config"]["analysis"]["bootstrap_samples"], seed=42)
    expected = {"base": model_identity(protocol["config"]), "tuned": model_identity(protocol["config"], adapter)}
    if (summary["identities"]["protocol_digest"] != protocol["protocol_digest"]
            or summary["identities"]["code_digest"] != protocol["code_digest"]
            or summary["identities"]["config_digest"] != digest(protocol["config"])
            or any(summary["identities"]["model:" + name] != value for name, value in expected.items())):
        raise ValueError("Report predictions belong to another candidate configuration/model")
    models = summary["splits"]["val"]["conditions"]["retrieval_all"]["models"]
    winner = "tuned" if models["tuned"]["correct"] > models["base"]["correct"] else "base"
    selection = {"protocol_digest": protocol["protocol_digest"], "rule": SELECTION_RULE,
                 "selected": winner, "model_digest": expected[winner], "validation": models,
                 "checkpoint": str(adapter.relative_to(Path(output).resolve())) if winner == "tuned" else None}
    summary["limitations"].append("One candidate selected on reused validation; no test cases were scored here.")
    atomic_json(Path(output) / "report.json", summary)
    atomic_json(Path(output) / "selected-model.json", selection)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "base", "train", "tuned", "report"))
    parser.add_argument("--study", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--resume")
    args = parser.parse_args()
    if args.phase == "prepare":
        if not args.manifest:
            parser.error("prepare requires --manifest")
        result = prepare_candidate(args.study, args.manifest, args.output)
    elif args.phase == "train":
        result = run_training(args.study, args.output, args.resume)
    elif args.phase == "report":
        result = report_candidate(args.study, args.output)
    else:
        result = run_scores(args.study, args.output, tuned=args.phase == "tuned")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
