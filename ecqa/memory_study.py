"""Prepare, score and report a separate, source-bound episodic-memory diagnostic.

Observation projection deliberately excludes question, options, labels and target
annotations. Stores are sealed before any question-dependent retrieval. The
existing reviewed cohort is development/regression evidence, never a fresh test.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

from .artifacts import atomic_json, code_identity, digest, file_digest, load_config, read_jsonl


def load_study_config(path):
    import yaml

    path = Path(path)
    settings = yaml.safe_load(path.read_text())
    if settings.get("schema_version") != 1:
        raise ValueError("Unknown memory study schema")
    if settings.get("interpretation") != "diagnostic_existing_cohort_not_fresh_test":
        raise ValueError("This runner only supports the explicitly diagnostic existing cohort")
    for field in ("reader_pairs", "capacity_pairs"):
        if type(settings.get(field)) is not int or not 1 <= settings[field] <= 4:
            raise ValueError(f"{field} must be one through four pairs")
    if type(settings.get("seed")) is not int or settings["seed"] < 0:
        raise ValueError("A nonnegative integer seed is required")
    supported = {"full", "memory_full", "retrieval", "oracle", "oracle_all",
                 "recent", "uniform", "wrong", "empty"}
    conditions = settings.get("conditions", [])
    if not conditions or len(set(conditions)) != len(conditions) or set(conditions) - supported:
        raise ValueError("Unknown, empty or duplicate diagnostic conditions")
    if "full" not in conditions or "retrieval" not in conditions or "oracle" not in conditions:
        raise ValueError("Full, retrieval and oracle comparisons are required")
    names = set(conditions)
    for item in settings.get("capacity_conditions", []):
        if (set(item) != {"name", "policy", "capacity_pairs"} or item["name"] in names
                or item["policy"] not in {"episode", "recent", "reservoir"}
                or type(item["capacity_pairs"]) is not int or not 1 <= item["capacity_pairs"] <= 4):
            raise ValueError("Invalid capacity comparison")
        names.add(item["name"])
    base = load_config(path.parent / settings["base_config"])
    return settings, base


def observation_projection(condition: dict, data_root: str | Path, *, episode_id: str) -> dict:
    """Accept visual condition only; bind a question-independent episode to pixels."""
    from .model import clip_map

    clip_map(condition)
    frames, ranges, fps = condition["frames"], condition["clip_ranges"], float(condition["fps"])
    if len(frames) != 8:
        raise ValueError("The legacy diagnostic projects exactly eight prepared frames")
    root = Path(data_root).resolve()
    files = []
    for name in frames:
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Source frame escapes data root")
        files.append({"path": str(path), "sha256": file_digest(path)})
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("An externally assigned episode identity is required")
    identity = {"episode_id": episode_id, "frame_sha256": [item["sha256"] for item in files],
                "clip_ranges": ranges, "fps": fps, "timeline": "synthetic_prepared_frame_order"}
    episode = episode_id
    observations = []
    for index in range(0, len(files), 2):
        clip = next(i for i, (start, end) in enumerate(ranges, 1) if start <= index < end)
        observations.append({"episode_id": episode, "scene_id": f"{episode}:clip:{clip}",
                             "clip_index": clip, "pair_index": index // 2,
                             "timestamp": (index + 0.5) / fps,
                             "frames": [item["path"] for item in files[index:index + 2]],
                             "source_id": f"{episode}:clip:{clip}"})
    return {"episode_id": episode, "identity": identity, "observations": observations}


def _write_store(projection, path, capacity, policy, seed):
    from .memory import MemoryStore

    started = time.monotonic()
    if path.exists():
        raise FileExistsError("Preparation requires new stores; choose a new output after interrupted preparation")
    with MemoryStore.create(path, capacity_pairs=capacity, policy=policy, seed=seed) as store:
        for observation in projection["observations"]:
            store.observe(**observation)
        receipt = store.seal()
        stats = store.stats()
    return receipt, stats, time.monotonic() - started


def prepare_study(settings, config, records, data_root, output):
    """Build immutable stores first, then safe reader cases and a frozen protocol."""
    from .memory import MemoryStore
    from .retrieval import make_memory_record, retrieve, retrieve_oracle

    output = Path(output).resolve()
    if (output / "protocol.json").exists():
        raise ValueError("Prepared study already exists; use it or choose a new output directory")
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("Cohort must be nonempty with unique IDs")
    output.mkdir(parents=True, exist_ok=True)
    prepared = []
    receipts = {}
    projected = {}
    # Phase 1 never passes questions/options/labels/target clip into the writer.
    for record in records:
        variants = ("original", "competing") if record["split"] == "test" else ("original",)
        for variant in variants:
            projection = observation_projection(record["conditions"][variant], data_root,
                                                episode_id=digest(record["id"]))
            projected[(record["id"], variant)] = projection
            specifications = [{"name": "all", "capacity_pairs": settings["capacity_pairs"], "policy": "episode"}]
            if record["split"] == "val":
                specifications.extend(settings.get("capacity_conditions", []))
            for spec in specifications:
                key = digest({"observation_identity": projection["identity"], "policy": spec["policy"],
                              "capacity_pairs": spec["capacity_pairs"], "seed": settings["seed"]})
                relative = f"memory/{key}"
                if relative not in receipts:
                    manifest, stats, seconds = _write_store(
                        projection, output / relative, spec["capacity_pairs"], spec["policy"], settings["seed"])
                    receipts[relative] = {"store_digest": manifest["store_digest"], "stats": stats,
                                          "write_seconds": seconds, "specification": spec,
                                          "observation_identity": projection["identity"]}
                prepared.append((record, variant, projection, spec, relative))
    atomic_json(output / "observation-commit.json", {"stores": receipts,
                "stores_digest": digest({key: value["store_digest"] for key, value in receipts.items()}),
                "question_independent_writer": True,
                "source_preparation": "Previously reviewed eight-frame samples, not newly streamed raw video"})
    # Phase 2: questions are revealed only after every store above is sealed.
    cases = []
    for record, variant, projection, spec, relative in prepared:
        target = record["audit"]["target_clip"]
        start, end = record["conditions"][variant]["clip_ranges"][target - 1]
        if spec["name"] != "all":
            names = [spec["name"]]
        else:
            names = ["full"] if record["split"] == "train" else settings["conditions"]
        with MemoryStore.open(output / relative) as store:
            all_observations = store.observations(projection["episode_id"])
            for name in names:
                clock = time.monotonic()
                if name == "full":
                    if len(all_observations) != 4:
                        raise ValueError("Exact full-input baseline requires retaining all four pairs")
                    condition = deepcopy(record["conditions"][variant])
                    condition["frames"] = [frame for obs in all_observations for frame in obs["frames"]]
                    reader = {"question": record["question"], "options": record["options"],
                              "conditions": {"original": condition}}
                    provenance = {"policy": "full_input_baseline", "observation_ids": []}
                else:
                    if name in {"oracle", "oracle_all"}:
                        result = retrieve_oracle(store, episode_id=projection["episode_id"],
                                                 target_clip=target,
                                                 max_pairs=4 if name == "oracle_all" else settings["reader_pairs"])
                    else:
                        policy = {"retrieval": "episode", "wrong": "wrong_episode",
                                  "memory_full": "uniform"}.get(name, name)
                        if spec["name"] != "all":
                            policy = "episode"
                        result = retrieve(store, episode_id=projection["episode_id"], question=record["question"],
                                          policy=policy, max_pairs=4 if name == "memory_full"
                                          else settings["reader_pairs"])
                    reader = make_memory_record(record["question"], record["options"], result)
                    provenance = result.provenance
                evidence = {"target_episode_id": projection["episode_id"], "target_clip_index": target,
                            "target_pair_count": (end - start) // 2,
                            "retrieval_seconds": time.monotonic() - clock,
                            "store_bytes": sum(p.stat().st_size for p in (output / relative).rglob("*")
                                               if p.is_file()),
                            "delay_clips": len(record["conditions"][variant]["clip_ranges"]) - target}
                cases.append({"id": record["id"], "split": record["split"],
                              "condition": name if variant == "original" else f"competing_{name}",
                              "record": reader, "gold": record["answer"], "data_root": relative,
                              "store_digest": receipts[relative]["store_digest"], "evidence": evidence,
                              "retrieval": provenance, "source_group": record["source_group"],
                              "reviewer_type": record["audit"].get("reviewer_type", "unknown")})
    protocol = {"version": 1, "settings": settings, "config": config,
                "cohort_digest": digest(records), "cases_digest": digest(cases),
                "observation_commit_sha256": file_digest(output / "observation-commit.json"),
                "code_digest": code_identity(), "cohort_counts": {
                    split: sum(record["split"] == split for record in records) for split in ("train", "val", "test")},
                "interpretation": settings["interpretation"],
                "limitations": ["Existing selected and AI-reviewed cohort; no fresh blind test",
                                "Prepared short concatenated histories; no continuous 3D map",
                                "Target-clip coverage is not proof of answer-bearing pixel coverage",
                                "Memory prompt and frame selection are controlled separately",
                                "No new weight updates in this diagnostic"]}
    protocol["protocol_digest"] = digest(protocol)
    atomic_json(output / "cases.json", cases)
    atomic_json(output / "protocol.json", protocol)
    return {"cases": len(cases), "stores": len(receipts), "protocol_digest": protocol["protocol_digest"]}


def load_study(output, *, check_code=True):
    output = Path(output).resolve()
    protocol = json.loads((output / "protocol.json").read_text())
    if digest({k: v for k, v in protocol.items() if k != "protocol_digest"}) != protocol["protocol_digest"]:
        raise ValueError("Protocol digest mismatch")
    cases = json.loads((output / "cases.json").read_text())
    if digest(cases) != protocol["cases_digest"]:
        raise ValueError("Case manifest changed")
    if file_digest(output / "observation-commit.json") != protocol["observation_commit_sha256"]:
        raise ValueError("Observation commit changed")
    if check_code and code_identity() != protocol["code_digest"]:
        raise ValueError("Code changed after preparation; create a separately frozen study")
    for case in cases:
        relative = Path(case["data_root"])
        path = (output / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(output / "memory"):
            raise ValueError("Case data root must be a retained store inside this study")
        case["data_root"] = str(path)
    return protocol, cases


def run_evaluation(output, adapter=None, splits=None):
    from .evaluate import model_identity
    from .memory_eval import evaluate_cases
    from .memory_model import MemoryRuntime

    output = Path(output).resolve()
    protocol, cases = load_study(output)
    if splits:
        cases = [case for case in cases if case["split"] in splits]
    name = "tuned" if adapter else "base"
    runtime = MemoryRuntime(protocol["config"], output, adapter=adapter)
    import torch
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    rows = evaluate_cases(runtime, cases, output / name, model_name=name,
                          model_digest=model_identity(protocol["config"], adapter),
                          protocol_digest=protocol["protocol_digest"])
    atomic_json(output / f"{name}-predictions.json", rows)
    atomic_json(output / f"{name}-runtime.json", {
        "model": name, "cases": len(rows), "seconds": time.monotonic() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "protocol_digest": protocol["protocol_digest"], "fresh_process": True})
    return {"model": name, "cases": len(rows)}


def observe_file(observations, output, capacity_pairs=4, policy="episode", seed=42):
    """Public writer command. Input contains observations only, never questions."""
    from .memory import MemoryStore

    source = Path(observations).resolve()
    with MemoryStore.create(output, capacity_pairs=capacity_pairs, policy=policy, seed=seed) as store:
        for row in read_jsonl(source):
            row = deepcopy(row)
            row["frames"] = [str((source.parent / name).resolve()) for name in row["frames"]]
            store.observe(**row)
        receipt = store.seal()
        return {"store_digest": receipt["store_digest"], **store.stats()}


def answer_file(config_path, store_path, question_file, output, adapter=None, max_pairs=4, use_spatial=False):
    """Ask new questions against an already sealed store; no dataset labels required."""
    from .evaluate import _checked_scores, model_identity
    from .memory import MemoryStore
    from .memory_model import MemoryRuntime
    from .retrieval import make_memory_record, retrieve

    if Path(output).resolve().is_relative_to(Path(store_path).resolve()):
        raise ValueError("Answer output must be outside the sealed memory store")
    if Path(output).exists():
        raise FileExistsError("Answer output exists; choose a new output file")
    if type(max_pairs) is not int or not 1 <= max_pairs <= 4:
        raise ValueError("Public answers require one through four evidence pairs")
    _, config = load_study_config(config_path)
    config["memory"] = {"use_spatial": use_spatial, "observation_timeline": "provided"}
    requests = read_jsonl(question_file)
    if not requests:
        raise ValueError("Question file is empty")
    records = []
    with MemoryStore.open(store_path) as store:
        for request in requests:
            if set(request) != {"episode_id", "question", "options"}:
                raise ValueError("Question inputs require only episode_id, question and A-D options")
            result = retrieve(store, episode_id=request["episode_id"], question=request["question"],
                              max_pairs=max_pairs)
            if not result.observations:
                raise ValueError("No retained evidence for the requested episode/clip; refusing an ungrounded answer")
            records.append((make_memory_record(request["question"], request["options"], result), result))
    runtime = MemoryRuntime(config, store_path, adapter=adapter)
    model = digest({"weights": model_identity(config, adapter), "reader": config["memory"]})
    answers = []
    for record, result in records:
        started = time.monotonic()
        scores = runtime.scores(record)
        prediction, margin = _checked_scores(scores)
        answers.append({"question": record["question"], "prediction": prediction, "scores": scores,
                        "margin": margin, "seconds": time.monotonic() - started,
                        "evidence": result.provenance})
    receipt = {"model_digest": model, "code_digest": code_identity(), "answers": answers,
               "questions_sha256": file_digest(question_file), "scope": "clip-ordinal episodic visual QA"}
    atomic_json(output, receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ecqa memory")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Seal observation memories before making reader cases")
    prepare.add_argument("--config", default="configs/memory.yaml")
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--data-root", required=True)
    prepare.add_argument("--output", required=True)
    evaluate = sub.add_parser("evaluate", help="Run fixed-choice inference from sealed memories")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--adapter")
    report = sub.add_parser("report", help="Summarize complete paired memory diagnostics")
    report.add_argument("--output", required=True)
    observe = sub.add_parser("observe", help="Write question-free JSONL observations into a sealed store")
    observe.add_argument("--observations", required=True)
    observe.add_argument("--output", required=True)
    observe.add_argument("--capacity-pairs", type=int, default=4)
    observe.add_argument("--policy", choices=["episode", "recent", "reservoir"], default="episode")
    observe.add_argument("--seed", type=int, default=42)
    answer = sub.add_parser("answer", help="Answer new JSONL questions from an existing sealed store")
    answer.add_argument("--config", default="configs/memory.yaml")
    answer.add_argument("--store", required=True)
    answer.add_argument("--questions", required=True)
    answer.add_argument("--output", required=True)
    answer.add_argument("--adapter")
    answer.add_argument("--max-pairs", type=int, choices=[1, 2, 3, 4], default=4)
    answer.add_argument("--use-spatial", action="store_true", help="Include supplied frame-local spatial annotations")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        settings, config = load_study_config(args.config)
        result = prepare_study(settings, config, read_jsonl(args.manifest), args.data_root, args.output)
    elif args.command == "evaluate":
        result = run_evaluation(args.output, args.adapter)
    elif args.command == "observe":
        result = observe_file(args.observations, args.output, args.capacity_pairs, args.policy, args.seed)
    elif args.command == "answer":
        result = answer_file(args.config, args.store, args.questions, args.output,
                             args.adapter, args.max_pairs, args.use_spatial)
    else:
        from .memory_eval import write_memory_report
        protocol, cases = load_study(args.output)
        rows = [row for name in ("base", "tuned")
                for row in json.loads((Path(args.output) / f"{name}-predictions.json").read_text())]
        conditions = protocol["settings"]["conditions"]
        expected = {"train": ["full"],
                    "val": conditions + [item["name"] for item in protocol["settings"]["capacity_conditions"]],
                    "test": conditions + [f"competing_{name}" for name in conditions]}
        result = write_memory_report(cases, rows, Path(args.output) / "report", expected_conditions=expected)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
