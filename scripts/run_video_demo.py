"""Reproduce one fixed actual-video demonstration with six fresh GPU predictions.

Run from the repository after installing its GPU extra. Supply the pinned MP4
and the completed memory-reader adapter. The two questions were selected from
prior AI review before this run; this is not a new held-out benchmark.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from ecqa.artifacts import atomic_json, code_identity, digest, file_digest, load_config
from ecqa.data import fit_canvas
from ecqa.evaluate import _checked_scores, model_identity
from ecqa.memory import MemoryStore
from ecqa.memory_model import MemoryRuntime
from ecqa.retrieval import make_memory_record, retrieve

SOURCE_SHA = "00fb1953fa10157fa4382b3eb00630eb562f61742863bab91311826de754a708"
SOURCE_URL = ("https://modelscope.cn/api/v1/datasets/choucisan/SIS-Motion-54K-Dataset/repo?"
              "Revision=edc34ea44f58a538622ef4221cf5a3b0fe3070a8&"
              "FilePath=AirScape_dataset%2FAirScape_Train_8918.mp4")
INDICES = [43, 86, 172, 215, 301, 344, 430, 473]
EXPECTED_PNGS = [
    "af7fc65bf879fd23190cf85d9f96f36006d60c919efcdbc0abefedfca9772cc9",
    "1afbf1ea226aab69d69092da22eed38186f7757f0a3f0247a00e486838f9c6b1",
    "ffdabb4cd7cc3396ce04c287d0719dbe0f770c3314287e6b92511a7c9db7adfc",
    "9bbda25830379cd3fae0c6a31f9be31ece1edd5e171ab870dc8d3ac63cd53025",
    "20b81d1e1d6638f6017345567fcd571cc82aee20f782dfc142664c0bf4189f77",
    "993e72ee686832f74438de882adadebaab0491255087f6eec78cf671a337ae78",
    "bb974e2b4dd3a19e79efd31eec0b5b42b3d0c1e7e30047179c31c23975321a03",
    "872105c4bb1010deaae238d532f0db70b125bb7e9f94ed4d9763ca346ee87bbd",
]
QUESTIONS = [
    {"id": "original-8918", "question": "The video consists of multiple concatenated clips. "
     "In the second clip, what object is located to the right of the cargo truck loaded with white bags?",
     "options": {"A": "The white sedan traveling in the left lane",
                 "B": "The spherical decorative street lights among green trees",
                 "C": "The lane dividing dashed white lines",
                 "D": "The overhead road sign structure spanning the highway"},
     "gold": "B", "reviewer_type": "ai", "origin": "Unchanged approved validation question"},
    {"id": "inverse-8918", "question": "In the second clip, compare the cargo truck carrying white bags "
     "with the cluster of bright spherical lights on the right-hand roadside. What is the truck's "
     "horizontal position in the image relative to that light cluster?",
     "options": {"A": "Entirely to the right of the light cluster",
                 "B": "Horizontally overlapping the light cluster",
                 "C": "Entirely to the left of the light cluster",
                 "D": "The bag-loaded truck is not visible in the supplied frames"},
     "gold": "C", "reviewer_type": "ai", "origin": "Previously AI-reviewed inverse question"},
]
EPISODE = "video-demo-8918"


def utc():
    return datetime.now(timezone.utc).isoformat()


def inventory(path):
    return {p.relative_to(path).as_posix(): file_digest(p)
            for p in sorted(path.rglob("*")) if p.is_file()}


def prepare(args):
    import av

    output, source = Path(args.output).resolve(), Path(args.source).resolve()
    if output.exists():
        raise ValueError("Use a new output directory; completed runs are immutable")
    if file_digest(source) != SOURCE_SHA:
        raise ValueError("Source MP4 differs from the pinned, previously reviewed video")
    config = load_config(args.config)
    config["memory"] = {"use_spatial": False, "observation_timeline": "synthetic"}
    output.mkdir(parents=True)
    # This commitment precedes decoding, memory construction and all new scoring.
    atomic_json(output / "questions.json", {"frozen_at_utc": utc(), "questions": QUESTIONS})
    full = output / "full"
    full.mkdir()
    pts, names, decoded = [], [], 0
    started = time.monotonic()
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        fps, duration = float(stream.average_rate), float(stream.duration * stream.time_base)
        geometry = [stream.width, stream.height]
        for index, frame in enumerate(container.decode(video=0)):
            decoded += 1
            if index in INDICES:
                name = f"frame_{len(names):04d}.png"
                picture = fit_canvas(frame.to_image(), (672, 384))
                picture.save(full / name)
                names.append(name)
                pts.append(float(frame.pts * frame.time_base))
    if decoded != 516 or fps != 24.0 or duration != 21.5:
        raise ValueError("Decoded source metadata differs from the fixed demonstration")
    if len(names) != 8 or [file_digest(full / name) for name in names] != EXPECTED_PNGS:
        raise ValueError("Freshly decoded prepared PNGs do not reproduce the reviewed evidence")
    if pts != [index / 24.0 for index in INDICES]:
        raise ValueError("Selected presentation timestamps differ")
    # Only chronological visual observations enter the writer. No question/label is passed.
    with MemoryStore.create(output / "memory", capacity_pairs=4, policy="episode", seed=42) as store:
        for pair in range(4):
            store.observe(episode_id=EPISODE, scene_id=f"clip-{pair + 1}", clip_index=pair + 1,
                          pair_index=pair, timestamp=pair + 0.25,
                          frames=[full / name for name in names[pair * 2:pair * 2 + 2]],
                          source_id=f"source-clip-{pair + 1}")
        sealed = store.seal()
    cases, selections = [], {}
    for question in QUESTIONS:
        safe = {key: question[key] for key in ("question", "options")}
        with MemoryStore.open(output / "memory") as store:
            selected = retrieve(store, episode_id=EPISODE, question=safe["question"], max_pairs=4)
        if len(selected.observations) != 1 or selected.observations[0]["clip_index"] != 2:
            raise ValueError("Question-only retrieval did not select the expected retained pair")
        memory = make_memory_record(safe["question"], safe["options"], selected)
        selections[question["id"]] = selected.provenance
        condition = {"frames": names, "fps": 2.0, "clip_ranges": [[0, 2], [2, 4], [4, 6], [6, 8]]}
        cases.extend([
            {"id": question["id"], "condition": "full", "root": "full", "frames": 8,
             "record": {**safe, "conditions": {"original": condition}}},
            {"id": question["id"], "condition": "memory", "root": "memory", "frames": 2,
             "record": memory}])
    atomic_json(output / "reader-inputs.json", {"config": config, "cases": cases})
    protocol = {
        "schema": "video-demo-protocol-v1", "created_at_utc": utc(), "code_digest": code_identity(),
        "script_sha256": file_digest(__file__), "config_digest": digest(config),
        "questions_sha256": file_digest(output / "questions.json"),
        "reader_inputs_sha256": file_digest(output / "reader-inputs.json"),
        "base_identity": model_identity(config),
        "decoder": {"pyav": av.__version__, "libraries": av.library_versions,
                    "platform": platform.platform(), "machine": platform.machine()},
        "source": {"id": "AirScape_Train_8918", "sha256": SOURCE_SHA, "url": SOURCE_URL,
                   "fps": fps, "duration": duration, "geometry_wh": geometry, "decoded_frames": decoded,
                   "sample_indices": INDICES, "sample_pts": pts, "prepared_png_sha256": EXPECTED_PNGS,
                   "clip_ranges_seconds": [[i * 5.375, (i + 1) * 5.375] for i in range(4)],
                   "synthetic_model_fps": 2.0, "all_eight_pngs_match_reviewed_evidence": True},
        "preparation_seconds": time.monotonic() - started,
        "memory": {"store_digest": sealed["store_digest"], "before": inventory(output / "memory"),
                   "capacity_pairs": 4, "retained_frames": 8, "selected_frames_per_question": 2,
                   "selections": selections, "question_blind_writer": True,
                   "sealed_before_question_retrieval": True},
    }
    protocol["preparation_digest"] = digest(protocol)
    atomic_json(output / "preparation.json", protocol)


def freeze_scoring(args):
    output = Path(args.output).resolve()
    protocol = json.loads((output / "preparation.json").read_text())
    body = {key: value for key, value in protocol.items() if key != "preparation_digest"}
    if (digest(body) != protocol["preparation_digest"] or protocol["code_digest"] != code_identity()
            or protocol["script_sha256"] != file_digest(__file__)
            or protocol["reader_inputs_sha256"] != file_digest(output / "reader-inputs.json")
            or protocol["questions_sha256"] != file_digest(output / "questions.json")
            or inventory(output / "memory") != protocol["memory"]["before"]
            or [file_digest(output / "full" / f"frame_{i:04d}.png") for i in range(8)] != EXPECTED_PNGS):
        raise ValueError("Prepared inputs changed before scoring")
    if (output / "protocol.json").exists():
        raise ValueError("This scoring protocol already exists; use a new run")
    config = json.loads((output / "reader-inputs.json").read_text())["config"]
    protocol.update(adapter_identity=model_identity(config, args.adapter),
                    adapter_files={p.name: file_digest(p) for p in Path(args.adapter).glob("adapter*")
                                   if p.is_file()}, scoring_frozen_at_utc=utc())
    protocol["digest"] = digest(protocol)
    atomic_json(output / "protocol.json", protocol)


def worker(args):
    import torch

    output = Path(args.output).resolve()
    protocol = json.loads((output / "protocol.json").read_text())
    body = {key: value for key, value in protocol.items() if key != "digest"}
    if (digest(body) != protocol["digest"] or protocol["code_digest"] != code_identity()
            or protocol["script_sha256"] != file_digest(__file__)
            or protocol["reader_inputs_sha256"] != file_digest(output / "reader-inputs.json")):
        raise ValueError("Frozen executable or input identity differs")
    payload = json.loads((output / "reader-inputs.json").read_text())
    config = payload["config"]
    adapter = args.adapter if args.worker == "tuned" else None
    if model_identity(config, adapter) != protocol["adapter_identity" if adapter else "base_identity"]:
        raise ValueError("Scoring model differs from the frozen identity")
    if not torch.cuda.is_available():
        raise RuntimeError("This recorded demonstration requires an NVIDIA GPU")
    destination = output / f"{args.worker}-predictions.json"
    if destination.exists():
        raise ValueError("Refusing to overwrite completed predictions")
    started_at, started = utc(), time.monotonic()
    runtime = MemoryRuntime(config, output, adapter=adapter)
    load_seconds = time.monotonic() - started
    rows, checks = [], []
    for condition in (["full", "memory"] if not adapter else ["memory"]):
        # During memory scoring the original decoded frame paths are absent.
        if condition == "memory":
            (output / "full").rename(output / "full.offline")
        try:
            selected = [case for case in payload["cases"] if case["condition"] == condition]
            for index, case in enumerate(selected):
                runtime.data_root = output / case["root"]
                if condition == "memory" and (output / "full").exists():
                    raise RuntimeError("Full-frame fallback was not made unavailable")
                torch.cuda.synchronize()
                before, tick = utc(), time.monotonic()
                scores = runtime.scores(case["record"])
                torch.cuda.synchronize()
                seconds, after = time.monotonic() - tick, utc()
                prediction, margin = _checked_scores(scores)
                row = {"id": case["id"], "condition": condition, "scores": scores,
                       "prediction": prediction, "margin": margin, "seconds": seconds,
                       "started_at_utc": before, "finished_at_utc": after,
                       "frame_count": case["frames"], "reader_record_digest": digest(case["record"])}
                rows.append(row)
                print(json.dumps(row), flush=True)
                if not adapter and index == 0:
                    reference = runtime.scores(case["record"], reference=True)
                    difference = max(abs(scores[key] - reference[key]) for key in scores)
                    if difference > 1e-6:
                        raise RuntimeError("Optimized fixed-choice scoring differs from the reference")
                    checks.append({"condition": condition, "max_abs_difference": difference,
                                   "reference_scores": reference})
        finally:
            if condition == "memory":
                (output / "full.offline").rename(output / "full")
    after = inventory(output / "memory")
    if after != protocol["memory"]["before"]:
        raise RuntimeError("The sealed store changed during scoring")
    atomic_json(destination, {"protocol_digest": protocol["digest"], "rows": rows,
                "reference_checks": checks, "gpu": torch.cuda.get_device_name(0),
                "started_at_utc": started_at, "finished_at_utc": utc(), "model_load_seconds": load_seconds,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "torch_version": torch.__version__, "store_unchanged": True,
                "full_paths_unavailable_during_memory_scoring": True, "model_name": args.worker})


def report(output):
    protocol = json.loads((output / "protocol.json").read_text())
    if file_digest(output / "questions.json") != protocol["questions_sha256"]:
        raise ValueError("Questions/labels changed after their pre-scoring commitment")
    questions = json.loads((output / "questions.json").read_text())["questions"]
    runs = {name: json.loads((output / f"{name}-predictions.json").read_text()) for name in ("base", "tuned")}
    if any(run["protocol_digest"] != protocol["digest"] for run in runs.values()):
        raise ValueError("Predictions belong to a different protocol")
    for question in questions:
        question["results"] = {}
        for name, condition, model in (("full", "full", "base"), ("memory_base", "memory", "base"),
                                       ("memory_tuned", "memory", "tuned")):
            result = next(row for row in runs[model]["rows"]
                          if row["id"] == question["id"] and row["condition"] == condition)
            question["results"][name] = {**result, "correct": result["prediction"] == question["gold"]}
    after = inventory(output / "memory")
    if after != protocol["memory"]["before"]:
        raise ValueError("Final memory inventory differs")
    model = json.loads((output / "reader-inputs.json").read_text())["config"]["model"]
    result = {"schema": "video-demo-v1", "source": protocol["source"], "model": model,
              "questions": questions, "memory": {**protocol["memory"], "after": after,
                  "unchanged": True, "full_paths_unavailable_during_memory_scoring": True},
              "run": {"timestamp": utc(), "gpu": runs["base"]["gpu"],
                      "decoder": protocol["decoder"],
                      "code_digest": protocol["code_digest"], "script_sha256": protocol["script_sha256"],
                      "protocol_digest": protocol["digest"], "config_digest": protocol["config_digest"],
                      "base_identity": protocol["base_identity"],
                      "adapter_identity": protocol["adapter_identity"],
                      "adapter_files": protocol["adapter_files"], "processes": runs},
              "scope": ["Two correlated AI-reviewed questions on a previously used validation video.",
                        "Fresh MP4 decoding and GPU inference; no training or new held-out test.",
                        "Full baseline uses 8 sampled frames; memory readers use 2 retained frames.",
                        "Memory retrieval parses the explicit clip ordinal; it is not learned semantic retrieval.",
                        "Full versus memory changes both evidence and prompt formatting.",
                        "Memory base versus tuned holds evidence and prompt fixed; only the adapter differs.",
                        "Scores are unnormalized answer-token log likelihoods, not calibrated confidence.",
                        "Timings include scoring and input preparation, exclude model loading/reference checks.",
                        "No automatic 3D mapping or continuous-video processing is demonstrated."]}
    atomic_json(output / "data.json", result)
    print(json.dumps({"status": "complete", "predictions": 6, "output": str(output / "data.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="Pinned original AirScape_Train_8918 MP4")
    parser.add_argument("--adapter", help="Completed memory-reader adapter directory (scoring phases)")
    parser.add_argument("--config", default="configs/fullstudy.yaml")
    parser.add_argument("--output", required=True, help="New directory for the complete run")
    parser.add_argument("--phase", choices=("all", "prepare", "score"), default="all",
                        help="Prepare on the reviewed decoder platform, then transfer losslessly for GPU score")
    parser.add_argument("--worker", choices=("base", "tuned"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.phase in {"all", "prepare"} and not args.source:
        parser.error("--source is required")
    if args.phase in {"all", "score"} and not args.adapter:
        parser.error("--adapter is required for scoring")
    if args.phase in {"all", "prepare"}:
        prepare(args)
    if args.phase == "prepare":
        print(json.dumps({"status": "prepared", "output": str(Path(args.output).resolve())}), flush=True)
        return
    freeze_scoring(args)
    for model in ("base", "tuned"):
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", model,
                   "--adapter", str(Path(args.adapter).resolve()), "--output", str(Path(args.output).resolve())]
        with (Path(args.output) / f"{model}.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    report(Path(args.output).resolve())


if __name__ == "__main__":
    main()
