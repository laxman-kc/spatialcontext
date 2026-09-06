"""Explicit GPU check for every new memory shape against the reference scorer."""
import argparse
import json
import math
from pathlib import Path
import time

from ecqa.artifacts import atomic_json
from ecqa.memory import MemoryStore
from ecqa.memory_model import MemoryRuntime
from ecqa.memory_study import load_study
from ecqa.retrieval import make_memory_record, retrieve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    args = parser.parse_args()
    protocol, cases = load_study(args.study)
    case = next(case for case in cases if case["condition"] == "retrieval")
    runtime = MemoryRuntime(protocol["config"], case["data_root"])
    checks = []
    for count in range(5):
        with MemoryStore.open(case["data_root"]) as store:
            result = retrieve(store, episode_id=case["record"]["memory"]["episode_id"],
                              question=case["record"]["question"], policy="uniform", max_pairs=count)
        record = make_memory_record(case["record"]["question"], case["record"]["options"], result)
        started = time.monotonic()
        actual, reference = runtime.scores(record), runtime.scores(record, reference=True)
        differences = {key: abs(actual[key] - reference[key]) for key in actual}
        if any(not math.isclose(actual[key], reference[key], abs_tol=1e-6, rel_tol=0) for key in actual):
            raise RuntimeError(f"Memory scorer differs from reference for {2 * count} frames: {differences}")
        checks.append({"frames": count * 2, "scores": actual, "max_abs_difference": max(differences.values()),
                       "seconds": time.monotonic() - started})
        print(json.dumps(checks[-1]), flush=True)
    atomic_json(Path(args.study) / "gpu-smoke.json", {"status": "passed", "checks": checks,
                "protocol_digest": protocol["protocol_digest"]})


if __name__ == "__main__":
    main()
