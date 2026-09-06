#!/usr/bin/env bash
# Run from an isolated source checkout with the verified GPU venv available.
set -euo pipefail
ECQA_RUN="${1:-artifacts/memory-v1}"
ECQA_SOURCE_DATA="${2:-data}"
ECQA_ADAPTER="${3:-artifacts/fullstudy/train/checkpoint-000027}"
export HF_HOME="${HF_HOME:-$PWD/.model-cache}"
export HF_HUB_OFFLINE=1
export PYTHONPATH=.
mkdir -p "$ECQA_RUN"
trap 'status=$?; echo "$status" > "$ECQA_RUN/exit-code.txt"' EXIT
.venv/bin/python -m pytest -q > "$ECQA_RUN/cpu-tests.txt"
.venv/bin/ruff check ecqa tests scripts > "$ECQA_RUN/lint.txt"
if [[ ! -f "$ECQA_RUN/protocol.json" ]]; then
  .venv/bin/python -m ecqa.cli memory prepare --config configs/memory.yaml \
    --manifest "$ECQA_SOURCE_DATA/fullstudy/approved.jsonl" --data-root "$ECQA_SOURCE_DATA" \
    --output "$ECQA_RUN" > "$ECQA_RUN/prepare.log" 2>&1
fi
.venv/bin/python scripts/memory_gpu_smoke.py --study "$ECQA_RUN" > "$ECQA_RUN/gpu-smoke.log" 2>&1
.venv/bin/python -m ecqa.cli memory evaluate --output "$ECQA_RUN" > "$ECQA_RUN/base.log" 2>&1
.venv/bin/python -m ecqa.cli memory evaluate --output "$ECQA_RUN" --adapter "$ECQA_ADAPTER" \
  > "$ECQA_RUN/tuned.log" 2>&1
.venv/bin/python -m ecqa.cli memory report --output "$ECQA_RUN" > "$ECQA_RUN/report.log" 2>&1
