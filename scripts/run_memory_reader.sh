#!/usr/bin/env bash
# One separately frozen validation-only candidate; never scores the old test set.
set -euo pipefail
ECQA_PARENT="${1:?Pass the completed memory study directory}"
ECQA_SOURCE="${2:?Pass the approved source manifest}"
ECQA_CANDIDATE="${3:?Pass a new candidate output directory}"
export HF_HOME="${HF_HOME:-$PWD/.model-cache}"
export HF_HUB_OFFLINE=1
export PYTHONPATH=.
trap 'status=$?; echo "$status" > "${ECQA_CANDIDATE}.exit-code.txt"' EXIT
.venv/bin/python -m ecqa.memory_train prepare --study "$ECQA_PARENT" --manifest "$ECQA_SOURCE" \
  --output "$ECQA_CANDIDATE"
.venv/bin/python -m ecqa.memory_train base --study "$ECQA_PARENT" --output "$ECQA_CANDIDATE" \
  > "$ECQA_CANDIDATE/base.log" 2>&1
.venv/bin/python -m ecqa.memory_train train --study "$ECQA_PARENT" --output "$ECQA_CANDIDATE" \
  > "$ECQA_CANDIDATE/train.log" 2>&1
.venv/bin/python -m ecqa.memory_train tuned --study "$ECQA_PARENT" --output "$ECQA_CANDIDATE" \
  > "$ECQA_CANDIDATE/tuned.log" 2>&1
.venv/bin/python -m ecqa.memory_train report --study "$ECQA_PARENT" --output "$ECQA_CANDIDATE" \
  > "$ECQA_CANDIDATE/report.log" 2>&1
