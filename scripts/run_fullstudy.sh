#!/usr/bin/env bash
set -euo pipefail
ECQA_ROOT="${ECQA_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd -- "$ECQA_ROOT"
ECQA_ROOT="$PWD"
export HF_HOME="${HF_HOME:-$ECQA_ROOT/.model-cache}"
export PYTHONPATH=.
mkdir -p artifacts/fullstudy
.venv/bin/python -m pytest -q > artifacts/fullstudy/cpu-tests.txt
.venv/bin/ruff check ecqa tests scripts > artifacts/fullstudy/lint.txt
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import torch
from transformers import AutoProcessor
from ecqa.artifacts import atomic_json, load_config
from ecqa.data import load_manifest
from ecqa.model import ModelRuntime
config=load_config('configs/fullstudy.yaml')
records=load_manifest('data/fullstudy/approved.jsonl', data_root='data', require_test_conditions=True)
runtime=object.__new__(ModelRuntime)
runtime.config=config
runtime.data_root=Path('data').resolve()
runtime.max_tokens=config['preprocessing']['max_sequence_tokens']
runtime.processor=AutoProcessor.from_pretrained(config['model']['id'],revision=config['model']['revision'])
runtime.processor.tokenizer.padding_side='left'
checks=[]
for record in records:
    original=runtime.encode(record)
    check={'id':record['id'],'tokens':original['input_ids'].shape[-1],'grid':original['video_grid_thw'].tolist()}
    if record['split']=='test':
        groups=original['video_grid_thw'][0,0].item()
        block=original['pixel_values_videos'].shape[0]//groups
        replaced=set(record['audit']['replacement_frame_indices'])
        for condition in ('neutral','competing'):
            encoded=runtime.encode(record,condition)
            for key in ('input_ids','attention_mask','video_grid_thw'):
                assert torch.equal(original[key],encoded[key]), (condition,key)
            for group in range(groups):
                if 2*group not in replaced:
                    span=slice(group*block,(group+1)*block)
                    assert torch.equal(original['pixel_values_videos'][span],encoded['pixel_values_videos'][span])
        text_only=runtime.encode(record,'text_only')
        assert 'pixel_values_videos' not in text_only and 'video_grid_thw' not in text_only
        check['matched_encoded_layout_and_preserved_target']=True
    checks.append(check)
atomic_json('artifacts/fullstudy/input-preflight.json',checks)
print(json.dumps({'preflight':'passed','examples':len(checks)}),flush=True)
PY
.venv/bin/python -m ecqa.cli freeze --config configs/fullstudy.yaml --manifest data/fullstudy/approved.jsonl --data-root data --protocol artifacts/fullstudy/protocol.json > artifacts/fullstudy/freeze.log
.venv/bin/python -m ecqa.cli train --config configs/fullstudy.yaml --manifest data/fullstudy/approved.jsonl --data-root data --protocol artifacts/fullstudy/protocol.json --output artifacts/fullstudy/train > artifacts/fullstudy/train.log 2>&1
ECQA_ADAPTER="artifacts/fullstudy/train/$(.venv/bin/python -c "import json; print(json.load(open('artifacts/fullstudy/train/summary.json'))['checkpoint'])")"
.venv/bin/python -m ecqa.cli evaluate --config configs/fullstudy.yaml --manifest data/fullstudy/approved.jsonl --data-root data --protocol artifacts/fullstudy/protocol.json --output artifacts/fullstudy/validation-base --split val > artifacts/fullstudy/validation-base.log 2>&1
.venv/bin/python -m ecqa.cli evaluate --config configs/fullstudy.yaml --manifest data/fullstudy/approved.jsonl --data-root data --protocol artifacts/fullstudy/protocol.json --output artifacts/fullstudy/validation-tuned --split val --adapter "$ECQA_ADAPTER" > artifacts/fullstudy/validation-tuned.log 2>&1
.venv/bin/python -m ecqa.cli evaluate --config configs/fullstudy.yaml --manifest data/fullstudy/approved.jsonl --data-root data --protocol artifacts/fullstudy/protocol.json --output artifacts/fullstudy/base > artifacts/fullstudy/base.log 2>&1
.venv/bin/python -m ecqa.cli evaluate --config configs/fullstudy.yaml --manifest data/fullstudy/approved.jsonl --data-root data --protocol artifacts/fullstudy/protocol.json --output artifacts/fullstudy/tuned --adapter "$ECQA_ADAPTER" > artifacts/fullstudy/tuned.log 2>&1
.venv/bin/python -m ecqa.cli report --protocol artifacts/fullstudy/protocol.json --manifest data/fullstudy/approved.jsonl --predictions artifacts/fullstudy/base/base-test.jsonl artifacts/fullstudy/tuned/tuned-test.jsonl --output artifacts/fullstudy/report
.venv/bin/python - <<'PY'
import json
from pathlib import Path
summary=json.loads(Path('artifacts/fullstudy/train/summary.json').read_text())
metrics=json.loads(Path('artifacts/fullstudy/report/metrics.json').read_text())
print(json.dumps({'status':'complete','training':summary,'cohort':metrics['cohort'],'accuracies':metrics['accuracies'],'interpretation':metrics['interpretation']},indent=2))
PY
