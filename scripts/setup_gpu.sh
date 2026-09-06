#!/usr/bin/env bash
set -euo pipefail
ECQA_ROOT="${ECQA_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd -- "$ECQA_ROOT"
ECQA_ROOT="$PWD"
export HF_HOME="${HF_HOME:-$ECQA_ROOT/.model-cache}"
if ! command -v uv >/dev/null 2>&1; then
  printf '%s\n' 'uv is required on PATH. Install uv before running GPU setup.' >&2
  exit 1
fi
mkdir -p artifacts/environment
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/bin/python transformers==4.57.1 peft==0.18.1 accelerate==1.12.0 huggingface_hub==0.36.0 av==13.1.0 Pillow==12.0.0 numpy==2.2.6 PyYAML==6.0.3 pytest==8.4.2 ruff==0.13.0
uv pip freeze --python .venv/bin/python > artifacts/environment/gpu-lock.txt
nvidia-smi > artifacts/environment/nvidia-smi.txt
.venv/bin/python - <<'PY'
import json, platform, torch, yaml
from pathlib import Path
from huggingface_hub import model_info, snapshot_download
config = yaml.safe_load(Path('configs/pilot.yaml').read_text())
info = model_info(config['model']['id'], revision=config['model']['revision'])
Path('artifacts/environment/model.json').write_text(json.dumps({'id':info.id,'revision':info.sha}, indent=2)+'\n')
snapshot_download(info.id, revision=info.sha, ignore_patterns=['*.gguf','*.bin','*.onnx'])
env={'python':platform.python_version(),'platform':platform.platform(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'bf16':torch.cuda.is_bf16_supported(),'model_revision':info.sha}
Path('artifacts/environment/runtime.json').write_text(json.dumps(env, indent=2)+'\n')
print(json.dumps(env, indent=2))
PY
