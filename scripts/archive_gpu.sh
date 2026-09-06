#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 || -z "${1:-}" ]]; then
  printf '%s\n' 'Usage: bash scripts/archive_gpu.sh /path/outside/project/backup-directory' >&2
  exit 2
fi
ECQA_ROOT="${ECQA_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
ECQA_DESTINATION="$("$ECQA_ROOT/.venv/bin/python" - "$ECQA_ROOT" "$1" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).expanduser().resolve()
if destination.is_relative_to(root):
    raise SystemExit("Backup destination must be outside the project directory.")
if destination.exists() and not destination.is_dir():
    raise SystemExit("Backup destination must be a directory.")
for name in ("ecqa-backup-inventory.json", "ecqa-backup.tar.gz", "ecqa-backup.tar.gz.part", "ecqa-backup-receipt.json"):
    if (destination / name).resolve().is_relative_to(root):
        raise SystemExit("Backup output paths must be outside the project directory.")
print(destination)
PY
)"
cd -- "$ECQA_ROOT"
export PYTHONPATH=.
mkdir -p -- "$ECQA_DESTINATION"
.venv/bin/python scripts/backup_inventory.py create --root . --inventory "$ECQA_DESTINATION/ecqa-backup-inventory.json"
.venv/bin/python - "$ECQA_DESTINATION" <<'PY'
import hashlib
import json
import tarfile
import sys
from pathlib import Path
root=Path.cwd()
destination=Path(sys.argv[1])
inventory=destination/'ecqa-backup-inventory.json'
contents=json.loads(inventory.read_text())
archive=destination/'ecqa-backup.tar.gz'
temporary=archive.with_suffix('.gz.part')
with tarfile.open(temporary,'w:gz',compresslevel=1) as bundle:
    for name in sorted(contents['files']):
        bundle.add(root/name,arcname='earlier-clip-qa/'+name,recursive=False)
    bundle.add(inventory,arcname='backup-inventory.json',recursive=False)
temporary.replace(archive)
hasher=hashlib.sha256()
with archive.open('rb') as handle:
    for chunk in iter(lambda:handle.read(1024*1024),b''):
        hasher.update(chunk)
receipt={'archive':archive.name,'bytes':archive.stat().st_size,'sha256':hasher.hexdigest(),
         'file_count':contents['file_count'],'uncompressed_bytes':contents['total_bytes']}
(destination/'ecqa-backup-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
PY
