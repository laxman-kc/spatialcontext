"""Create or verify a portable integrity inventory before temporary GPU deletion."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["create", "verify"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "create":
        paths = []
        for name in ("ecqa", "configs", "scripts", "tests", "data", "artifacts"):
            paths.extend(p for p in (root / name).rglob("*") if p.is_file())
        paths.extend(p for p in root.iterdir() if p.is_file())
        files = {}
        for path in sorted(paths):
            if "__pycache__" in path.parts or path.name.endswith(".pyc"):
                continue
            if path.resolve() == args.inventory.resolve():
                continue
            if path.is_symlink():
                raise ValueError(f"Unique artifacts must not be external symlinks: {path}")
            name = path.relative_to(root).as_posix()
            files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        inventory = {"schema_version": 1, "files": files, "file_count": len(files),
                     "total_bytes": sum(item["bytes"] for item in files.values()),
                     "excluded_reproducible": [".venv", "public base model cache", "Python/tool caches"]}
        args.inventory.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    else:
        inventory = json.loads(args.inventory.read_text())
        for name, item in inventory["files"].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
                raise ValueError(f"Backup failed integrity verification: {name}")
    print(json.dumps({"status": "verified" if args.mode == "verify" else "created",
                      "file_count": inventory["file_count"], "total_bytes": inventory["total_bytes"]}))


if __name__ == "__main__":
    main()
