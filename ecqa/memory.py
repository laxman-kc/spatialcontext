"""Question-blind, bounded RGB-pair storage; sealing creates a read-only episode snapshot.

Capacity is global to one store. Prefer one store per experimental episode.
Reservoir retains the lowest seeded hash priorities. Episode retention removes
from the largest retained (episode, clip) group, breaking ties by seeded hashes.
It is causal balancing, not retrospective sampling from an archive.
"""

from collections import Counter
import json
import math
import os
from pathlib import Path
import sqlite3

from PIL import Image

from .artifacts import atomic_json, digest, file_digest


POLICIES = {"recent": "fifo-v1", "reservoir": "seeded-bottom-k-v1",
            "episode": "causal-clip-balance-bottom-k-v1"}


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 512 or "\x00" in value:
        raise ValueError(f"{name} must be a nonempty identity string of at most 512 characters")


class MemoryStore:
    """An exclusive writer until seal(); open() accepts sealed stores only."""

    def __init__(self, root, connection, writable):
        self.root, self._db, self._writable = root, connection, writable
        self._failed = False

    @classmethod
    def create(cls, path, capacity_pairs, policy="episode", seed=42):
        _integer(capacity_pairs, "capacity_pairs", 1)
        _integer(seed, "seed")
        if seed >= 2**32 or policy not in POLICIES:
            raise ValueError("Use a supported policy and seed < 2**32")
        path = Path(path)
        if path.exists() or path.is_symlink():
            raise FileExistsError("Memory store path must be new")
        root = path.resolve()
        root.mkdir(parents=True)
        (root / "frames").mkdir()
        connection = sqlite3.connect(root / "memory.sqlite")
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA secure_delete=ON")
        connection.execute("PRAGMA synchronous=FULL")
        connection.executescript("""
            CREATE TABLE metadata (value TEXT NOT NULL);
            CREATE TABLE progress (episode_id TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE observations (observation_id INTEGER PRIMARY KEY,
                episode_id TEXT NOT NULL, value TEXT NOT NULL);
        """)
        policy_info = {"name": policy, "version": POLICIES[policy], "seed": seed,
                       "capacity_pairs": capacity_pairs, "frame_format": "rgb-png-no-resampling-v1"}
        metadata = {"schema_version": 1, "policy": policy_info,
                    "policy_identity": digest(policy_info), "write_count": 0, "sealed": False}
        connection.execute("INSERT INTO metadata VALUES (?)", (json.dumps(metadata, sort_keys=True),))
        connection.commit()
        return cls(root, connection, True)

    @classmethod
    def open(cls, path):
        root = Path(path).resolve()
        if (Path(path).is_symlink() or (root / "manifest.json").is_symlink()
                or not (root / "manifest.json").is_file()):
            raise ValueError("A sealed memory manifest is required")
        manifest = json.loads((root / "manifest.json").read_text())
        body = {key: value for key, value in manifest.items() if key != "store_digest"}
        if manifest.get("store_digest") != digest(body):
            raise ValueError("Memory manifest digest differs")
        database = root / "memory.sqlite"
        if database.is_symlink() or file_digest(database) != manifest["database"]["sha256"]:
            raise ValueError("Sealed memory database differs")
        connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
        connection.execute("PRAGMA query_only=ON")
        result = cls(root, connection, False)
        try:
            result.verify()
        except Exception:
            result.close()
            raise
        return result

    def _check_open(self):
        if self._db is None:
            raise ValueError("Memory store is closed")
        if self._failed:
            raise ValueError("Memory writer failed; do not use an incomplete store")

    def _metadata(self):
        self._check_open()
        return json.loads(self._db.execute("SELECT value FROM metadata").fetchone()[0])

    def _rows(self):
        return [json.loads(row[0]) for row in self._db.execute(
            "SELECT value FROM observations ORDER BY observation_id")]

    def _image_path(self, name):
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or len(path.parts) != 2:
            raise ValueError("Retained frame path escapes the store")
        resolved = (self.root / path).resolve()
        if (path.parts[0] != "frames" or not resolved.is_relative_to(self.root)
                or (self.root / path).is_symlink() or (self.root / "frames").is_symlink()):
            raise ValueError("Retained frame path escapes the store")
        return resolved

    def _inventory(self, rows, *, sealed):
        expected = {"memory.sqlite"} | ({"manifest.json"} if sealed else set())
        expected.update(name for row in rows for name in row["frames"])
        actual = set()
        for path in self.root.rglob("*"):
            if path.is_symlink():
                raise ValueError("Memory store must not contain symlinks")
            if path.is_file():
                actual.add(path.relative_to(self.root).as_posix())
            elif path.relative_to(self.root).as_posix() != "frames":
                raise ValueError("Unexpected directory in memory store")
        if actual != expected:
            raise ValueError("Memory file inventory differs; missing or unretained files exist")

    def observe(self, *, episode_id, scene_id, clip_index, pair_index, timestamp, frames, source_id,
                spatial=None):
        """Consume one chronological pair. No question, options or gold are accepted."""
        self._check_open()
        if not self._writable:
            raise ValueError("Sealed memory is read-only")
        for name, value in (("episode_id", episode_id), ("scene_id", scene_id), ("source_id", source_id)):
            _identifier(value, name)
        _integer(clip_index, "clip_index", 1)
        _integer(pair_index, "pair_index")
        if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp) or timestamp < 0):
            raise ValueError("timestamp must be finite and nonnegative")
        if not isinstance(frames, (list, tuple)) or len(frames) != 2:
            raise ValueError("An observation requires exactly two RGB frames")
        if spatial is not None:
            from .spatial import normalize_spatial
            spatial = normalize_spatial(spatial)
        metadata = self._metadata()
        progress = self._db.execute("SELECT value FROM progress WHERE episode_id=?", (episode_id,)).fetchone()
        if progress:
            previous = json.loads(progress[0])
            if (pair_index <= previous["pair_index"] or timestamp < previous["timestamp"]
                    or clip_index < previous["clip_index"]):
                raise ValueError("Observations must be chronological within each episode")
            if clip_index == previous["clip_index"] and (scene_id, source_id) != (
                    previous["scene_id"], previous["source_id"]):
                raise ValueError("A clip cannot change scene or source identity")
        current = self._rows()
        self._inventory(current, sealed=False)
        images = []
        for source in frames:
            with Image.open(source) as image:
                image.load()
                if image.mode != "RGB" or getattr(image, "n_frames", 1) != 1:
                    raise ValueError("Retained input frames must be single RGB images")
                images.append(Image.frombytes("RGB", image.size, image.tobytes()))
        if images[0].size != images[1].size:
            raise ValueError("Both frames in a temporal pair must have the same size")
        identifier = metadata["write_count"] + 1
        row = {"observation_id": identifier, "episode_id": episode_id, "scene_id": scene_id,
               "clip_index": clip_index, "pair_index": pair_index, "timestamp": float(timestamp),
               "source_id": source_id, "frames": [f"frames/{identifier:08d}_{i}.png" for i in range(2)],
               "size": list(images[0].size)}
        if spatial is not None:
            row["spatial"] = spatial
        victim = self._victim(current + [row], metadata["policy"])
        retained = victim != identifier
        new_files = []
        try:
            if retained:
                for image, name in zip(images, row["frames"], strict=True):
                    path = self._image_path(name)
                    new_files.append(path)
                    image.save(path, format="PNG", compress_level=6)
                    with path.open("rb") as handle:
                        os.fsync(handle.fileno())
                row["frame_sha256"] = [file_digest(path) for path in new_files]
                row["frame_bytes"] = [path.stat().st_size for path in new_files]
            metadata["write_count"] = identifier
            state = {key: row[key] for key in ("episode_id", "scene_id", "source_id", "clip_index",
                                               "pair_index", "timestamp")}
            state["write_count"] = (previous["write_count"] if progress else 0) + 1
            with self._db:
                self._db.execute("UPDATE metadata SET value=?", (json.dumps(metadata, sort_keys=True),))
                self._db.execute("INSERT OR REPLACE INTO progress VALUES (?,?)",
                                 (episode_id, json.dumps(state, sort_keys=True)))
                if retained:
                    if victim is not None:
                        self._db.execute("DELETE FROM observations WHERE observation_id=?", (victim,))
                    self._db.execute("INSERT INTO observations VALUES (?,?,?)",
                                     (identifier, episode_id, json.dumps(row, sort_keys=True)))
        except Exception:
            for path in new_files:
                path.unlink(missing_ok=True)
            raise
        if retained and victim is not None:
            try:
                for old in current:
                    if old["observation_id"] == victim:
                        for name in old["frames"]:
                            self._image_path(name).unlink()
            except Exception:
                self._failed = True
                raise
        return {"observation_id": identifier, "retained": retained, "evicted_observation_id":
                victim if retained else None}

    @staticmethod
    def _victim(rows, policy):
        if len(rows) <= policy["capacity_pairs"]:
            return None
        if policy["name"] == "recent":
            return min(row["observation_id"] for row in rows)
        candidates = rows
        if policy["name"] == "episode":
            counts = Counter((row["episode_id"], row["clip_index"]) for row in rows)
            group = max(counts, key=lambda key: (counts[key], digest([policy["seed"], "group", *key])))
            candidates = [row for row in rows if (row["episode_id"], row["clip_index"]) == group]
        worst = max(candidates, key=lambda row: digest([policy["seed"], "pair", row["episode_id"],
                                                       row["pair_index"]]))
        return worst["observation_id"]

    def observations(self, episode_id):
        self._check_open()
        _identifier(episode_id, "episode_id")
        if not self._writable:
            self.verify()
        return sorted([json.loads(row[0]) for row in self._db.execute(
            "SELECT value FROM observations WHERE episode_id=?", (episode_id,))], key=lambda row: row["pair_index"])

    def _snapshot(self):
        metadata = self._metadata()
        rows = self._rows()
        database = self.root / "memory.sqlite"
        return {**metadata, "episodes": [json.loads(row[0]) for row in self._db.execute(
                    "SELECT value FROM progress ORDER BY episode_id")], "observations": rows,
                "retained_pairs": len(rows), "retained_frames": len(rows) * 2,
                "image_bytes": sum(sum(row["frame_bytes"]) for row in rows),
                "database": {"sha256": file_digest(database), "bytes": database.stat().st_size}}

    def seal(self):
        self._check_open()
        if not self._writable:
            return self.verify()
        rows = self._rows()
        self._inventory(rows, sealed=False)
        self._verify_images(rows)
        metadata = self._metadata()
        if not metadata["write_count"]:
            raise ValueError("Cannot seal a store with no observations")
        try:
            metadata["sealed"] = True
            self._db.execute("UPDATE metadata SET value=?", (json.dumps(metadata, sort_keys=True),))
            self._db.commit()
            self._db.execute("VACUUM")
            self._db.close()
            self._db = sqlite3.connect((self.root / "memory.sqlite").as_uri() + "?mode=ro", uri=True)
            self._db.execute("PRAGMA query_only=ON")
            self._writable = False
            manifest = self._snapshot()
            manifest["store_digest"] = digest(manifest)
            atomic_json(self.root / "manifest.json", manifest)
        except Exception:
            self._failed = True
            raise
        return self.verify()

    def _verify_images(self, rows):
        for row in rows:
            if "spatial" in row:
                from .spatial import normalize_spatial
                if normalize_spatial({"objects": row["spatial"]["objects"]}) != row["spatial"]:
                    raise ValueError("Retained spatial annotations are not canonical")
            if any(len(row[key]) != 2 for key in ("frames", "frame_sha256", "frame_bytes")):
                raise ValueError("Invalid retained temporal pair")
            for name, expected, size in zip(row["frames"], row["frame_sha256"], row["frame_bytes"], strict=True):
                path = self._image_path(name)
                if path.stat().st_size != size or file_digest(path) != expected:
                    raise ValueError("Retained image integrity differs")
                with Image.open(path) as image:
                    image.load()
                    if image.mode != "RGB" or list(image.size) != row["size"]:
                        raise ValueError("Retained RGB image geometry differs")

    def verify(self):
        self._check_open()
        if self._writable or not (self.root / "manifest.json").exists():
            raise ValueError("Only a sealed memory store can be verified")
        manifest = json.loads((self.root / "manifest.json").read_text())
        body = {key: value for key, value in manifest.items() if key != "store_digest"}
        if manifest.get("store_digest") != digest(body) or body != self._snapshot():
            raise ValueError("Sealed memory metadata integrity differs")
        if (not body["sealed"] or body["schema_version"] != 1
                or body["policy_identity"] != digest(body["policy"])
                or body["retained_pairs"] > body["policy"]["capacity_pairs"]):
            raise ValueError("Invalid sealed memory policy or capacity")
        self._inventory(body["observations"], sealed=True)
        self._verify_images(body["observations"])
        return manifest

    def stats(self):
        self._check_open()
        if not self._writable:
            self.verify()
        metadata = self._metadata()
        rows = self._rows()
        files = [path for path in self.root.rglob("*") if path.is_file()]
        return {"policy": metadata["policy"], "policy_identity": metadata["policy_identity"],
                "sealed": metadata["sealed"], "write_count": metadata["write_count"],
                "retained_pairs": len(rows), "retained_frames": 2 * len(rows),
                "discarded_pairs": metadata["write_count"] - len(rows),
                "image_files": 2 * len(rows), "image_bytes": sum(sum(row["frame_bytes"]) for row in rows),
                "spatial_json_bytes": sum(len(json.dumps(row["spatial"], sort_keys=True, separators=(",", ":"),
                                                         allow_nan=False).encode()) for row in rows
                                          if "spatial" in row),
                "file_count": len(files), "total_file_bytes": sum(path.stat().st_size for path in files),
                "database_bytes": (self.root / "memory.sqlite").stat().st_size,
                "manifest_bytes": (self.root / "manifest.json").stat().st_size if not self._writable else 0}

    def close(self):
        if self._db is not None:
            self._db.close()
            self._db = None

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *_):
        self.close()
