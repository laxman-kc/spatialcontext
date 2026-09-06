"""Bounded persistent-memory behavior without model weights or external media."""

from collections import Counter
import json
from pathlib import Path
import sqlite3

from PIL import Image
import pytest

from ecqa.memory import MemoryStore


@pytest.fixture
def frames(tmp_path):
    result = []
    for index in range(2):
        path = tmp_path / f"input-{index}.png"
        Image.new("RGB", (32, 32), (index * 100, 20, 30)).save(path)
        result.append(path)
    return result


def observe(store, frames, pair, *, episode="episode-a", clip=1, **kwargs):
    return store.observe(episode_id=episode, scene_id=f"scene-{clip}", clip_index=clip,
                         pair_index=pair, timestamp=pair * 1.0, frames=frames,
                         source_id=f"source-{clip}", **kwargs)


def test_recent_stream_is_bounded_and_eviction_removes_active_files(tmp_path, frames):
    with MemoryStore.create(tmp_path / "memory", 3, policy="recent") as store:
        for pair in range(9):
            result = observe(store, frames, pair)
            assert result["retained"]
            assert store.stats()["retained_pairs"] == min(pair + 1, 3)
            assert len(list((store.root / "frames").iterdir())) == min(pair + 1, 3) * 2
        manifest = store.seal()
        assert [row["pair_index"] for row in store.observations("episode-a")] == [6, 7, 8]
        assert manifest["write_count"] == 9
        assert manifest["retained_frames"] == 6
        assert not (store.root / "frames/00000001_0.png").exists()
        assert str(tmp_path) not in json.dumps(manifest)
        stats = store.stats()
        assert stats["discarded_pairs"] == 6
        assert stats["total_file_bytes"] == stats["image_bytes"] + stats["database_bytes"] + stats["manifest_bytes"]
        assert stats["file_count"] == 8
    frames[0].unlink()
    with MemoryStore.open(tmp_path / "memory") as opened:
        assert opened.verify() == manifest
        assert opened.seal() == manifest
        with Image.open(opened.root / opened.observations("episode-a")[0]["frames"][0]) as image:
            assert image.getpixel((0, 0)) == (0, 20, 30)
        with pytest.raises(ValueError, match="read-only"):
            observe(opened, frames, 9)
    with pytest.raises(ValueError, match="closed"):
        opened.stats()


@pytest.mark.parametrize("policy", ["reservoir", "episode"])
def test_replay_is_deterministic_and_keeps_no_evicted_archive(tmp_path, frames, policy):
    manifests = []
    for name in ("first", "replay"):
        with MemoryStore.create(tmp_path / name, 4, policy=policy, seed=71) as store:
            for pair in range(30):
                observe(store, frames, pair, clip=pair // 10 + 1)
            manifests.append(store.seal())
    assert manifests[0] == manifests[1]
    assert len(manifests[0]["observations"]) == 4
    with sqlite3.connect(tmp_path / "first/memory.sqlite") as database:
        assert database.execute("SELECT count(*) FROM observations").fetchone()[0] == 4
        assert database.execute("SELECT count(*) FROM progress").fetchone()[0] == 1
    if policy == "episode":
        counts = Counter(row["clip_index"] for row in manifests[0]["observations"])
        assert set(counts) == {1, 2, 3}
        assert sorted(counts.values()) == [1, 1, 2]
    else:
        assert min(row["pair_index"] for row in manifests[0]["observations"]) < 20


def test_episode_retention_is_bounded_when_clips_outnumber_capacity(tmp_path, frames):
    with MemoryStore.create(tmp_path / "memory", 2, policy="episode") as store:
        for pair in range(25):
            observe(store, frames, pair, clip=pair + 1)
            assert store.stats()["retained_pairs"] <= 2
        assert store.seal()["retained_pairs"] == 2
        assert len({row["clip_index"] for row in store.observations("episode-a")}) == 2


def test_chronology_and_episode_reads_are_isolated(tmp_path, frames):
    with MemoryStore.create(tmp_path / "memory", 4, policy="recent") as store:
        observe(store, frames, 0)
        observe(store, frames, 0, episode="episode-b")
        observe(store, frames, 3, clip=2)
        for pair, clip in ((3, 2), (2, 2), (4, 1)):
            with pytest.raises(ValueError, match="chronological"):
                observe(store, frames, pair, clip=clip)
        with pytest.raises(ValueError, match="chronological"):
            store.observe(episode_id="episode-a", scene_id="scene-2", source_id="source-2",
                          clip_index=2, pair_index=4, timestamp=2, frames=frames)
        store.seal()
        assert [row["pair_index"] for row in store.observations("episode-a")] == [0, 3]
        assert [row["pair_index"] for row in store.observations("episode-b")] == [0]
        assert store.observations("unseen-episode") == []


def test_writer_rejects_question_fields_and_invalid_frames_before_mutation(tmp_path, frames):
    with MemoryStore.create(tmp_path / "memory", 2) as store:
        with pytest.raises(TypeError, match="question"):
            observe(store, frames, 0, question="What is beside the tower?")
        with pytest.raises(ValueError, match="exactly two"):
            observe(store, frames[:1], 0)
        Image.new("RGBA", (32, 32)).save(frames[0])
        with pytest.raises(ValueError, match="RGB"):
            observe(store, frames, 0)
        assert store.stats()["write_count"] == 0
        assert list((store.root / "frames").iterdir()) == []


def test_failed_image_write_is_rolled_back(tmp_path, frames, monkeypatch):
    original_save = Image.Image.save

    def failing_save(image, fp, *args, **kwargs):
        if str(fp).endswith("_1.png"):
            raise OSError("synthetic disk failure")
        return original_save(image, fp, *args, **kwargs)

    with MemoryStore.create(tmp_path / "memory", 2) as store:
        with monkeypatch.context() as patch:
            patch.setattr(Image.Image, "save", failing_save)
            with pytest.raises(OSError, match="disk failure"):
                observe(store, frames, 0)
        assert store.stats()["write_count"] == 0
        assert list((store.root / "frames").iterdir()) == []
        assert observe(store, frames, 0)["observation_id"] == 1
        store.seal()


@pytest.mark.parametrize("change", ["image", "database", "manifest", "archive", "symlink"])
def test_sealed_changes_are_rejected(tmp_path, frames, change):
    root = tmp_path / "memory"
    with MemoryStore.create(root, 1) as store:
        observe(store, frames, 0)
        manifest = store.seal()
    image = root / manifest["observations"][0]["frames"][0]
    if change == "image":
        Image.new("RGB", (32, 32), "red").save(image)
    elif change == "database":
        with sqlite3.connect(root / "memory.sqlite") as database:
            database.execute("DELETE FROM observations")
    elif change == "manifest":
        manifest["write_count"] = 400
        (root / "manifest.json").write_text(json.dumps(manifest))
    elif change == "archive":
        (root / "evicted-full-history.json").write_text("[]")
    else:
        image.unlink()
        image.symlink_to(frames[0])
    with pytest.raises(ValueError):
        MemoryStore.open(root)


def test_unsealed_and_reused_paths_are_refused(tmp_path, frames):
    root = tmp_path / "memory"
    with MemoryStore.create(root, 1) as store:
        with pytest.raises(ValueError, match="no observations"):
            store.seal()
        observe(store, frames, 0)
        with pytest.raises(ValueError, match="sealed"):
            MemoryStore.open(root)
        with pytest.raises(FileExistsError):
            MemoryStore.create(root, 1)
    with pytest.raises(ValueError):
        MemoryStore.create(tmp_path / "invalid", True)


def test_eviction_failure_cannot_be_sealed_as_success(tmp_path, frames, monkeypatch):
    with MemoryStore.create(tmp_path / "memory", 1, policy="recent") as store:
        observe(store, frames, 0)
        original_unlink = Path.unlink

        def failing_unlink(path, *args, **kwargs):
            if path.name == "00000001_0.png":
                raise OSError("synthetic eviction failure")
            return original_unlink(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "unlink", failing_unlink)
            with pytest.raises(OSError, match="eviction failure"):
                observe(store, frames, 1)
        with pytest.raises(ValueError, match="writer failed"):
            store.seal()


def spatial_fixture():
    return {"objects": [
        {"frame_offset": 0, "object_id": "tower", "label": "tower", "bbox": [0.1, 0.1, 0.2, 0.3],
         "status": "ai_inferred", "confidence": 0.7},
        {"frame_offset": 0, "object_id": "bridge", "label": "bridge", "bbox": [0.6, 0.1, 0.8, 0.3],
         "status": "ai_inferred", "confidence": 0.6},
    ]}


def test_spatial_annotations_are_bound_to_retained_pair_and_accounted_once(tmp_path, frames):
    with MemoryStore.create(tmp_path / "memory", 1, policy="recent") as store:
        observe(store, frames, 0, spatial=spatial_fixture())
        row = store.observations("episode-a")[0]
        assert row["spatial"]["coordinate_system"] == "normalized_image_plane"
        assert len(row["spatial"]["relations"]) > 0
        assert store.stats()["spatial_json_bytes"] > 0
        observe(store, frames, 1)
        assert "spatial" not in store.observations("episode-a")[0]
        assert store.stats()["spatial_json_bytes"] == 0
        store.seal()
        stats = store.stats()
        assert stats["total_file_bytes"] == stats["image_bytes"] + stats["database_bytes"] + stats["manifest_bytes"]
    with MemoryStore.create(tmp_path / "annotated", 1) as annotated:
        observe(annotated, frames, 0, spatial=spatial_fixture())
        expected = annotated.seal()
    with MemoryStore.open(tmp_path / "annotated") as reopened:
        assert reopened.verify() == expected
        assert reopened.stats()["spatial_json_bytes"] > 0


def test_spatial_derived_relations_cannot_be_fabricated_before_sealing(tmp_path, frames):
    with MemoryStore.create(tmp_path / "memory", 1) as store:
        payload = spatial_fixture()
        payload["answer"] = "B"
        with pytest.raises(ValueError):
            observe(store, frames, 0, spatial=payload)
        assert store.stats()["write_count"] == 0
        observe(store, frames, 0, spatial=spatial_fixture())
        row = store.observations("episode-a")[0]
        row["spatial"]["relations"] = []
        with sqlite3.connect(store.root / "memory.sqlite") as database:
            database.execute("UPDATE observations SET value=?", (json.dumps(row),))
        with pytest.raises(ValueError, match="not canonical"):
            store.seal()


def test_seal_write_failure_marks_writer_unusable(tmp_path, frames, monkeypatch):
    with MemoryStore.create(tmp_path / "memory", 1) as store:
        observe(store, frames, 0)

        def failed_manifest(*args):
            raise OSError("synthetic manifest failure")

        monkeypatch.setattr("ecqa.memory.atomic_json", failed_manifest)
        with pytest.raises(OSError, match="manifest failure"):
            store.seal()
        with pytest.raises(ValueError, match="writer failed"):
            store.observations("episode-a")
        with pytest.raises(ValueError, match="sealed"):
            MemoryStore.open(store.root)
