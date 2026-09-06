"""CPU software proof of restart persistence, not learned QA accuracy.

Observe uses the real CLI. Answer substitutes weight loading and score values
only; retrieval, sealed-store verification, and pixel materialization stay real.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
ANSWER_PROCESS = """
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from ecqa.cli import main
from ecqa.memory_model import MemoryRuntime

materialized = []

def without_weights(self, config, data_root, adapter=None):
    self.config = config
    self.data_root = Path(data_root).resolve()
    self.processor = SimpleNamespace(
        video_processor=SimpleNamespace(patch_size=16, merge_size=2))

def fixture_scores(self, record):
    messages, video, fps = self._materialize(record)
    materialized.append({
        'reference': record['memory'],
        'pixel_sha256': hashlib.sha256(video.tobytes()).hexdigest(),
        'shape': list(video.shape),
        'fps': fps,
        'prompt': messages[0]['content'][-1]['text'],
    })
    return {'A': -1.0, 'B': -2.0, 'C': -3.0, 'D': -4.0}

MemoryRuntime.__init__ = without_weights
MemoryRuntime.scores = fixture_scores
main()
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
Path(os.environ['ECQA_TEST_AUDIT']).write_text(json.dumps({
    'pid': os.getpid(), 'materialized': materialized,
    'scope': 'synthetic CPU software proof; no model accuracy measured',
}))
"""


def _store_hashes(root):
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def test_public_cli_queries_same_sealed_pixels_after_process_restarts(tmp_path):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(PROJECT), environment.get("PYTHONPATH", "")) if part)
    # There must be no model or dataset network dependency in this proof.
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    source = tmp_path / "original source"
    source.mkdir()
    observations, expected_pixels = [], {}
    y, x = np.indices((128, 128))
    for clip in (1, 2):
        names, arrays = [], []
        for offset in (0, 1):
            index = 2 * (clip - 1) + offset
            pixels = np.stack(((x + 31 * index) % 256, (y + 17 * index) % 256,
                               np.full_like(x, 40 + index * 20)), axis=-1).astype(np.uint8)
            name = f"frame{index}.png"
            Image.fromarray(pixels).save(source / name)
            names.append(name)
            arrays.append(pixels)
        expected_pixels[clip] = hashlib.sha256(np.stack(arrays).tobytes()).hexdigest()
        observations.append({"episode_id": "observed-video", "scene_id": f"scene-{clip}",
                             "clip_index": clip, "pair_index": clip - 1,
                             "timestamp": 100.125 * clip, "source_id": f"camera-clip-{clip}",
                             "frames": names})
    observation_file = source / "observations.jsonl"
    observation_file.write_text("\n".join(json.dumps(row) for row in observations))
    store = tmp_path / "retained memory"
    observed = subprocess.run(
        [sys.executable, "-m", "ecqa.cli", "memory", "observe",
         "--observations", str(observation_file), "--output", str(store),
         "--capacity-pairs", "2", "--policy", "recent"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30,
    )
    assert observed.returncode == 0, observed.stdout + observed.stderr
    receipt = json.loads(observed.stdout)
    assert receipt["retained_pairs"] == 2 and receipt["write_count"] == 2
    sealed_hashes = _store_hashes(store)
    assert len(sealed_hashes) == 6  # SQLite, manifest, and four retained PNGs.
    shutil.rmtree(source)

    # Questions are first created after observation/sealing and source deletion.
    batches = [
        [(1, "What was visible in the first clip?"), (2, "What appeared in the second clip?")],
        [(2, "Which landmark was in the second clip?"), (1, "Which object was in the first clip?")],
    ]
    process_ids = []
    for batch_index, batch in enumerate(batches, 1):
        questions = tmp_path / f"questions-{batch_index}.jsonl"
        questions.write_text("\n".join(json.dumps({
            "episode_id": "observed-video", "question": question,
            "options": dict(zip("ABCD", ("tower", "road", "house", "river"))),
        }) for _, question in batch))
        output = tmp_path / f"answers-{batch_index}.json"
        audit = tmp_path / f"materialized-{batch_index}.json"
        answered = subprocess.run(
            [sys.executable, "-c", ANSWER_PROCESS, "memory", "answer",
             "--config", str(PROJECT / "configs" / "memory.yaml"),
             "--store", str(store), "--questions", str(questions), "--output", str(output)],
            cwd=tmp_path, env={**environment, "ECQA_TEST_AUDIT": str(audit)},
            capture_output=True, text=True, timeout=30,
        )
        assert answered.returncode == 0, answered.stdout + answered.stderr
        report = json.loads(output.read_text())
        proof = json.loads(audit.read_text())
        process_ids.append(proof["pid"])
        assert len(report["answers"]) == len(proof["materialized"]) == 2
        for (clip, question), answer, materialized in zip(
                batch, report["answers"], proof["materialized"], strict=True):
            assert answer["question"] == question
            assert answer["evidence"]["store_digest"] == receipt["store_digest"]
            assert answer["evidence"]["observation_ids"] == [clip]
            assert materialized["reference"] == {
                "episode_id": "observed-video", "store_digest": receipt["store_digest"],
                "observation_ids": [clip],
            }
            assert materialized["pixel_sha256"] == expected_pixels[clip]
            assert materialized["shape"] == [2, 128, 128, 3]
            assert materialized["fps"] == 2.0
            assert f"original Clip {clip}" in materialized["prompt"]
            assert f"original Clip {3 - clip}" not in materialized["prompt"]
        assert _store_hashes(store) == sealed_hashes
        assert not source.exists()
    assert len(set(process_ids + [os.getpid()])) == 3
