"""GPU plumbing check for supplied spatial observations, using a synthetic fixture.

These rendered boxes and their exact annotations test integration, not detection
or real-world spatial reasoning. No study questions, labels or stores are used.
"""
import argparse
from copy import deepcopy
import math
from pathlib import Path

from PIL import Image, ImageDraw

from ecqa.artifacts import atomic_json, code_identity
from ecqa.memory import MemoryStore
from ecqa.memory_model import MemoryRuntime
from ecqa.memory_study import load_study_config
from ecqa.retrieval import make_memory_record, retrieve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((32, 96, 79, 159), fill="red")
    draw.rectangle((176, 96, 223, 159), fill="blue")
    frame = output / "fixture.png"
    image.save(frame)
    objects = []
    for offset in (0, 1):
        for label, box in (("red rectangle", [32/256, 96/256, 80/256, 160/256]),
                           ("blue rectangle", [176/256, 96/256, 224/256, 160/256])):
            objects.append({"object_id": label.replace(" ", "_"), "label": label, "frame_offset": offset,
                            "bbox": box, "status": "measured"})
    with MemoryStore.create(output / "store", capacity_pairs=1) as store:
        store.observe(episode_id="rendered-fixture", scene_id="rendered-scene", clip_index=1,
                      pair_index=0, timestamp=0.25, frames=[frame, frame], source_id="fixture-renderer",
                      spatial={"objects": objects})
        manifest = store.seal()
    frame.unlink()
    question = "In the first clip, where is the red rectangle relative to the blue rectangle?"
    options = {"A": "To its left", "B": "To its right", "C": "Above it", "D": "Below it"}
    with MemoryStore.open(output / "store") as store:
        selected = retrieve(store, episode_id="rendered-fixture", question=question, max_pairs=1)
    record = make_memory_record(question, options, selected)
    _, config = load_study_config("configs/memory.yaml")
    config = deepcopy(config)
    config["memory"] = {"use_spatial": True, "observation_timeline": "provided"}
    runtime = MemoryRuntime(config, output / "store")
    scores, reference = runtime.scores(record), runtime.scores(record, reference=True)
    differences = [abs(scores[key] - reference[key]) for key in scores]
    if any(not math.isclose(scores[key], reference[key], abs_tol=1e-6, rel_tol=0) for key in scores):
        raise RuntimeError("Spatial evidence scorer/reference mismatch")
    prediction = sorted(scores, key=lambda key: (-scores[key], key))[0]
    with MemoryStore.open(output / "store") as store:
        if store.verify()["store_digest"] != manifest["store_digest"]:
            raise RuntimeError("Store changed during spatial scoring")
    atomic_json(output / "result.json", {"status": "passed", "code_digest": code_identity(),
                "store_digest": manifest["store_digest"], "scores": scores, "prediction": prediction,
                "fixture_gold": "A", "fixture_correct": prediction == "A",
                "max_abs_reference_difference": max(differences), "source_image_removed": True,
                "scope": "Synthetic supplied-box integration only; no object detection or benchmark claim"})


if __name__ == "__main__":
    main()
