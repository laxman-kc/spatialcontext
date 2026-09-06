"""Build a local recording stage from actual GPU results and verified source media."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("results/video-demo/data.json"))
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output/playwright/video-demo"))
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    if data.get("schema") != "video-demo-v1":
        raise ValueError("Expected completed video-demo-v1 GPU results")
    source = data["source"]
    if (source["id"] != "AirScape_Train_8918" or source["fps"] != 24.0
            or source["duration"] != 21.5 or source["decoded_frames"] != 516
            or source["sample_indices"] != [43, 86, 172, 215, 301, 344, 430, 473]
            or source["clip_ranges_seconds"] != [[i * 5.375, (i + 1) * 5.375] for i in range(4)]
            or [q["id"] for q in data["questions"]] != ["original-8918", "inverse-8918"]
            or [q["gold"] for q in data["questions"]] != ["B", "C"]):
        raise ValueError("This recording template is specific to the fixed 8918 demonstration")
    for question in data["questions"]:
        for mode, count in (("full", 8), ("memory_base", 2), ("memory_tuned", 2)):
            result = question["results"][mode]
            if (result["frame_count"] != count
                    or result["prediction"] != max(result["scores"], key=result["scores"].get)
                    or result["correct"] != (result["prediction"] == question["gold"])):
                raise ValueError("Recorded prediction or frame count is inconsistent")
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != data["source"]["sha256"]:
        raise ValueError("Source video differs from the recorded GPU test")
    frames = [args.frames / f"frame_{index:04d}.png" for index in range(8)]
    if [hashlib.sha256(path.read_bytes()).hexdigest() for path in frames] != data["source"]["prepared_png_sha256"]:
        raise ValueError("Sampled frames differ from the recorded GPU test")
    template = Path(__file__).with_name("templates") / "video_demo.html"
    html = template.read_text().replace("__VIDEO_DEMO_JSON__", json.dumps(data).replace("<", "\\u003c"))
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.source, args.output / "source.mp4")
    for path in frames:
        shutil.copyfile(path, args.output / path.name)
    (args.output / "index.html").write_text(html)
    print(f"Recording stage: {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
