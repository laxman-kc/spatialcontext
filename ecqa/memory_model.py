"""Use the audited model/scorer with only retained, sealed memory pixels."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from ecqa.model import CHOICES, ModelRuntime


def build_memory_prompt(question: str, options: dict, observations: list[dict], fps: float = 2.0,
                        *, observation_timeline: str = "synthetic") -> str:
    """Map local model timestamps to unchanged original clip identities."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be nonempty text")
    if not isinstance(options, dict) or set(options) != set(CHOICES) or any(
        not isinstance(options[label], str) or not options[label].strip() for label in CHOICES
    ):
        raise ValueError("Exactly four nonempty A-D options are required")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Prepared fps must be positive and finite")
    if not isinstance(observation_timeline, str) or observation_timeline not in {"synthetic", "provided"}:
        raise ValueError("memory.observation_timeline must be synthetic or provided")
    if observation_timeline == "synthetic":
        timeline_text = (
            "Prepared timestamps identify the local supplied frame groups; observation timestamps "
            "belong to the synthetic prepared-frame timeline, not real elapsed flight time. ")
    else:
        timeline_text = (
            "Prepared timestamps are synthetic identifiers for the local supplied frame groups. "
            "Observation timestamps are observer-supplied values; their time origin, units, and semantics "
            "are unknown beyond the caller's supplied metadata. ")
    lines = []
    for index, item in enumerate(observations):
        timestamp = float(item["timestamp"])
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Observation timestamps must be finite and nonnegative")
        if type(item["clip_index"]) is not int or item["clip_index"] < 1:
            raise ValueError("Original clip indices must be positive integers")
        local_stamp = ((2 * index / fps) + ((2 * index + 1) / fps)) / 2
        observation_stamp = (f"{timestamp:.6f} seconds" if observation_timeline == "synthetic"
                             else f"{timestamp:.6f} (observer-supplied)")
        lines.append(f"Visual group {index + 1}, local reference frames {2 * index + 1}-{2 * index + 2}, "
                     f"prepared timestamp <{local_stamp:.1f} seconds>: "
                     f"original Clip {item['clip_index']}, original pair index {item['pair_index']}, "
                     f"observation timestamp {observation_stamp}.")
    evidence = "\n".join(lines) if lines else "No visual observations are retained for this answer."
    return ("Answer the spatial question using the supplied retained visual history. "
            "Original clip numbers refer to the observed video and are not renumbered after retrieval. "
            f"{timeline_text}"
            "Only listed observations are available. "
            "Reply with exactly one option letter: A, B, C, or D.\n\n"
            f"Evidence map:\n{evidence}\n\nQuestion: {question}\n"
            + "\n".join(f"{label}. {options[label]}" for label in CHOICES))


class MemoryRuntime(ModelRuntime):
    """Reuse weights and fixed-choice scorer; no persistent tensor or KV cache.

Memory records resolve IDs only inside the currently selected sealed store.
For an explicit full-input baseline, plain records delegate to the unchanged
base materializer. The evaluator sets data_root separately for each case.
    """

    def _materialize(self, record: dict, condition: str = "original"):
        if "memory" not in record:
            return super()._materialize(record, condition)
        if condition != "original":
            raise ValueError("Memory records use the original condition with explicit selected observations")
        import numpy as np
        from PIL import Image

        from ecqa.memory import MemoryStore

        root = Path(self.data_root).resolve()
        reference = record["memory"]
        if not isinstance(reference, dict) or set(reference) != {
            "episode_id", "store_digest", "observation_ids"
        }:
            raise ValueError("Invalid memory reference allowlist")
        ids = reference["observation_ids"]
        if (not isinstance(ids, list) or len(ids) > 4
                or any(type(value) is not int or value < 1 for value in ids) or len(set(ids)) != len(ids)):
            raise ValueError("Memory reference requires at most four unique positive observation IDs")
        with MemoryStore.open(root) as store:
            manifest = store.verify()
            if reference["store_digest"] != manifest["store_digest"]:
                raise ValueError("Memory record/store digest mismatch")
            retained = {item["observation_id"]: item for item in store.observations(reference["episode_id"])}
            if any(value not in retained for value in ids):
                raise ValueError("Memory reference contains unavailable or cross-episode observations")
            observations = [retained[value] for value in ids]
            if observations != sorted(observations, key=lambda item: (item["pair_index"], item["observation_id"])):
                raise ValueError("Memory observations must preserve original chronological order")
            arrays = []
            for item in observations:
                if len(item["frames"]) != 2 or len(item["frame_sha256"]) != 2:
                    raise ValueError("Each retained observation must contain exactly one frame pair")
                for name, digest in zip(item["frames"], item["frame_sha256"]):
                    relative = Path(name)
                    path = (root / relative).resolve()
                    if relative.is_absolute() or not path.is_relative_to(root):
                        raise ValueError("Retained frame path escapes the sealed store")
                    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                        raise ValueError("Retained frame hash differs from sealed evidence")
                    with Image.open(path) as picture:
                        if picture.format != "PNG" or picture.mode != "RGB":
                            raise ValueError("Retained frames must be lossless RGB PNG images")
                        if list(picture.size) != item["size"]:
                            raise ValueError("Retained frame geometry differs from sealed evidence")
                        arrays.append(np.array(picture, dtype=np.uint8))
        fps = 2.0
        prompt = build_memory_prompt(
            record["question"], record["options"], observations, fps,
            observation_timeline=self.config.get("memory", {}).get("observation_timeline", "synthetic"))
        use_spatial = self.config.get("memory", {}).get("use_spatial", False)
        if type(use_spatial) is not bool:
            raise ValueError("memory.use_spatial must be an explicit boolean")
        if use_spatial:
            from ecqa.spatial import format_spatial_evidence

            supplied = []
            for index, item in enumerate(observations):
                if "spatial" in item:
                    supplied.append(
                        f"Visual group {index + 1}, original Clip {item['clip_index']}: "
                        f"pair frame_offset 0 is local reference frame {2 * index + 1}; "
                        f"pair frame_offset 1 is local reference frame {2 * index + 2}.\n"
                        + format_spatial_evidence(item["spatial"]))
            if supplied:
                prompt += "\n\nOptional supplied spatial annotations:\n" + "\n\n".join(supplied)
        video = None
        content = []
        if arrays:
            if len({array.shape for array in arrays}) != 1:
                raise ValueError("Retained frames must share exact dimensions; resizing is forbidden")
            video = np.stack(arrays)
            _, height, width, _ = video.shape
            vp = self.processor.video_processor
            factor = vp.patch_size * vp.merge_size
            if height % factor or width % factor:
                raise ValueError("Retained H/W must be divisible by Qwen's patch/merge factor")
            prep = self.config.get("preprocessing", {})
            if not int(prep.get("pixel_budget_per_frame_min", 128 * 128)) <= height * width <= int(
                prep.get("pixel_budget_per_frame_max", 512 * 512)
            ):
                raise ValueError("Retained frame area is outside the configured pixel budget")
            content.append({"type": "video"})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}], video, fps
