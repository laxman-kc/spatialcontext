"""Small, explicit data pipeline for the earlier-clip QA pilot.

All automatically produced records require review. Source filenames, inferred
scene cuts, and unique constructed-video IDs do not establish independence.
This module never constructs model prompts.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Iterable

DATASETS = {
    "train": {
        "repo": "choucsan/SIS-Motion-54K",
        "revision": "6f34a53c77d1709f7494c88833c0ce37e029913e",
        "annotation": "SIS-Motion-54K.jsonl",
    },
    "test": {
        "repo": "choucsan/SIS-Bench",
        "revision": "da50acd6e80cf27413caa27cf90aaf9a6d04e16b",
        "annotation": "SIS-Bench.jsonl",
    },
}
MODELSCOPE_REPO = "choucisan/SIS-Motion-54K-Dataset"
# Observed video-directory commit; two real pinned media downloads verified.
MODELSCOPE_REVISION = "edc34ea44f58a538622ef4221cf5a3b0fe3070a8"
LABELS = ("A", "B", "C", "D")
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4,
            "1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
ORDINAL_RE = re.compile(r"\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+clip\b", re.I)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def safe_path(data_root: str | Path, relative: str) -> Path:
    root = Path(data_root).resolve()
    supplied = Path(relative)
    if supplied.is_absolute():
        raise ValueError("artifact paths must be relative to data_root")
    resolved = (root / supplied).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("artifact escapes data_root")
    return resolved


def download_file(url: str, destination: str | Path, *, expected_sha256: str | None = None,
                  max_bytes: int = 256 * 1024 * 1024, retries: int = 3) -> dict:
    """Stream one explicitly selected public file; never snapshot a media repo."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        digest = sha256_file(destination)
        if expected_sha256 and digest != expected_sha256:
            raise ValueError(f"existing file hash mismatch: {destination}")
        if destination.stat().st_size > max_bytes:
            raise ValueError("existing file exceeds byte limit")
        return {"url": url, "path": str(destination), "bytes": destination.stat().st_size,
                "sha256": digest, "status": "cached"}
    temporary = destination.with_name(destination.name + ".part")
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "earlier-clip-qa/0.1"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
                length = response.headers.get("Content-Length")
                if length and int(length) > max_bytes:
                    raise ValueError("selected file exceeds byte limit")
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError("selected file exceeds byte limit")
                    handle.write(chunk)
            if not total:
                raise ValueError("empty download")
            digest = sha256_file(temporary)
            if expected_sha256 and digest != expected_sha256:
                raise ValueError("download hash mismatch")
            temporary.replace(destination)
            return {"url": url, "path": str(destination), "bytes": total,
                    "sha256": digest, "status": "downloaded"}
        except (OSError, urllib.error.URLError):
            if attempt + 1 == retries:
                raise
            time.sleep(min(2 ** attempt, 4))
    raise RuntimeError("unreachable download state")


def annotation_url(dataset: str) -> str:
    config = DATASETS[dataset]
    return (f"https://huggingface.co/datasets/{config['repo']}/resolve/"
            f"{config['revision']}/{config['annotation']}")


def media_url(record: dict) -> str:
    path = record["source_path"]
    if record["dataset"] == "train":
        query = urllib.parse.urlencode({"Revision": MODELSCOPE_REVISION, "FilePath": path})
        return f"https://modelscope.cn/api/v1/datasets/{MODELSCOPE_REPO}/repo?{query}"
    if not path.startswith("UAVideo/"):
        raise ValueError("unexpected benchmark media prefix")
    path = "video/" + path[len("UAVideo/"):]
    config = DATASETS["test"]
    return (f"https://huggingface.co/datasets/{config['repo']}/resolve/"
            f"{config['revision']}/{urllib.parse.quote(path, safe='/')}")


def normalize_annotations(rows: Iterable[dict], dataset: str) -> list[dict]:
    """Preserve gold and raw source records, with unknown provenance explicit."""
    config = DATASETS[dataset]
    result, seen = [], set()
    for row in rows:
        if row.get("task_type") != "positional_relationship":
            continue
        identifier = str(row["id" if dataset == "train" else "question_id"])
        identifier = f"{dataset}:{identifier}"
        if identifier in seen:
            raise ValueError(f"duplicate annotation ID: {identifier}")
        seen.add(identifier)
        if set(row["options"]) != set(LABELS) or row["answer"] not in LABELS:
            raise ValueError(f"invalid options/label for {identifier}")
        matches = ORDINAL_RE.findall(row["question"])
        target_clip = ORDINALS[matches[0].lower()] if len(matches) == 1 else None
        count = row.get("concat_num")
        if count is not None and (count not in (2, 3, 4) or (target_clip and target_clip > count)):
            raise ValueError(f"invalid clip reference: {identifier}")
        result.append({
            "id": identifier, "dataset": dataset, "dataset_revision": config["revision"],
            "source_path": row["video_path"], "raw": copy.deepcopy(row),
            "question": row["question"], "options": dict(row["options"]), "answer": row["answer"],
            "target_clip": target_clip, "clip_count": count,
            "is_earlier_candidate": target_clip < count if target_clip and count else None,
            "source_group": f"unresolved:{identifier}", "provenance_status": "unknown",
            "audit": {"status": "needs_review", "reviewer": None},
        })
    return result


def choose_candidates(records: list[dict], count: int, seed: int = 42,
                      first_clip_only: bool = True) -> list[dict]:
    candidates = [r for r in records
                  if r["is_earlier_candidate"] is not False
                  and (not first_clip_only or r["target_clip"] == 1)]
    candidates.sort(key=lambda r: r["id"])
    random.Random(seed).shuffle(candidates)
    return candidates[:count]


def source_groups(ids: Iterable[str], confirmed_edges: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Connected components of confirmed links; singleton does not imply known independence."""
    parent = {identifier: identifier for identifier in ids}

    def find(identifier: str) -> str:
        if identifier not in parent:
            raise ValueError(f"unknown source graph node: {identifier}")
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier

    for left, right in confirmed_edges:
        left, right = find(left), find(right)
        parent[max(left, right)] = min(left, right)
    components: dict[str, list[str]] = {}
    for identifier in parent:
        components.setdefault(find(identifier), []).append(identifier)
    labels = {key: "source:" + json_digest(sorted(members))[:16] for key, members in components.items()}
    return {identifier: labels[find(identifier)] for identifier in parent}


def assert_split_separation(records: list[dict], *, require_known: bool = False) -> None:
    """Catch known cross-split identities, including recorded donor groups."""
    ownership: dict[str, str] = {}
    for record in records:
        provenance = record.get("provenance_status", record.get("audit", {}).get("provenance_status", "unknown"))
        if require_known and provenance != "verified":
            raise ValueError("source independence has unresolved provenance")
        groups = [record["source_group"], *record.get("audit", {}).get("donor_source_groups", [])]
        for group in groups:
            previous = ownership.setdefault(group, record["split"])
            if previous != record["split"]:
                raise ValueError(f"known source/donor overlap across splits: {group}")


def sample_frame_indices(clip_ranges: list[list[int]], total_frames: int = 8) -> list[int]:
    """Sample using only clip boundaries and the frozen policy, never gold/evidence."""
    if len(clip_ranges) not in (2, 3, 4) or total_frames != 8:
        raise ValueError("pilot requires eight frames over two to four clips")
    counts = [2] * len(clip_ranges)
    for pair in range((total_frames - sum(counts)) // 2):
        counts[pair % len(counts)] += 2
    selected = []
    previous = None
    for (start, end), count in zip(clip_ranges, counts):
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
            raise ValueError("invalid half-open frame boundary")
        if previous is not None and start != previous:
            raise ValueError("clip boundaries must be contiguous and ordered")
        previous = end
        if end - start < count:
            raise ValueError("clip has too few distinct frames")
        indices = [min(end - 1, start + (j * (end - start)) // (count + 1))
                   for j in range(1, count + 1)]
        if len(set(indices)) != count:
            raise ValueError("sampling produced duplicate frames")
        selected.extend(indices)
    return selected


def prepared_clip_ranges(clip_count: int) -> list[list[int]]:
    allocations = {2: [4, 4], 3: [4, 2, 2], 4: [2, 2, 2, 2]}
    if clip_count not in allocations:
        raise ValueError("two to four clips required")
    end, ranges = 0, []
    for count in allocations[clip_count]:
        ranges.append([end, end + count])
        end += count
    return ranges


def canvas_size(width: int, height: int, *, max_pixels_per_frame: int = 512 * 512,
                min_pixels_per_frame: int = 128 * 128) -> tuple[int, int]:
    """Qwen-compatible (width,height) canvas; bounds are sequence budget divided by F."""
    if min(width, height) < 32 or max(width, height) / min(width, height) > 200:
        raise ValueError("unsupported image geometry")
    out_w, out_h = round(width / 32) * 32, round(height / 32) * 32
    if out_w * out_h > max_pixels_per_frame:
        scale = math.sqrt(width * height / max_pixels_per_frame)
        out_w, out_h = max(32, math.floor(width / scale / 32) * 32), max(32, math.floor(height / scale / 32) * 32)
    elif out_w * out_h < min_pixels_per_frame:
        scale = math.sqrt(min_pixels_per_frame / (width * height))
        out_w, out_h = math.ceil(width * scale / 32) * 32, math.ceil(height * scale / 32) * 32
    return out_w, out_h


def fit_canvas(image: Any, size: tuple[int, int]):
    from PIL import Image, ImageOps
    image = ImageOps.contain(image.convert("RGB"), size, method=Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", size, (0, 0, 0))
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def index_video(video_path: str | Path) -> dict:
    import av
    frame_metadata = []
    width = height = None
    with av.open(str(video_path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("source has missing presentation timestamps")
            pts = float(frame.pts * frame.time_base)
            if frame_metadata and pts < frame_metadata[-1]["pts"]:
                raise ValueError("source timestamps are not ordered")
            frame_metadata.append({"frame": index, "pts": pts})
            width, height = frame.width, frame.height
    if not frame_metadata:
        raise ValueError("video has no decodable frames")
    return {"width": width, "height": height, "frames": frame_metadata, "decoder": f"PyAV {av.__version__}"}


def prepare_record(candidate: dict, video_path: str | Path, clip_boundaries: list[list[int]],
                   data_root: str | Path, *, split: str, review: dict | None = None) -> dict:
    """Decode exact frame indices to PNG, retaining source PTS and synthetic timing.

    Boundaries are explicit inputs. Suggested boundaries are allowed for review
    artifacts but the result remains needs_review until an actual review record
    is supplied by the caller.
    """
    import av
    data_root = Path(data_root).resolve()
    metadata = index_video(video_path)
    if clip_boundaries[0][0] != 0 or clip_boundaries[-1][1] != len(metadata["frames"]):
        raise ValueError("clip boundaries must cover the decoded source")
    if candidate.get("clip_count") and candidate["clip_count"] != len(clip_boundaries):
        raise ValueError("boundary count disagrees with source annotation")
    selected = sample_frame_indices(clip_boundaries)
    target = candidate.get("target_clip")
    if target is None or not 1 <= target < len(clip_boundaries):
        raise ValueError("referenced clip is unknown or final")
    size = canvas_size(metadata["width"], metadata["height"])
    media_hash = sha256_file(video_path)
    artifact_id = json_digest({"media": media_hash, "indices": selected, "canvas": size,
                               "policy": "interior-even-pairs-v1", "decoder": metadata["decoder"]})[:24]
    output = data_root / "prepared" / artifact_id / "original"
    output.mkdir(parents=True, exist_ok=True)
    wanted = {source_index: position for position, source_index in enumerate(selected)}
    paths, hashes = [None] * 8, [None] * 8
    with av.open(str(video_path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index not in wanted:
                continue
            position = wanted[index]
            destination = output / f"frame_{position:04d}.png"
            fit_canvas(frame.to_image(), size).save(destination, format="PNG")
            paths[position] = destination.relative_to(data_root).as_posix()
            hashes[position] = sha256_file(destination)
    if any(path is None for path in paths):
        raise ValueError("source changed or selected frame could not be decoded")
    ranges = prepared_clip_ranges(len(clip_boundaries))
    audit = {"status": "needs_review", "reviewer": None, "provenance_status": candidate.get("provenance_status", "unknown"),
             "target_clip": target, "source_clip_ranges": clip_boundaries}
    if review:
        audit.update(copy.deepcopy(review))
    record = {
        "id": candidate["id"], "split": split, "source_group": candidate["source_group"],
        "question": candidate["question"], "options": dict(candidate["options"]), "answer": candidate["answer"],
        "option_permutation": {label: label for label in LABELS},
        "audit": audit,
        "conditions": {"original": {"frames": paths, "clip_ranges": ranges, "fps": 2.0},
                       "text_only": {"frames": [], "clip_ranges": copy.deepcopy(ranges), "fps": 2.0}},
        "provenance": {"dataset": candidate["dataset"], "revision": candidate["dataset_revision"],
                       "source_media_sha256": media_hash, "source_path": candidate["source_path"],
                       "frame_indices": selected, "source_pts": [metadata["frames"][i]["pts"] for i in selected],
                       "prepared_sha256": hashes, "decoder": metadata["decoder"],
                       "decoded_frame_count": len(metadata["frames"]),
                       "canvas_hw": [size[1], size[0]], "timeline": "synthetic prepared-frame order; not flight time",
                       "provenance_status": candidate.get("provenance_status", "unknown")},
    }
    validate_record(record, data_root=data_root)
    return record


def create_conditions(record: dict, neutral_donors: dict[int, list[str | Path]],
                      competing_donors: dict[int, list[str | Path]], data_root: str | Path,
                      *, donor_source_groups: list[str] | None = None) -> dict:
    """Replace the same later whole clip slots; all other frames are reused.

    Donor keys are zero-based constituent clip positions. Donors are explicit
    image paths, not automatically selected from labels or model predictions.
    Edited variants always require a fresh review.
    """
    from PIL import Image
    result = copy.deepcopy(record)
    root = Path(data_root).resolve()
    original = record["conditions"]["original"]
    ranges = original["clip_ranges"]
    target = record["audit"].get("target_clip")
    if target is None:
        raise ValueError("review metadata must identify the referenced clip")
    if not neutral_donors or set(neutral_donors) != set(competing_donors):
        raise ValueError("neutral and competing must replace the same nonempty clip slots")
    if any(not target <= index < len(ranges) for index in neutral_donors):
        raise ValueError("only clips strictly later than the target may be replaced")
    if record["source_group"] in (donor_source_groups or []):
        raise ValueError("donor shares known target source group")
    with Image.open(safe_path(root, original["frames"][0])) as image:
        size = image.size
    donor_hashes = {kind: {str(i): [sha256_file(path) for path in donors]
                          for i, donors in mapping.items()}
                    for kind, mapping in [("neutral", neutral_donors), ("competing", competing_donors)]}
    variant_id = json_digest({"original": original, "donors": donor_hashes, "canvas": size})[:24]
    replaced = []
    for kind, donors in [("neutral", neutral_donors), ("competing", competing_donors)]:
        condition = copy.deepcopy(original)
        folder = root / "prepared" / variant_id / kind
        folder.mkdir(parents=True, exist_ok=True)
        for clip_index, donor_paths in sorted(donors.items()):
            start, end = ranges[clip_index]
            if len(donor_paths) != end - start:
                raise ValueError("donor must supply exactly the whole clip allocation")
            for position, donor_path in zip(range(start, end), donor_paths):
                path = folder / f"frame_{position:04d}.png"
                with Image.open(donor_path) as image:
                    fit_canvas(image, size).save(path, format="PNG")
                condition["frames"][position] = path.relative_to(root).as_posix()
            if kind == "neutral":
                replaced.extend(range(start, end))
        result["conditions"][kind] = condition
    result["audit"].update({"status": "needs_review", "reviewer": None,
                           "replacement_clip_indices": sorted(neutral_donors),
                           "replacement_frame_indices": replaced,
                           "donor_source_groups": donor_source_groups or [], "donor_sha256": donor_hashes})
    assert_condition_invariants(result, root)
    return result


def prepare_donor_frames(video_path: str | Path, source_clip_range: list[int], count: int,
                         data_root: str | Path, canvas_wh: tuple[int, int]) -> dict:
    """Decode a specifically reviewed donor clip with the same interior policy."""
    import av
    start, end = source_clip_range
    if count not in (2, 4) or start < 0 or end - start < count:
        raise ValueError("invalid donor allocation")
    indices = [start + j * (end - start) // (count + 1) for j in range(1, count + 1)]
    digest = sha256_file(video_path)
    identity = json_digest({"media": digest, "indices": indices, "canvas": canvas_wh})[:24]
    root = Path(data_root).resolve()
    folder = root / "prepared" / identity / "donor"
    folder.mkdir(parents=True, exist_ok=True)
    positions = {index: position for position, index in enumerate(indices)}
    paths, timestamps = [None] * count, [None] * count
    with av.open(str(video_path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index not in positions:
                continue
            position = positions[index]
            destination = folder / f"frame_{position:04d}.png"
            fit_canvas(frame.to_image(), canvas_wh).save(destination, format="PNG")
            paths[position] = destination.relative_to(root).as_posix()
            timestamps[position] = float(frame.pts * frame.time_base)
    if any(path is None for path in paths):
        raise ValueError("donor allocation outside decoded source")
    return {"frames": paths, "source_media_sha256": digest, "source_clip_range": source_clip_range,
            "source_frame_indices": indices, "source_pts": timestamps, "canvas_wh": list(canvas_wh)}


def validate_record(record: dict, *, require_approved: bool = False,
                    data_root: str | Path | None = None, require_test_conditions: bool = False) -> None:
    if record.get("split") not in ("train", "val", "test") or not record.get("id") or not record.get("source_group"):
        raise ValueError("record requires id, split and source_group")
    if set(record.get("options", {})) != set(LABELS) or record.get("answer") not in LABELS:
        raise ValueError("invalid options or gold label")
    if not isinstance(record.get("question"), str) or not record["question"].strip():
        raise ValueError("missing question")
    audit = record.get("audit", {})
    if audit.get("status") not in ("approved", "needs_review"):
        raise ValueError("explicit audit status required")
    if audit.get("status") == "approved" and not audit.get("reviewer"):
        raise ValueError("approval requires named reviewer")
    if audit.get("status") == "approved":
        if audit.get("reviewer_type") not in ("human", "ai"):
            raise ValueError("approval requires explicit human/ai reviewer_type")
        for field in ("boundaries_verified", "evidence_verified", "answer_verified", "viewpoint_verified"):
            if audit.get(field) is not True:
                raise ValueError(f"approval requires {field}")
        boundaries = audit.get("source_clip_ranges")
        if not isinstance(boundaries, list) or len(boundaries) not in (2, 3, 4):
            raise ValueError("approval requires explicit source_clip_ranges")
        sample_frame_indices(boundaries)
        if boundaries[0][0] != 0:
            raise ValueError("approved source boundaries must start at zero")
        target = audit.get("target_clip")
        if not isinstance(target, int) or not 1 <= target < len(boundaries):
            raise ValueError("approved target_clip must be nonfinal")
        evidence = audit.get("evidence_frame_range")
        if (not isinstance(evidence, list) or len(evidence) != 2
                or not all(isinstance(i, int) for i in evidence)
                or not boundaries[target - 1][0] <= evidence[0] < evidence[1] <= boundaries[target - 1][1]):
            raise ValueError("approval requires evidence_frame_range within target clip")
        if not isinstance(audit.get("viewpoint"), str) or not audit["viewpoint"].strip():
            raise ValueError("approval requires an explicit viewpoint")
        decoded_count = record.get("provenance", {}).get("decoded_frame_count")
        if decoded_count is not None and boundaries[-1][1] != decoded_count:
            raise ValueError("approved boundaries do not cover decoded source")
        if len(boundaries) != len(record.get("conditions", {}).get("original", {}).get("clip_ranges", [])):
            raise ValueError("source/prepared clip counts disagree")
        for condition in ("neutral", "competing"):
            if condition in record.get("conditions", {}):
                review = audit.get("condition_review", {}).get(condition, {})
                if review.get("answer_preserved") is not True or review.get("role_verified") is not True:
                    raise ValueError(f"approval requires {condition} condition_review")
    if require_approved and audit.get("status") != "approved":
        raise ValueError(f"example requires review: {record['id']}")
    conditions = record.get("conditions", {})
    if "original" not in conditions or "text_only" not in conditions:
        raise ValueError("original and text-only conditions required")
    if require_test_conditions and record["split"] == "test" and set(conditions) != {"original", "neutral", "competing", "text_only"}:
        raise ValueError("test record requires all four conditions")
    reference_ranges = conditions["original"].get("clip_ranges")
    if reference_ranges != prepared_clip_ranges(len(reference_ranges or [])):
        raise ValueError("invalid eight-frame clip allocation")
    for kind, condition in conditions.items():
        if kind not in ("original", "neutral", "competing", "text_only"):
            raise ValueError("unknown condition")
        if condition.get("clip_ranges") != reference_ranges or condition.get("fps") != 2.0:
            raise ValueError("condition timing/boundaries must match frozen policy")
        frames = condition.get("frames", [])
        if len(frames) != (0 if kind == "text_only" else 8):
            raise ValueError("wrong frame count")
        if kind != "text_only" and len(set(frames)) != 8:
            raise ValueError("visual input contains duplicate frame paths")
        for path in frames:
            resolved = safe_path(data_root or Path.cwd(), path)
            if data_root is not None and not resolved.is_file():
                raise ValueError(f"missing prepared frame: {path}")


def assert_condition_invariants(record: dict, data_root: str | Path) -> None:
    from PIL import Image
    validate_record(record, data_root=data_root, require_test_conditions=True)
    conditions = record["conditions"]
    if not {"neutral", "competing"}.issubset(conditions):
        raise ValueError("paired visual variants missing")
    original = conditions["original"]
    replaced = set(record["audit"].get("replacement_frame_indices", []))
    clip_indices = record["audit"].get("replacement_clip_indices", [])
    expected = {i for clip in clip_indices for i in range(*original["clip_ranges"][clip])}
    target = record["audit"].get("target_clip")
    if target is None or not clip_indices or any(clip < target for clip in clip_indices) or expected != replaced:
        raise ValueError("replacement mask must contain whole clips later than target")
    with Image.open(safe_path(data_root, original["frames"][0])) as first:
        size = first.size
    for kind in ("original", "neutral", "competing"):
        condition = conditions[kind]
        for index, path in enumerate(condition["frames"]):
            with Image.open(safe_path(data_root, path)) as image:
                if image.mode != "RGB" or image.size != size:
                    raise ValueError("all paired frames require identical RGB canvas")
                pixels = image.tobytes()
            if index not in replaced:
                with Image.open(safe_path(data_root, original["frames"][index])) as reference:
                    if pixels != reference.tobytes():
                        raise ValueError("unselected/target pixels changed")


def load_manifest(path: str | Path, *, require_approved: bool = True,
                  data_root: str | Path | None = None, require_test_conditions: bool = False) -> list[dict]:
    records = read_jsonl(path)
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest has duplicate example IDs")
    for record in records:
        validate_record(record, require_approved=require_approved, data_root=data_root,
                        require_test_conditions=require_test_conditions)
    assert_split_separation(records)
    return records
