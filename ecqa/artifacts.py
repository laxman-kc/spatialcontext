"""Small, content-addressed artifacts and experiment gates."""
import hashlib
import json
import math
import os
from pathlib import Path
import re


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_config(path):
    import yaml
    config = yaml.safe_load(Path(path).read_text())
    validate_config(config)
    return config


def validate_config(config):
    """Fail before downloads or GPU work when the batch-loop contract is invalid."""
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Configuration requires schema_version: 1")
    for section in ("model", "preprocessing", "training", "analysis"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
    model, prep, train, analysis = (config[key] for key in ("model", "preprocessing", "training", "analysis"))
    if not isinstance(model.get("id"), str) or not model["id"].strip():
        raise ValueError("Model id must be nonempty")
    if not isinstance(model.get("revision"), str) or not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
        raise ValueError("Resolve the base model to an immutable 40-character hexadecimal revision first")
    if model.get("processor_revision", model["revision"]) != model["revision"]:
        raise ValueError("Processor and model revisions must match")
    if model.get("attention_backend") not in {"sdpa", "sdpa_math"}:
        raise ValueError("This pilot requires one shared SDPA backend")
    if model.get("dtype", "bfloat16") != "bfloat16":
        raise ValueError("The implemented training loop requires bfloat16")
    if "deterministic" in model and not isinstance(model["deterministic"], bool):
        raise ValueError("model.deterministic must be boolean")
    if prep.get("frames") != 8 or prep.get("synthetic_fps") != 2.0:
        raise ValueError("Prepared input policy requires eight frames and synthetic_fps=2.0")
    integers = [(prep, "max_sequence_tokens"), (train, "gradient_accumulation_steps"),
                (train, "checkpoint_every_optimizer_steps"), (train, "lora_rank"),
                (analysis, "bootstrap_samples")]
    for section, field in integers:
        value = section.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    for section in (train, analysis):
        seed = section.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("Seeds must be integers from 0 through 2**32-1")
    for field in ("max_cumulative_seconds", "learning_rate", "max_grad_norm", "lora_alpha"):
        value = train.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{field} must be finite and positive")
    for field in ("warmup_ratio", "weight_decay", "lora_dropout"):
        value = train.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{field} must be finite and nonnegative")
    if train["warmup_ratio"] > 1 or train["lora_dropout"] >= 1:
        raise ValueError("warmup_ratio must be at most 1 and lora_dropout must be below 1")
    if train.get("epochs") != 1 or train.get("checkpoint_selection") != "end_of_run":
        raise ValueError("The pilot requires one epoch and end_of_run checkpoint selection")
    minimum = prep.get("pixel_budget_per_frame_min", 128 * 128)
    maximum = prep.get("pixel_budget_per_frame_max", 512 * 512)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (minimum, maximum)) or minimum > maximum:
        raise ValueError("Pixel area bounds must be ordered positive integers")


def input_identity(record, condition, data_root):
    item = record["conditions"][condition]
    root = Path(data_root).resolve()
    frames = []
    for name in item["frames"]:
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Frame must resolve inside the data root")
        frames.append({"path": name, "sha256": file_digest(path)})
    return digest({"question": record["question"], "options": record["options"],
                   "frames": frames, "clip_ranges": item["clip_ranges"], "fps": item["fps"]})


def code_identity():
    root = Path(__file__).parent
    return digest({str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob("*.py"))})


def _identity_strings(values, field, *, sha256=False):
    if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{field} must be a list of nonempty identity strings")
    if sha256 and any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in values):
        raise ValueError(f"{field} requires exact SHA256 identities")
    return values


def _known_source_gate(config, records, data_root):
    """Reject known overlap; no match is not evidence of complete independence."""
    from PIL import Image

    exclusions = config.get("exposure_exclusions")
    if not isinstance(exclusions, dict):
        raise ValueError("New freezes require an explicit exposure_exclusions record")
    exposure = set()
    for field, prefix in (("ids", "id:"), ("source_groups", "group:"),
                          ("source_clip_fingerprints", "clip:"), ("media_sha256", "media:")):
        values = _identity_strings(exclusions.get(field, []), f"exposure_exclusions.{field}",
                                   sha256=field in {"source_clip_fingerprints", "media_sha256"})
        exposure.update(prefix + value for value in values)
    root = Path(data_root).resolve()
    owners = {}
    minimum = config["preprocessing"].get("pixel_budget_per_frame_min", 128 * 128)
    maximum = config["preprocessing"].get("pixel_budget_per_frame_max", 512 * 512)
    for record in records:
        identifier, split = record["id"], record["split"]
        provenance, audit = record.get("provenance", {}), record["audit"]
        if provenance.get("source_clip_fingerprint_method") != "rgb24-all-frames-sha256-v1":
            raise ValueError(f"{identifier}: exact all-frame constituent fingerprint method is required")
        clips = _identity_strings(provenance.get("source_clip_fingerprints"),
                                  "source_clip_fingerprints", sha256=True)
        if len(clips) != len(audit["source_clip_ranges"]):
            raise ValueError(f"{identifier}: fingerprint count differs from constituent clip count")
        tokens = {"id:" + identifier, "group:" + record["source_group"]}
        tokens.update("clip:" + value for value in clips)
        if "source_media_sha256" in provenance:
            media = _identity_strings([provenance["source_media_sha256"]], "source_media_sha256", sha256=True)
            tokens.update("media:" + value for value in media)
        donor_groups = _identity_strings(audit.get("donor_source_groups", []), "donor_source_groups")
        tokens.update("group:" + value for value in donor_groups)
        donor_clips = _identity_strings(audit.get("donor_source_clip_fingerprints", []),
                                        "donor_source_clip_fingerprints", sha256=True)
        if {"neutral", "competing"}.intersection(record["conditions"]) and not donor_clips:
            raise ValueError(f"{identifier}: edited conditions need donor constituent fingerprints")
        if set(donor_clips).intersection(clips):
            raise ValueError(f"{identifier}: donor reuses a constituent of its own target source")
        tokens.update("clip:" + value for value in donor_clips)
        canvas, rgb_hashes, file_hashes = None, {}, {}
        for condition, item in record["conditions"].items():
            if condition == "text_only":
                continue
            rgb_hashes[condition], file_hashes[condition] = [], []
            for name in item["frames"]:
                path = root / name
                with Image.open(path) as image:
                    if image.mode != "RGB" or image.width % 32 or image.height % 32:
                        raise ValueError(f"{identifier}: prepared frames must be RGB and patch-aligned")
                    if not minimum <= image.width * image.height <= maximum:
                        raise ValueError(f"{identifier}: prepared frame area exceeds frozen pixel bounds")
                    if canvas is None:
                        canvas = image.size
                    if image.size != canvas:
                        raise ValueError(f"{identifier}: prepared frame geometry differs between conditions")
                    rgb_hashes[condition].append(hashlib.sha256(image.tobytes()).hexdigest())
                file_hashes[condition].append(file_digest(path))
        if "prepared_sha256" in provenance and provenance["prepared_sha256"] != file_hashes["original"]:
            raise ValueError(f"{identifier}: prepared frame provenance hashes are stale")
        if split == "test":
            if len({tuple(rgb_hashes[name]) for name in ("original", "neutral", "competing")}) != 3:
                raise ValueError(f"{identifier}: visual conditions are pixel-identical; no controlled replacement")
            if tokens.intersection(exposure):
                raise ValueError(f"{identifier}: test target or donor overlaps previously exposed material")
        tokens.add("prepared:" + digest({"canvas": canvas, "pixels": rgb_hashes["original"]}))
        for token in sorted(tokens):
            previous = owners.setdefault(token, (split, identifier))
            if previous[0] != split:
                raise ValueError(f"Known constituent/source/donor overlap across splits: {previous[1]} and {identifier}")


def freeze(config, records, data_root, path):
    from .data import assert_split_separation, validate_record, assert_condition_invariants
    validate_config(config)
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("Empty cohort or duplicate IDs")
    assert_split_separation(records)
    groups = {}
    inputs = {}
    counts = {"train": 0, "val": 0, "test": 0}
    for record in records:
        validate_record(record, require_approved=True, data_root=data_root)
        split = record["split"]
        counts[split] += 1
        group = record["source_group"]
        if not group:
            raise ValueError("Every record needs a declared conservative source group")
        if group in groups and groups[group] != split:
            raise ValueError("Known source group overlaps splits")
        groups[group] = split
        conditions = {"original", "neutral", "competing", "text_only"} if split == "test" else {"original"}
        if not conditions.issubset(record["conditions"]):
            raise ValueError("Missing required conditions")
        if split == "test":
            assert_condition_invariants(record, data_root)
        inputs[record["id"]] = {c: input_identity(record, c, data_root) for c in sorted(conditions)}
    if not all(counts.values()):
        raise ValueError("Research freeze needs nonempty train, validation, and test splits")
    _known_source_gate(config, records, data_root)
    updates = math.ceil(counts["train"] / config["training"]["gradient_accumulation_steps"])
    if updates == 1 and config["training"]["warmup_ratio"] > 0:
        raise ValueError("A one-update run needs zero warmup; otherwise its only optimizer update uses LR=0")
    body = {"schema_version": 1, "config": config, "manifest_digest": digest(records),
            "code_digest": code_identity(), "inputs": inputs, "counts": counts,
            "checkpoint_selection": "end_of_run"}
    body["protocol_digest"] = digest(body)
    selection_path = Path(path).with_name("selected-model.json")
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        if selection.get("protocol_digest") != body["protocol_digest"]:
            raise ValueError("Run directory already contains a model selection for another protocol")
    if Path(path).exists() and json.loads(Path(path).read_text()) != body:
        raise ValueError("Frozen protocol already exists with different contents; create a new version")
    atomic_json(path, body)
    return body


def verify_freeze(path, config, records, data_root):
    frozen = json.loads(Path(path).read_text())
    saved = frozen.pop("protocol_digest")
    if digest(frozen) != saved:
        raise ValueError("Protocol integrity failure")
    frozen["protocol_digest"] = saved
    if frozen["config"] != config or frozen["manifest_digest"] != digest(records):
        raise ValueError("Configuration/manifest changed after freeze")
    if frozen["code_digest"] != code_identity():
        raise ValueError("Code changed after freeze; create a new protocol before test use")
    for record in records:
        for condition, expected in frozen["inputs"][record["id"]].items():
            if input_identity(record, condition, data_root) != expected:
                raise ValueError("Prepared input changed after freeze")
    return frozen


def verify_report_inputs(protocol_path, records, predictions):
    """Bind saved predictions and labels/groups to a frozen protocol, without media.

    Completeness and numerical checks remain the responsibility of analysis.py.
    This gate needs no model, GPU, frame files, or the original training runtime.
    """
    protocol = json.loads(Path(protocol_path).read_text())
    saved = protocol.pop("protocol_digest", None)
    if not saved or digest(protocol) != saved:
        raise ValueError("Report protocol integrity failure")
    if protocol.get("manifest_digest") != digest(records):
        raise ValueError("Report manifest differs from frozen labels, groups, or inputs")
    selection = json.loads(Path(protocol_path).with_name("selected-model.json").read_text())
    if selection.get("protocol_digest") != saved or selection.get("rule") != "end_of_run":
        raise ValueError("Report model selection differs from the frozen protocol")
    if not isinstance(selection.get("model_digest"), str) or not selection["model_digest"]:
        raise ValueError("Report model selection has no tuned model identity")
    base_digest = digest({"model": protocol["config"]["model"]})
    test_ids = {record["id"] for record in records if record["split"] == "test"}
    frozen_inputs = protocol.get("inputs", {})
    for row in predictions:
        if row.get("protocol_digest") != saved:
            raise ValueError("Prediction belongs to a different report protocol")
        expected_model = base_digest if row.get("model") == "base" else selection["model_digest"]
        if row.get("model") not in {"base", "tuned"} or row.get("model_digest") != expected_model:
            raise ValueError("Report prediction model differs from the frozen base or selected adapter")
        identifier, condition = row.get("id"), row.get("condition")
        if identifier not in test_ids:
            raise ValueError("Report prediction is outside the frozen test cohort")
        expected = frozen_inputs.get(identifier, {}).get(condition)
        if not expected or row.get("input_digest") != expected:
            raise ValueError("Report prediction input differs from the frozen protocol")
    protocol["protocol_digest"] = saved
    return protocol
