"""Pinned Qwen3-VL encoding, language-only LoRA, and fixed-choice scoring.

Encodings are batch-size-one CPU tensors. Only explicit question/options and the
selected condition enter the prompt; callers must supply a training answer.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

CHOICES = ("A", "B", "C", "D")


def clip_map(condition: dict) -> str:
    """Describe the actual timestamped two-frame groups in the prepared video."""
    fps = float(condition["fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Prepared synthetic fps must be positive and finite")
    ranges = condition["clip_ranges"]
    if not ranges:
        raise ValueError("At least one clip range is required")
    previous = 0
    lines = []
    for index, interval in enumerate(ranges, 1):
        if len(interval) != 2:
            raise ValueError("Clip ranges must be [start, end) pairs")
        start, end = interval
        if (not isinstance(start, int) or not isinstance(end, int)
                or start != previous or end <= start or start % 2 or end % 2):
            raise ValueError("Clip ranges must be contiguous, nonempty, and pair-aligned")
        stamps = [f"<{((i / fps) + ((i + 1) / fps)) / 2:.1f} seconds>"
                  for i in range(start, end, 2)]
        lines.append(f"Clip {index}: visual groups {start // 2 + 1}-{end // 2}; "
                     f"timestamps {', '.join(stamps)}.")
        previous = end
    frames = condition.get("frames", [])
    if frames and len(frames) != previous:
        raise ValueError("Clip ranges do not cover the prepared frames")
    return "\n".join(lines)


def build_prompt(record: dict, condition: str = "original") -> str:
    """Deliberate allowlist: do not serialize the record or its audit metadata."""
    question, options = record["question"], record["options"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be nonempty text")
    if set(options) != set(CHOICES) or any(
        not isinstance(options[label], str) or not options[label].strip() for label in CHOICES
    ):
        raise ValueError("Exactly four nonempty A-D options are required")
    mapping = clip_map(record["conditions"][condition])
    return ("Answer the spatial question using the supplied visual history. "
            "The timestamps below describe a synthetic prepared timeline. "
            "Reply with exactly one option letter: A, B, C, or D.\n\n"
            f"Clip map:\n{mapping}\n\nQuestion: {question}\n"
            + "\n".join(f"{label}. {options[label]}" for label in CHOICES))


def continuation_start(prefix_ids: list[int], full_ids: list[int]) -> int:
    if not prefix_ids or full_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("Chat continuation is not token-prefix stable")
    if len(full_ids) <= len(prefix_ids):
        raise ValueError("Assistant continuation has no tokens")
    return len(prefix_ids)


def last_valid_position(attention_mask: list[int]) -> int:
    positions = [i for i, value in enumerate(attention_mask) if value]
    if not positions:
        raise ValueError("Prompt has no valid tokens")
    return positions[-1]


class ModelRuntime:
    """Load a pinned base and optional adapter; no network-independent fallback."""

    def __init__(self, config: dict, data_root: str | Path, adapter: str | Path | None = None):
        model_cfg = config["model"]
        deterministic = bool(model_cfg.get("deterministic", False))
        if deterministic:
            # Set before creating this process's CUDA/cuBLAS context.
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.config = config
        self.data_root = Path(data_root).resolve()
        backend = model_cfg.get("attention_backend", "sdpa")
        torch.use_deterministic_algorithms(deterministic)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False
        if backend in {"sdpa", "sdpa_math"}:
            math_only = backend == "sdpa_math"
            torch.backends.cuda.enable_flash_sdp(not math_only)
            torch.backends.cuda.enable_mem_efficient_sdp(not math_only)
            torch.backends.cuda.enable_math_sdp(True)
            if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
                torch.backends.cuda.enable_cudnn_sdp(not math_only)
        revision = model_cfg.get("revision")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Model revision must be an immutable 40-character commit hash")
        processor_revision = model_cfg.get("processor_revision") or revision
        if processor_revision != revision:
            raise ValueError("Processor and base model revisions must match")
        dtype_name = model_cfg.get("dtype", "bfloat16")
        dtypes = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}
        if dtype_name not in dtypes:
            raise ValueError(f"Unsupported model dtype: {dtype_name}")
        self.device = torch.device(model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = AutoProcessor.from_pretrained(model_cfg["id"], revision=revision)
        self.processor.tokenizer.padding_side = "left"
        self.max_tokens = int(config.get("preprocessing", {}).get("max_sequence_tokens", 4096))
        self.processor.tokenizer.model_max_length = self.max_tokens
        vp = self.processor.video_processor
        if vp.temporal_patch_size != 2 or vp.merge_size != 2:
            raise ValueError("This encoder requires the audited Qwen two-frame/two-spatial merge profile")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_cfg["id"], revision=revision, torch_dtype=dtypes[dtype_name],
            attn_implementation="sdpa" if backend == "sdpa_math" else backend,
        ).to(self.device)
        if adapter is not None:
            from peft import PeftConfig, PeftModel
            saved = PeftConfig.from_pretrained(str(adapter))
            if saved.base_model_name_or_path != model_cfg["id"] or saved.revision != revision:
                raise ValueError("Adapter was not saved for the configured pinned base")
            self.model = PeftModel.from_pretrained(self.model, str(adapter), is_trainable=False)
        self.model.eval()
        self.target_modules: list[str] = []

    def _materialize(self, record: dict, condition: str):
        import numpy as np
        from PIL import Image

        selected = record["conditions"][condition]
        prompt = build_prompt(record, condition)
        files = selected["frames"]
        if condition == "text_only" and files:
            raise ValueError("Text-only condition must have no frames")
        if condition != "text_only" and not files:
            raise ValueError("Visual condition must contain prepared frames")
        content = []
        video = None
        if files:
            expected_frames = int(self.config.get("preprocessing", {}).get("frames", 8))
            if len(files) != expected_frames or len(files) % 2:
                raise ValueError("Prepared frame count differs from the fixed even-frame policy")
            arrays = []
            for name in files:
                path = (self.data_root / name).resolve()
                if not path.is_relative_to(self.data_root):
                    raise ValueError("Frame path escapes data_root")
                with Image.open(path) as picture:
                    if picture.mode != "RGB":
                        raise ValueError("Prepared frames must already be RGB")
                    arrays.append(np.array(picture, dtype=np.uint8))
            if len({array.shape for array in arrays}) != 1:
                raise ValueError("All prepared frames must share exact dimensions")
            video = np.stack(arrays)
            _, height, width, _ = video.shape
            factor = self.processor.video_processor.patch_size * self.processor.video_processor.merge_size
            if height % factor or width % factor:
                raise ValueError("Prepared H/W must be divisible by Qwen's patch/merge factor")
            prep = self.config.get("preprocessing", {})
            area = height * width
            if not int(prep.get("pixel_budget_per_frame_min", 128 * 128)) <= area <= int(
                prep.get("pixel_budget_per_frame_max", 512 * 512)
            ):
                raise ValueError("Prepared frame area is outside the frozen pixel budget")
            content.append({"type": "video"})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}], video, float(selected["fps"])

    def _process(self, text: str, video, fps: float):
        kwargs = {"text": [text], "return_tensors": "pt", "padding": False,
                  "truncation": False, "add_special_tokens": False}
        if video is not None:
            kwargs.update(videos=[video], videos_kwargs={
                "do_resize": False, "do_sample_frames": False,
                "input_data_format": "channels_last",
                "video_metadata": [{"total_num_frames": len(video), "fps": fps,
                                    "frames_indices": list(range(len(video)))}],
            })
        encoded = self.processor(**kwargs)
        if encoded["input_ids"].shape[1] > self.max_tokens:
            raise ValueError("Encoded input exceeds max_sequence_tokens; truncation is forbidden")
        allowed = {"input_ids", "attention_mask", "pixel_values_videos", "video_grid_thw"}
        if set(encoded) - allowed:
            raise ValueError(f"Unexpected processor model inputs: {set(encoded) - allowed}")
        if video is not None:
            patch = self.processor.video_processor.patch_size
            merge = self.processor.video_processor.merge_size
            expected = [len(video) // 2, video.shape[1] // patch, video.shape[2] // patch]
            if encoded["video_grid_thw"].tolist() != [expected]:
                raise ValueError("Processor changed prepared temporal or spatial geometry")
            count = int((encoded["input_ids"] == self.processor.video_token_id).sum().item())
            if count != math.prod(expected) // (merge * merge):
                raise ValueError("Visual placeholder/grid mismatch")
            if encoded["pixel_values_videos"].shape[0] != math.prod(expected):
                raise ValueError("Video patch/grid mismatch")
        elif "pixel_values_videos" in encoded or "video_grid_thw" in encoded:
            raise ValueError("Text-only encoding unexpectedly contains video tensors")
        return encoded

    def encode(self, record: dict, condition: str = "original", answer: str | None = None):
        messages, video, fps = self._materialize(record, condition)
        prefix = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt = self._process(prefix, video, fps)
        if answer is None:
            return prompt
        if answer not in CHOICES:
            raise ValueError("Training answer must be A, B, C, or D")
        full_text = self.processor.apply_chat_template(
            messages + [{"role": "assistant", "content": answer}],
            tokenize=False, add_generation_prompt=False,
        )
        full = self._process(full_text, video, fps)
        start = continuation_start(prompt["input_ids"][0].tolist(), full["input_ids"][0].tolist())
        labels = full["input_ids"].clone()
        labels[:, :start] = -100
        labels[full["attention_mask"] == 0] = -100
        full["labels"] = labels
        return full

    def move_inputs(self, encoded):
        return {key: value.to(self.device) for key, value in encoded.items()}

    def scores(self, record: dict, condition: str = "original", reference: bool = False) -> dict[str, float]:
        import torch

        messages, video, fps = self._materialize(record, condition)
        prefix_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt = self._process(prefix_text, video, fps)
        prefix_ids = prompt["input_ids"][0].tolist()
        candidates = {}
        for label in CHOICES:
            full_text = self.processor.apply_chat_template(
                messages + [{"role": "assistant", "content": label}],
                tokenize=False, add_generation_prompt=False,
            )
            if not full_text.startswith(prefix_text + label):
                raise ValueError("Template assistant content does not match the exact generation prefix")
            candidate = self._process(prefix_text + label, video, fps)
            start = continuation_start(prefix_ids, candidate["input_ids"][0].tolist())
            complete = self._process(full_text, video, fps)
            if complete["input_ids"][0].tolist()[:candidate["input_ids"].shape[1]] != candidate["input_ids"][0].tolist():
                raise ValueError("Terminator changes candidate tokenization")
            candidates[label] = (candidate, start)
        self.model.eval()
        with torch.inference_mode():
            if not reference and all(enc["input_ids"].shape[1] - start == 1
                                     for enc, start in candidates.values()):
                position = last_valid_position(prompt["attention_mask"][0].tolist())
                if position != prompt["input_ids"].shape[1] - 1:
                    raise ValueError("Optimized scorer requires no right padding")
                # Match the reference sequence AND lm_head shapes. BF16 kernels
                # can otherwise differ materially between prefix-only/last-logit
                # and full-candidate forwards. The fixed future A token is causally
                # masked at `position`; it is never a supplied gold answer.
                canonical = candidates["A"][0]
                out = self.model(**self.move_inputs(canonical), use_cache=False)
                logp = torch.log_softmax(out.logits[0, position].float(), dim=-1)
                values = {label: float(logp[enc["input_ids"][0, start].item()].item())
                          for label, (enc, start) in candidates.items()}
            else:
                values = {}
                for label, (encoded, start) in candidates.items():
                    inputs = self.move_inputs(encoded)
                    out = self.model(**inputs, use_cache=False)
                    end = inputs["input_ids"].shape[1]
                    logp = torch.log_softmax(out.logits[0, start - 1:end - 1].float(), dim=-1)
                    targets = inputs["input_ids"][0, start:end]
                    score = logp.gather(-1, targets[:, None]).sum()
                    values[label] = float(score.item())
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Nonfinite candidate scores")
        return values

    def add_lora(self):
        from peft import LoraConfig, TaskType, get_peft_model

        if hasattr(self.model, "peft_config"):
            raise ValueError("Adapters already attached")
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        pattern = re.compile(r"model\.language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)")
        targets = [name for name, _ in self.model.named_modules() if pattern.fullmatch(name)]
        expected_layers = self.model.config.text_config.num_hidden_layers
        if len(targets) != 4 * expected_layers:
            raise ValueError("Language attention target set differs from the audited architecture")
        train = self.config.get("training", {})
        adapter = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            revision=self.config["model"]["revision"],
            r=int(train.get("lora_rank", 32)), lora_alpha=int(train.get("lora_alpha", 64)),
            lora_dropout=float(train.get("lora_dropout", 0.05)),
            target_modules=targets, bias="none", modules_to_save=None, init_lora_weights=True,
        )
        self.model = get_peft_model(self.model, adapter)
        self.target_modules = targets
        trainable = [name for name, value in self.model.named_parameters() if value.requires_grad]
        if not trainable or any(".language_model.layers." not in name or ".lora_" not in name
                                for name in trainable):
            raise ValueError("Unexpected parameters enabled for training")
        self.model.config.use_cache = False
        return self.model
