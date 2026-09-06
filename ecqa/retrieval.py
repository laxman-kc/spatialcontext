"""Question-only retrieval from a sealed, bounded observation store.

Normal retrieval accepts no benchmark record or target annotations. The oracle
entry point is deliberately separate and marks every result as diagnostic.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re

from ecqa.model import CHOICES

_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4,
             "1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
_ORDINAL_PATTERN = re.compile(r"\b(first|second|third|fourth|1st|2nd|3rd|4th)\b", re.I)
_NUMBERED_CLIP = re.compile(r"\b(?:clip|scene|segment|video)\s*(?:number\s*)?([0-9]+)\b", re.I)
_ORDINAL_CLIP = re.compile(
    r"\b(?:first|second|third|fourth|1st|2nd|3rd|4th)\s+"
    r"(?:(?:original|video|recorded)\s+)?(?:clips?|scenes?|segments?|videos?)\b", re.I)
POLICIES = frozenset({"episode", "uniform", "recent", "wrong_episode", "empty"})


class OrdinalResolutionError(ValueError):
    """The question alone does not identify exactly one supported clip."""


@dataclass(frozen=True)
class RetrievalResult:
    observations: tuple[dict, ...]
    provenance: dict


def question_clip(question: str) -> int:
    """Resolve one explicit clip ordinal, refusing missing/ambiguous references.

Multiple ordinal words are conservatively ambiguous even when one might refer
to a non-clip object. No annotation supplies the missing interpretation.
"""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be nonempty text")
    values = {_ORDINALS[match.group(1).lower()] for match in _ORDINAL_PATTERN.finditer(question)}
    values.update(int(match.group(1)) for match in _NUMBERED_CLIP.finditer(question))
    if 0 in values:
        raise OrdinalResolutionError("Explicit numeric clip references must be positive integers")
    if not (_ORDINAL_CLIP.search(question) or _NUMBERED_CLIP.search(question)) or not values:
        raise OrdinalResolutionError("Question has no explicit supported clip ordinal")
    if len(values) != 1:
        raise OrdinalResolutionError("Question contains multiple distinct clip ordinal candidates")
    return values.pop()


def _even(items: list[dict], count: int) -> list[dict]:
    """Deterministic endpoint-spanning selection; one slot uses the midpoint."""
    if count >= len(items):
        return items
    if count == 0:
        return []
    if count == 1:
        return [items[(len(items) - 1) // 2]]
    return [items[(index * (len(items) - 1)) // (count - 1)] for index in range(count)]


def _validate_request(episode_id: str, max_pairs: int):
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("episode_id must be nonempty text")
    if type(max_pairs) is not int or not 0 <= max_pairs <= 4:
        raise ValueError("Retrieval budget must be an integer from zero to four pairs")


def _read(store, episode_id: str) -> tuple[dict, list[dict]]:
    manifest = store.verify()
    digest = manifest.get("store_digest")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Retrieval requires a verified sealed store digest")
    items = store.observations(episode_id)
    if any(item["episode_id"] != episode_id for item in items):
        raise ValueError("Store returned observations from another episode")
    items = sorted(items, key=lambda item: (item["pair_index"], item["observation_id"]))
    if len({item["observation_id"] for item in items}) != len(items):
        raise ValueError("Duplicate observation identities")
    return manifest, items


def _result(manifest: dict, episode_id: str, available: list[dict], selected: list[dict],
            *, policy: str, max_pairs: int, requested_clip: int | None,
            fallback: str | None, reason: str | None, diagnostic: bool) -> RetrievalResult:
    return RetrievalResult(tuple(deepcopy(selected)), {
        "store_digest": manifest["store_digest"], "episode_id": episode_id,
        "policy": policy, "max_pairs": max_pairs, "available_pairs": len(available),
        "selected_pairs": len(selected), "selected_frames": len(selected) * 2,
        "requested_clip": requested_clip, "fallback": fallback, "reason": reason,
        "diagnostic": diagnostic,
        "observation_ids": [item["observation_id"] for item in selected],
        "clip_indices": [item["clip_index"] for item in selected],
        "pair_indices": [item["pair_index"] for item in selected],
        "timestamps": [item["timestamp"] for item in selected],
        "source_ids": [item["source_id"] for item in selected],
        "scene_ids": [item["scene_id"] for item in selected],
        "frames": [frame for item in selected for frame in item["frames"]],
        "frame_sha256": [digest for item in selected for digest in item["frame_sha256"]],
        "evidence_usage_proven": False,
    })


def retrieve(store, *, episode_id: str, question: str, policy: str = "episode",
             max_pairs: int = 4, ordinal_fallback: str = "refuse") -> RetrievalResult:
    """Select retained evidence without using options, gold, or audit metadata.

``episode`` filters to the question's original clip; ``wrong_episode`` is a
control selecting other clips *within the same episode/video*. Neither can
recover evicted observations. Fallback is explicit and recorded.
    """
    _validate_request(episode_id, max_pairs)
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be nonempty text")
    if policy not in POLICIES:
        raise ValueError(f"Unknown retrieval policy: {policy}")
    if ordinal_fallback not in {"refuse", "uniform", "recent"}:
        raise ValueError("ordinal_fallback must be refuse, uniform, or recent")
    manifest, available = _read(store, episode_id)
    candidates = available
    requested_clip = None
    fallback = reason = None
    selection = policy
    if policy in {"episode", "wrong_episode"}:
        try:
            requested_clip = question_clip(question)
        except OrdinalResolutionError as exc:
            if ordinal_fallback == "refuse":
                raise
            fallback, reason = ordinal_fallback, str(exc)
            selection = ordinal_fallback
        else:
            if policy == "episode":
                candidates = [item for item in available if item["clip_index"] == requested_clip]
            else:
                candidates = [item for item in available if item["clip_index"] != requested_clip]
            if not candidates:
                reason = "No retained observations satisfy the question-derived clip filter"
    if selection == "empty":
        selected = []
        reason = "Empty-memory control"
    elif selection == "recent":
        selected = candidates[-max_pairs:] if max_pairs else []
    else:
        selected = _even(candidates, max_pairs)
    if not max_pairs:
        reason = "Zero retrieval budget"
    elif not available and reason is None:
        reason = "No retained observations for this episode"
    return _result(manifest, episode_id, available, selected, policy=policy, max_pairs=max_pairs,
                   requested_clip=requested_clip, fallback=fallback, reason=reason, diagnostic=False)


def retrieve_oracle(store, *, episode_id: str, target_clip: int,
                    max_pairs: int = 4) -> RetrievalResult:
    """Annotation-assisted target retrieval, permitted only as a named diagnostic."""
    _validate_request(episode_id, max_pairs)
    if type(target_clip) is not int or target_clip < 1:
        raise ValueError("Oracle target_clip must be a positive one-based integer")
    manifest, available = _read(store, episode_id)
    candidates = [item for item in available if item["clip_index"] == target_clip]
    selected = _even(candidates, max_pairs)
    return _result(manifest, episode_id, available, selected, policy="oracle", max_pairs=max_pairs,
                   requested_clip=target_clip, fallback=None,
                   reason=None if candidates else "Oracle target has no retained observations",
                   diagnostic=True)


def make_memory_record(question: str, options: dict, result: RetrievalResult) -> dict:
    """Create the entire model-facing record from an explicit allowlist."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be nonempty text")
    if not isinstance(options, dict) or set(options) != set(CHOICES) or any(
        not isinstance(options[label], str) or not options[label].strip() for label in CHOICES
    ):
        raise ValueError("Exactly four nonempty A-D options are required")
    return {"question": question, "options": {label: options[label] for label in CHOICES},
            "memory": {"episode_id": result.provenance["episode_id"],
                       "store_digest": result.provenance["store_digest"],
                       "observation_ids": [item["observation_id"] for item in result.observations]}}
