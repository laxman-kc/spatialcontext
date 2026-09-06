"""Validate supplied boxes and derive conservative frame-local image-plane relations.

This module neither detects objects nor verifies the supplied annotation status.
No cross-frame identity, depth, world-coordinate, or camera-relative inference.
"""

from __future__ import annotations

import math

STATUSES = frozenset({"measured", "ai_inferred", "human_verified"})
SEPARATION_MARGIN = 0.02
ANNOTATION_NOTICE = (
    "Objects and boxes are supplied annotations, not automatically extracted or verified. "
    "Status values are supplied provenance claims; confidence values are uncalibrated. "
    "Axes are local to each image: x increases rightward and y downward. "
    "Relations describe separated image-plane boxes, not depth or world geometry."
)


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _text(value, name, limit):
    if (not isinstance(value, str) or not value.strip() or len(value) > limit
            or any(ord(char) < 32 for char in value)):
        raise ValueError(f"{name} must be nonempty text without control characters")
    return value.strip()


def _axis_relation(a1, a2, b1, b2, low, high):
    # Boundary equality and gaps <= margin remain explicitly unknown.
    if b1 - a2 > SEPARATION_MARGIN + 1e-12:
        return low
    if a1 - b2 > SEPARATION_MARGIN + 1e-12:
        return high
    return "unknown"


def normalize_spatial(payload: dict) -> dict:
    """Accept only {'objects': [...]} and regenerate all derived information.

Object IDs are local to frame_offset 0 or 1, not tracking IDs. Strictly
separated full boxes establish axis relations; overlapping axes are unknown.
    """
    if not isinstance(payload, dict) or set(payload) != {"objects"}:
        raise ValueError("Spatial input must contain only supplied objects")
    if not isinstance(payload["objects"], list) or len(payload["objects"]) > 128:
        raise ValueError("Spatial objects must be a list of at most 128 annotations")
    required = {"frame_offset", "object_id", "label", "bbox", "status"}
    objects, identities = [], set()
    for supplied in payload["objects"]:
        if (not isinstance(supplied, dict) or not required <= set(supplied)
                or set(supplied) - required - {"confidence"}):
            raise ValueError("Invalid supplied object annotation fields")
        offset = supplied["frame_offset"]
        if type(offset) is not int or offset not in {0, 1}:
            raise ValueError("frame_offset must be integer zero or one within the retained pair")
        object_id = _text(supplied["object_id"], "object_id", 128)
        if (offset, object_id) in identities:
            raise ValueError("Object IDs must be unique within each frame")
        identities.add((offset, object_id))
        if not isinstance(supplied["bbox"], list) or len(supplied["bbox"]) != 4:
            raise ValueError("bbox must be four normalized coordinates [x1,y1,x2,y2]")
        box = [_number(value, "bbox coordinate") for value in supplied["bbox"]]
        if not all(0 <= value <= 1 for value in box) or box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("bbox must be normalized, nonempty, and ordered")
        if not isinstance(supplied["status"], str) or supplied["status"] not in STATUSES:
            raise ValueError("Annotation status must be measured, ai_inferred, or human_verified")
        item = {"frame_offset": offset, "object_id": object_id,
                "label": _text(supplied["label"], "label", 256), "bbox": box, "status": supplied["status"]}
        if "confidence" in supplied:
            confidence = _number(supplied["confidence"], "confidence")
            if not 0 <= confidence <= 1:
                raise ValueError("Uncalibrated confidence must be between zero and one")
            item["confidence"] = confidence
        objects.append(item)
    objects.sort(key=lambda item: (item["frame_offset"], item["object_id"]))
    relations = []
    for a in objects:
        for b in objects:
            if a["frame_offset"] != b["frame_offset"] or a["object_id"] == b["object_id"]:
                continue
            ax1, ay1, ax2, ay2 = a["bbox"]
            bx1, by1, bx2, by2 = b["bbox"]
            relations.append({"frame_offset": a["frame_offset"], "subject_id": a["object_id"],
                              "object_id": b["object_id"],
                              "horizontal": _axis_relation(ax1, ax2, bx1, bx2, "left", "right"),
                              "vertical": _axis_relation(ay1, ay2, by1, by2, "above", "below")})
    return {"schema_version": 1, "coordinate_system": "normalized_image_plane",
            "annotation_notice": ANNOTATION_NOTICE, "separation_margin": SEPARATION_MARGIN,
            "objects": objects, "relations": relations}


def format_spatial_evidence(canonical: dict) -> str:
    """Render validated supplied annotations; never trust supplied relations."""
    if not isinstance(canonical, dict) or "objects" not in canonical:
        raise ValueError("Canonical supplied spatial evidence is required")
    expected = normalize_spatial({"objects": canonical["objects"]})
    if canonical != expected:
        raise ValueError("Spatial evidence differs from regenerated canonical annotations")
    lines = [ANNOTATION_NOTICE]
    if not canonical["objects"]:
        return "\n".join(lines + ["No supplied spatial annotations."])
    for item in canonical["objects"]:
        confidence = (f"; uncalibrated confidence {item['confidence']:.6g}"
                      if "confidence" in item else "")
        lines.append(f"Pair frame_offset {item['frame_offset']}: object {item['object_id']} "
                     f"({item['label']}), supplied status {item['status']}, "
                     f"normalized bbox {item['bbox']}{confidence}.")
    for relation in canonical["relations"]:
        lines.append(f"Pair frame_offset {relation['frame_offset']}: {relation['subject_id']} relative to "
                     f"{relation['object_id']}: horizontal {relation['horizontal']}; "
                     f"vertical {relation['vertical']} (derived from supplied boxes).")
    return "\n".join(lines)
