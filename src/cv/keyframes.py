"""
Wayfinder keyframe renderer.

Produces the four visual layers the UI offers (raw / detections / segmentation /
hybrid) for one clip, with a provenance banner burned into every frame. The banner
is deliberately impossible to miss: these clips are public reference footage, and the
UI must never imply they are RTL's own cameras.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

BANNER_BG = (18, 16, 12)
BANNER_FG = (235, 235, 235)
BANNER_ACCENT = (60, 170, 250)


def load_sources(path: str = "data/footage/SOURCES.json") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fit_width(img: np.ndarray, max_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / w
    return cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)


def draw_banner(
    img: np.ndarray,
    lines: List[str],
    max_width: Optional[int] = None,
) -> np.ndarray:
    """Adds a provenance banner above the frame (expands the canvas, crops nothing)."""
    if max_width:
        img = _fit_width(img, max_width)
    h, w = img.shape[:2]
    scale = max(0.5, min(1.6, w / 1280.0))
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.42 * scale
    th = max(1, int(round(1.4 * scale)))
    pad = int(round(8 * scale))
    line_h = int(round(20 * scale))
    banner_h = pad * 2 + line_h * len(lines)

    canvas = np.zeros((h + banner_h, w, 3), dtype=np.uint8)
    canvas[banner_h:, :] = img
    canvas[:banner_h, :] = BANNER_BG
    cv2.line(canvas, (0, banner_h - 1), (w, banner_h - 1), BANNER_ACCENT, max(1, int(scale)))

    y = pad + int(line_h * 0.72)
    for i, text in enumerate(lines):
        color = BANNER_FG if i else BANNER_ACCENT
        cv2.putText(canvas, text, (pad, y), font, fs, color, th, cv2.LINE_AA)
        y += line_h
    return canvas


def draw_waiting_zone(img: np.ndarray, bounds_frac: Optional[Dict[str, float]]) -> np.ndarray:
    """Outlines the derived waiting zone so the auto-derived region is auditable."""
    if not bounds_frac:
        return img
    out = img.copy()
    h, w = out.shape[:2]
    x1 = int(bounds_frac["x1_frac"] * w)
    y1 = int(bounds_frac["y1_frac"] * h)
    x2 = int(bounds_frac["x2_frac"] * w)
    y2 = int(bounds_frac["y2_frac"] * h)
    scale = max(0.5, min(2.0, w / 1280.0))
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), max(2, int(2 * scale)))
    label = "derived waiting zone (auto)"
    font_scale = 0.45 * scale
    thickness = max(1, int(scale))
    (tw, thh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    # Badge INSIDE the top edge of the zone so it never floats over unrelated pixels.
    cv2.rectangle(out, (x1, y1), (x1 + tw + 10, y1 + thh + 10), (0, 200, 255), -1)
    cv2.putText(
        out, label, (x1 + 5, y1 + thh + 4),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), thickness, cv2.LINE_AA,
    )
    return out


def extract_frame(clip_path: str, at_fraction: float = 0.0) -> Tuple[np.ndarray, float]:
    """Grabs a representative frame. Returns (frame, timestamp_seconds)."""
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open clip: {clip_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idx = max(0, int(total * at_fraction))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"could not read frame {idx} from {clip_path}")
    return frame, round(idx / fps, 2)


def render_keyframes(
    clip_path: str,
    out_dir: str,
    detector,
    sam3=None,
    analysis: Optional[Dict[str, Any]] = None,
    source_meta: Optional[Dict[str, Any]] = None,
    at_fraction: float = 0.0,
    max_width: int = 1600,
) -> Dict[str, Any]:
    """
    Renders raw / yolo / sam3 / hybrid JPEGs for one clip and returns a manifest.

    ``analysis`` is the serialised stop_analysis artifact, used for the waiting-zone
    overlay and for the telemetry line in the banner, so the picture and the numbers
    come from the same run.

    The frame defaults to the START of the clip because the live monitor also starts
    there: the moment a stop is opened, the drawn boxes and the playing picture then
    show the same instant instead of being seconds apart.
    """
    os.makedirs(out_dir, exist_ok=True)
    frame, ts = extract_frame(clip_path, at_fraction)
    clip_name = os.path.basename(clip_path)
    meta = (source_meta or {}).get(clip_name, {})

    detection = detector.detect_frame(frame)
    prov = [
        f"REFERENCE FOOTAGE - {meta.get('title', clip_name)}",
        f"Source: {meta.get('source', 'unknown')} | {meta.get('license', 'unknown')} | {meta.get('disclosure', 'not RTL footage')}",
    ]
    if analysis:
        m = analysis.get("metrics", {})
        conf = analysis.get("confidence", {}).get("grade", "low")
        prov.append(
            f"t={ts}s | detected in frame: {detection['count']} | measured waiting inflow: "
            f"{m.get('arrival_rate_pax_per_min', 0)} pax/min | queue mean: {m.get('queue_mean_pax', 0)} "
            f"| confidence: {conf}"
        )
    else:
        prov.append(f"t={ts}s | detected in frame: {detection['count']}")

    bounds = (analysis or {}).get("waiting_zone", {}).get("bounds_frac")

    outputs: Dict[str, str] = {}

    raw = draw_banner(frame, prov, max_width=max_width)
    p = os.path.join(out_dir, "raw.jpg")
    cv2.imwrite(p, raw, [cv2.IMWRITE_JPEG_QUALITY, 88])
    outputs["raw"] = p

    yolo = detector.render_overlay(frame, detection, show_labels=True, label="YOLOv8 detections")
    yolo = draw_waiting_zone(yolo, bounds)
    yolo = draw_banner(yolo, prov, max_width=max_width)
    p = os.path.join(out_dir, "detections.jpg")
    cv2.imwrite(p, yolo, [cv2.IMWRITE_JPEG_QUALITY, 88])
    outputs["detections"] = p

    sam_meta: Dict[str, Any] = {"available": False}
    if sam3 is not None:
        seg = sam3.segment_frame(frame, text_prompts=["waiting passenger"], conf_threshold=0.45)
        sam = sam3.render_overlay(frame, seg)
        sam = draw_waiting_zone(sam, bounds)
        sam = draw_banner(sam, prov, max_width=max_width)
        p = os.path.join(out_dir, "segmentation.jpg")
        cv2.imwrite(p, sam, [cv2.IMWRITE_JPEG_QUALITY, 88])
        outputs["segmentation"] = p
        sam_meta = {
            "available": True,
            "masks": seg["total_segments"],
            "presence_score": seg["results"][0]["presence_score"] if seg["results"] else 0.0,
            "used_low_confidence_fallback": seg.get("used_low_confidence_fallback", False),
        }

        # Hybrid: segmentation underneath, detection boxes on top. Both captions are
        # preserved by drawing the SAM caption inside the banner lines instead of
        # letting two top-left badges overwrite each other.
        hybrid = sam3.render_overlay(frame, seg)
        hybrid = detector.render_overlay(
            hybrid, detection, show_labels=False, label="YOLOv8 detections"
        )
        hybrid = draw_waiting_zone(hybrid, bounds)
        hybrid = draw_banner(hybrid, prov, max_width=max_width)
        p = os.path.join(out_dir, "hybrid.jpg")
        cv2.imwrite(p, hybrid, [cv2.IMWRITE_JPEG_QUALITY, 88])
        outputs["hybrid"] = p

    return {
        "clip": clip_name,
        "timestamp_s": ts,
        "frame_size": [frame.shape[1], frame.shape[0]],
        "detections_in_frame": detection["count"],
        "segmentation": sam_meta,
        "sources": meta,
        "images": {k: os.path.relpath(v, os.path.dirname(out_dir)) for k, v in outputs.items()},
        "absolute_paths": outputs,
    }
