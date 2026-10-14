"""
Wayfinder segmentation.

The project originally used SAM 3 for everything. Measured on this machine's CPU, one
SAM 3 pass at its required 1008x1008 input takes about 14 seconds, and the model rejects
any other input resolution, so it can never run per frame. It is a strong model being
used for a job that does not need it.

A YOLOv8 segmentation model does what this application actually needs - a mask per
person, in the same forward pass that produces the bounding boxes - and on the same CPU:

    model            ms/frame    fps     weights
    yolov8n-seg        39.2      25.5    6.7 MB
    yolo11n-seg       112.2       8.9    5.9 MB
    yolov8s-seg       237.0       4.2   22.8 MB
    SAM 3            ~14000       0.07    3.4 GB

So the live path uses a YOLO segmentation model, which makes per-frame masks affordable
and removes the need for a second slow worker. SAM 3 stays available as an optional
"deep pass" for occasional close inspection, because it genuinely segments better on
ambiguous objects - it is just far too slow to be the default.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class LiveSegmentation:
    """One segmentation result, in the shape the dashboard consumes."""

    model: str
    device: str
    latency_ms: float
    person_count: int
    masks: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self, frame_w: int, frame_h: int) -> Dict[str, Any]:
        return {
            "model": self.model,
            "device": self.device,
            "latency_ms": round(self.latency_ms, 1),
            "person_count": self.person_count,
            "masks": self.masks,
            "error": self.error,
            "frame_size": [frame_w, frame_h],
        }


def _polygons_from_mask(mask: np.ndarray, min_area_frac: float) -> List[List[List[float]]]:
    """Converts a binary mask into simplified normalised polygons."""
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[List[List[float]]] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area_frac * w * h:
            continue
        # Keep silhouettes recognisable: a coarse epsilon turns people into blobs.
        epsilon = 0.0015 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        out.append([[round(float(x) / w, 5), round(float(y) / h, 5)] for x, y in approx])
    return out


class YoloSegmenter:
    """
    Fast per-frame person segmentation and detection in a single pass.

    Masks are returned as normalised polygons so the browser can draw them over the
    video without the server encoding a JPEG per update.
    """

    def __init__(
        self,
        weights: str = "yolov8n-seg.pt",
        conf_thresh: float = 0.35,
        min_mask_area_frac: float = 0.0005,
        imgsz: int = 640,
    ):
        from ultralytics import YOLO

        self.weights = weights
        self.conf_thresh = conf_thresh
        self.min_mask_area_frac = min_mask_area_frac
        self.imgsz = imgsz
        self.model = YOLO(weights)
        self.name = weights.replace(".pt", "")
        self.person_class = 0  # COCO
        self.is_sam3 = False

    @property
    def device_label(self) -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # pragma: no cover
            return "unknown"

    def segment(self, frame: np.ndarray) -> LiveSegmentation:
        started = time.time()
        try:
            result = self.model.predict(
                source=frame,
                classes=[self.person_class],
                conf=self.conf_thresh,
                imgsz=self.imgsz,
                verbose=False,
                retina_masks=True,
            )[0]
        except Exception as exc:  # keep monitoring alive if one frame fails
            return LiveSegmentation(
                model=self.name,
                device=self.device_label,
                latency_ms=(time.time() - started) * 1000.0,
                person_count=0,
                error=f"{type(exc).__name__}: {exc}",
            )

        latency = (time.time() - started) * 1000.0
        h, w = frame.shape[:2]
        masks: List[Dict[str, Any]] = []
        boxes: List[Tuple[float, float, float, float]] = []

        boxes_t = getattr(result, "boxes", None)
        if boxes_t is not None and len(boxes_t) > 0:
            boxes = [tuple(float(v) for v in b) for b in boxes_t.xyxy.cpu().numpy()]

        masks_t = getattr(result, "masks", None)
        if masks_t is not None and getattr(masks_t, "data", None) is not None:
            raw = masks_t.data.cpu().numpy()
            for index, mask in enumerate(raw):
                binary = (mask > 0.5).astype(np.uint8)
                if binary.shape[:2] != (h, w):
                    binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
                polygons = _polygons_from_mask(binary, self.min_mask_area_frac)
                if not polygons:
                    continue
                box = boxes[index] if index < len(boxes) else None
                masks.append(
                    {
                        "score": None,
                        "polygons": polygons,
                        "box": [round(float(v) / (w if i % 2 == 0 else h), 5) for i, v in enumerate(box)]
                        if box
                        else None,
                    }
                )

        return LiveSegmentation(
            model=self.name,
            device=self.device_label,
            latency_ms=latency,
            person_count=len(boxes),
            masks=masks,
        )

    def segment_frame(self, frame: np.ndarray, **_kwargs: Any) -> Dict[str, Any]:
        """
        Compatibility shim for the SAM 3 interface used by the offline renderer.

        Returns the same envelope shape (image_size, results, total_segments) so the
        keyframe pipeline can use either model without branching.
        """
        result = self.segment(frame)
        h, w = frame.shape[:2]
        return {
            "image_size": [w, h],
            "results": [
                {
                    "prompt": "person",
                    "presence_score": None,
                    "count": len(result.masks),
                    "masks": result.masks,
                    "used_low_confidence_fallback": False,
                }
            ],
            "total_segments": len(result.masks),
            "used_low_confidence_fallback": False,
        }

    def mask_polygons(self, segment_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Polygons already normalised by ``segment``; kept for interface parity."""
        out: List[Dict[str, Any]] = []
        for prompt_data in segment_results.get("results", []):
            for mask in prompt_data.get("masks", []):
                out.append(
                    {
                        "prompt": prompt_data.get("prompt"),
                        "score": mask.get("score"),
                        "area_frac": mask.get("area_frac"),
                        "polygons": mask.get("polygons", []),
                    }
                )
        return out

    def render_overlay(self, frame: np.ndarray, segment_results: Dict[str, Any], alpha: float = 0.35) -> np.ndarray:
        """Draws mask outlines and fills onto a frame copy."""
        output = frame.copy()
        overlay = frame.copy()
        for prompt_data in segment_results.get("results", []):
            for mask in prompt_data.get("masks", []):
                for polygon in mask.get("polygons", []):
                    points = np.array(
                        [[int(x * frame.shape[1]), int(y * frame.shape[0])] for x, y in polygon],
                        dtype=np.int32,
                    )
                    cv2.fillPoly(overlay, [points], (200, 60, 255))
                    cv2.polylines(output, [points], True, (255, 255, 255), 1)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        count = segment_results.get("total_segments", 0)
        cv2.putText(
            output,
            f"YOLOv8-seg masks: {count}",
            (14, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 60, 255),
            2,
            cv2.LINE_AA,
        )
        return output


def load_segmenter(prefer: Optional[str] = None) -> Any:
    """
    Returns the best available segmenter.

    ``prefer="sam3"`` forces the slow model (used only for deep-pass comparison).
    Otherwise the fast YOLO segmenter is used, falling back to SAM 3 only if the YOLO
    weights cannot be loaded.
    """
    if prefer == "sam3":
        from src.cv.sam3_masker import SAM3Masker

        return SAM3Masker.get_instance()
    try:
        return YoloSegmenter()
    except Exception as exc:  # noqa: BLE001
        print(f"[segmenter] YOLO segmentation unavailable ({exc}); falling back to SAM 3")
        from src.cv.sam3_masker import SAM3Masker

        return SAM3Masker.get_instance()
