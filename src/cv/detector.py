"""
Wayfinder computer vision: YOLOv8 person detection for stop-level footage.

Scope, stated plainly: this module detects people in a single frame. It does not
track, and it does not count a queue on its own - a frame gives you one snapshot of
who is in view, not how many people are waiting or how fast they are arriving.
Temporal metrics live in src/cv/tracker.py and src/cv/stop_analysis.py.
"""

from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np
from ultralytics import YOLO


class PersonDetector:
    """YOLOv8 person detector for stationary / high-angle stop cameras."""

    def __init__(self, model_weight: str = "yolov8n.pt", conf_thresh: float = 0.30):
        self.conf_thresh = conf_thresh
        self.model = YOLO(model_weight)
        self.target_class_id = 0  # COCO class 0 = person

    def detect_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Runs person detection on one BGR frame. Returns boxes, confidences, count."""
        h, w = frame.shape[:2]
        results = self.model.predict(
            source=frame,
            classes=[self.target_class_id],
            conf=self.conf_thresh,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for i, (box, conf) in enumerate(zip(boxes, confs)):
                x1, y1, x2, y2 = (float(v) for v in box)
                detections.append(
                    {
                        "id": i + 1,
                        "box": [x1, y1, x2, y2],
                        "norm_box": [x1 / w, y1 / h, x2 / w, y2 / h],
                        "confidence": round(float(conf), 3),
                        "center": [(x1 + x2) / 2, (y1 + y2) / 2],
                    }
                )

        return {
            "count": len(detections),
            "detections": detections,
            "width": w,
            "height": h,
        }

    def render_overlay(
        self,
        frame: np.ndarray,
        detection_result: Dict[str, Any],
        show_labels: bool = True,
        label: str = "PERSON DETECTIONS",
    ) -> np.ndarray:
        """
        Draws bounding boxes and a count badge.

        The badge deliberately says "detections", not "queue", because a single frame
        cannot tell you how many people are waiting. Formatting is scale-aware so the
        annotation stays legible on 440p footage as well as 1080p.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        count = detection_result["count"]
        scale = max(0.5, min(2.0, w / 1280.0))
        box_color = (255, 229, 0) if count < 25 else (0, 68, 239)
        thickness = max(1, int(round(2 * scale)))

        for det in detection_result["detections"]:
            x1, y1, x2, y2 = (int(v) for v in det["box"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)

            line_len = max(6, int((x2 - x1) * 0.2))
            cv2.line(annotated, (x1, y1), (x1 + line_len, y1), (255, 255, 255), thickness)
            cv2.line(annotated, (x1, y1), (x1, y1 + line_len), (255, 255, 255), thickness)
            cv2.line(annotated, (x2, y2), (x2 - line_len, y2), (255, 255, 255), thickness)
            cv2.line(annotated, (x2, y2), (x2, y2 - line_len), (255, 255, 255), thickness)

            if show_labels and (y2 - y1) > 40 * scale:
                lbl = f"{int(det['confidence'] * 100)}%"
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale, 1)
                cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), (15, 20, 32), -1)
                cv2.putText(
                    annotated, lbl, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale, (255, 255, 255), 1, cv2.LINE_AA,
                )

        badge_text = f"{label}: {count}"
        font_scale = 0.55 * scale
        pad = int(round(10 * scale))
        (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(annotated, (pad, pad), (pad + bw + 20, pad + bh + 18), (10, 13, 20), -1)
        cv2.rectangle(annotated, (pad, pad), (pad + bw + 20, pad + bh + 18), box_color, max(1, thickness - 1))
        cv2.putText(
            annotated, badge_text, (pad + 10, pad + bh + 10),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, thickness, cv2.LINE_AA,
        )
        return annotated
