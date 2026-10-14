"""
Wayfinder multi-object tracker for fixed-camera stop footage.

Deliberately small and dependency-light (numpy + scipy only) because the point is
to be auditable: it is an IoU / normalized-distance association tracker with a
constant-velocity motion model, not a learned re-identification tracker. Track
identity is only reliable inside a single clip with a fixed camera; that
limitation is reported in the analysis artifact rather than hidden.

Why this exists: passenger arrival rate (pax/min) is the single most important
number in the dispatch decision, and it cannot be derived from a single frame.
It requires associating detections across time and observing where people are and
how fast they are moving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

Box = Tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


def box_iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def box_diag(b: Box) -> float:
    return float(np.hypot(b[2] - b[0], b[3] - b[1]))


def _center(b: Box) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0], dtype=np.float64)


@dataclass
class Track:
    """A single tracked person."""

    track_id: int
    box: Box
    first_frame: int
    last_frame: int
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    # Per-frame history of (frame_index, center_x, center_y, box) while observed.
    history: List[Tuple[int, float, float, Box]] = field(default_factory=list)

    @property
    def center(self) -> np.ndarray:
        return _center(self.box)

    def speed_px_s(self, fps: float, window_frames: int = 0) -> float:
        """
        Smoothed speed in px/s over the last ``window_frames`` observations
        (0 = use the tracker's internal velocity estimate).
        """
        if window_frames <= 0 or len(self.history) < 2:
            return float(np.hypot(*self.velocity) * fps)

        recent = self.history[-window_frames:]
        if len(recent) < 2:
            return 0.0
        (f0, x0, y0, _), (f1, x1, y1, _) = recent[0], recent[-1]
        dt = (f1 - f0) / fps if fps > 0 else 0.0
        if dt <= 0:
            return 0.0
        return float(np.hypot(x1 - x0, y1 - y0) / dt)

    def box_at(self, frame: int) -> Box:
        """Box extrapolated to ``frame`` using the constant-velocity model."""
        dt = (frame - self.last_frame) / max(1e-9, self._fps)
        cx, cy = self.center + self.velocity * dt
        w = self.box[2] - self.box[0]
        h = self.box[3] - self.box[1]
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    _fps: float = 25.0


class IoUTracker:
    """
    Simple, tunable multi-object tracker.

    Association cost blends IoU with centre distance normalised by box size, which
    handles small/fast pedestrians better than pure IoU while still refusing to
    match across the frame.
    """

    def __init__(
        self,
        fps: float = 25.0,
        iou_threshold: float = 0.25,
        max_center_dist_factor: float = 1.5,
        max_misses: int = 12,
        min_hits: int = 3,
        velocity_smoothing: float = 0.6,
    ):
        self.fps = fps
        self.iou_threshold = iou_threshold
        self.max_center_dist_factor = max_center_dist_factor
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.velocity_smoothing = velocity_smoothing

        self.tracks: List[Track] = []
        self._next_id = 1

    def _cost(self, track: Track, box: Box, frame: int) -> Optional[float]:
        """Returns an association cost in [0, 2], or None if the pair is implausible."""
        pred = track.box_at(frame)
        iou = box_iou(pred, box)
        norm_dist = float(np.linalg.norm(_center(pred) - _center(box)) / max(1e-6, box_diag(pred)))
        if iou < self.iou_threshold and norm_dist > self.max_center_dist_factor:
            return None
        return (1.0 - iou) + 0.5 * norm_dist

    def update(self, boxes: Sequence[Box], frame: int) -> List[Track]:
        """Associates ``boxes`` (one frame) with existing tracks. Returns live tracks."""
        boxes = [tuple(float(v) for v in b) for b in boxes]

        matches: Dict[int, int] = {}
        if self.tracks and boxes:
            costs = np.full((len(self.tracks), len(boxes)), 1e6, dtype=np.float64)
            for ti, tr in enumerate(self.tracks):
                for di, b in enumerate(boxes):
                    c = self._cost(tr, b, frame)
                    if c is not None:
                        costs[ti, di] = c
            rows, cols = linear_sum_assignment(costs)
            for r, c in zip(rows, cols):
                if costs[r, c] < 1e5:
                    matches[int(r)] = int(c)

        # Update matched tracks.
        for ti, di in matches.items():
            tr = self.tracks[ti]
            new_box = boxes[di]
            new_center = _center(new_box)
            if tr.misses == 0:
                inst_v = new_center - tr.center
                tr.velocity = (
                    self.velocity_smoothing * tr.velocity
                    + (1.0 - self.velocity_smoothing) * inst_v
                )
            else:
                # After an occlusion gap, trust the new measurement outright.
                tr.velocity = new_center - tr.center
            tr.box = new_box
            tr.last_frame = frame
            tr.hits += 1
            tr.misses = 0
            tr._fps = self.fps
            tr.history.append((frame, float(new_center[0]), float(new_center[1]), new_box))
            if tr.hits >= self.min_hits:
                tr.confirmed = True

        # Age unmatched tracks.
        matched_indices = set(matches.keys())
        for ti, tr in enumerate(self.tracks):
            if ti not in matched_indices:
                tr.misses += 1

        # Spawn new tentative tracks for unmatched detections.
        used = set(matches.values())
        for di, b in enumerate(boxes):
            if di in used:
                continue
            tr = Track(track_id=self._next_id, box=b, first_frame=frame, last_frame=frame)
            tr._fps = self.fps
            c = _center(b)
            tr.history.append((frame, float(c[0]), float(c[1]), b))
            self.tracks.append(tr)
            self._next_id += 1

        # Retire stale tracks.
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return [t for t in self.tracks if t.confirmed]

    def finalize(self) -> List[Track]:
        """Returns all confirmed tracks ever created, sorted by id."""
        return sorted((t for t in self.tracks if t.confirmed), key=lambda t: t.track_id)
