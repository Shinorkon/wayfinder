"""
Wayfinder continuous monitoring.

This is the part the project previously claimed to have and did not: a long-running
process that watches a stop's camera feed and keeps the dashboard's numbers current
by itself, rather than producing one cached snapshot per clip.

Two workers share one stop's feed:

  detector worker (fast, always on)
      Reads the feed continuously, runs YOLOv8 detection on every sampled frame and
      feeds a tracker. Emits the live state: how many people are in frame, how many
      are in the waiting zone, how many are actually waiting, and a rolling arrival
      rate measured from real zone-entry events. Runs at ``fps`` samples/second;
      measured cost is ~26-51 ms per detection on CPU, so 2 fps is affordable.

  segmentation worker (slow, always on)
      SAM 3 is a promptable segmentation model: measured at ~13.5 s per 1008x1008
      frame on this machine's CPU, and it rejects any other input resolution (the
      rotary position embeddings are built for 1008). It therefore cannot run at frame
      rate, and nothing in this project pretends otherwise. Instead it runs in a tight
      loop of its own: segment a frame, publish it, wait a short interval, segment the
      next. The dashboard shows the newest segmentation with its real age and cadence.

Both run in daemon threads and publish into a lock-protected state snapshot. Nothing
here invents numbers: every value is either counted from detections or measured from
wall-clock timings reported alongside it.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.cv.stop_analysis import (
    DEFAULTS as ZONE_DEFAULTS,
    assess_zone,
    summarise_live,
    zone_bounds,
)
from src.cv.tracker import IoUTracker

# Arrival-rate window. Long enough to smooth per-frame noise, short enough that the
# dashboard reacts while you are watching it.
RATE_WINDOW_S = 60.0


@dataclass
class LiveSnapshot:
    """One consistent view of a monitored stop."""

    stop_id: str
    clip: str
    running: bool
    frame_index: int
    video_time_s: float
    clip_duration_s: float
    people_in_frame: int
    people_in_waiting_zone: int
    people_waiting: int
    arrival_rate_pax_per_min: float
    arrival_events: int
    rate_window_s: float
    mean_wait_dwell_s: float
    detector_fps_actual: float
    detector_ms: float
    detector_device: str
    tracks: List[Dict[str, Any]]
    zone_bounds_frac: Optional[Dict[str, float]]
    frame_size: Tuple[int, int]
    segmentation: Dict[str, Any]
    updated_at: float
    warnings: List[str] = field(default_factory=list)


class StopMonitor:
    """
    Continuously monitors one clip. Safe to start/stop repeatedly.

    Frame pacing: the loop never sleeps longer than needed to hit its target sample
    rate, and if detection is slower than the target it simply runs slower and reports
    the achieved rate instead of claiming the target.
    """

    def __init__(
        self,
        stop_id: str,
        clip_path: str,
        detector,
        zone_mask: Optional[np.ndarray] = None,
        params: Optional[Dict[str, Any]] = None,
        fps: float = 5.0,
        segmenter=None,
        deep_pass_model=None,
        deep_pass_interval_s: float = 0.0,
        loop_clip: bool = True,
    ):
        self.stop_id = stop_id
        self.clip_path = clip_path
        self.detector = detector
        self.zone_mask = zone_mask
        self.params = dict(ZONE_DEFAULTS)
        if params:
            self.params.update(params)
        self.target_fps = float(fps)
        # A YOLO segmentation model produces boxes and masks in ONE forward pass, so the
        # live view needs a single worker: no second slow loop, no box/mask desync.
        self.segmenter = segmenter
        # Optional slow model, run rarely, purely to compare against the fast one.
        self.deep_pass_model = deep_pass_model
        self.deep_pass_interval_s = float(deep_pass_interval_s)
        self.loop_clip = loop_clip

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._snapshot: Optional[LiveSnapshot] = None

        # Rolling arrival events (video seconds when a track first entered the zone).
        self._arrivals: Deque[float] = deque()
        self._seg_state: Dict[str, Any] = {
            "enabled": segmenter is not None,
            "status": "idle" if segmenter is not None else "disabled",
            "model": getattr(segmenter, "name", None),
            "device": getattr(segmenter, "device_label", None),
            "last_run_at": None,
            "last_duration_s": None,
            "runs": 0,
            "masks": None,
            "note": (
                "Masks are produced in the same forward pass as the detection boxes, so "
                "they are as fresh as the box overlay."
                if segmenter is not None
                else "No segmentation model loaded; boxes only."
            ),
        }
        self._deep_state: Dict[str, Any] = {
            "enabled": deep_pass_model is not None and self.deep_pass_interval_s > 0,
            "model": getattr(deep_pass_model, "name", "sam3") if deep_pass_model else None,
            "interval_s": self.deep_pass_interval_s,
            "runs": 0,
            "last_run_at": None,
            "last_duration_s": None,
            "note": (
                "Slow deep pass kept for comparison only; it cannot run per frame."
                if deep_pass_model is not None and self.deep_pass_interval_s > 0
                else "Deep pass disabled."
            ),
        }

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._threads:
            return
        self._stop.clear()
        worker = threading.Thread(target=self._monitor_loop, name=f"monitor-{self.stop_id}", daemon=True)
        worker.start()
        self._threads.append(worker)
        if self._deep_state["enabled"]:
            deep = threading.Thread(target=self._deep_pass_loop, name=f"deep-{self.stop_id}", daemon=True)
            deep.start()
            self._threads.append(deep)

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=5.0)
        self._threads = []
        with self._lock:
            if self._snapshot:
                self._snapshot.running = False

    @property
    def running(self) -> bool:
        return bool(self._threads) and not self._stop.is_set()

    def snapshot(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._snapshot is None:
                return None
            snap = self._snapshot
            return {
                "stop_id": snap.stop_id,
                "clip": snap.clip,
                "running": snap.running and not self._stop.is_set(),
                "frame_index": snap.frame_index,
                "video_time_s": round(snap.video_time_s, 2),
                "clip_duration_s": round(snap.clip_duration_s, 2),
                "people_in_frame": snap.people_in_frame,
                "people_in_waiting_zone": snap.people_in_waiting_zone,
                "people_waiting": snap.people_waiting,
                "arrival_rate_pax_per_min": snap.arrival_rate_pax_per_min,
                "arrival_events": snap.arrival_events,
                "arrival_rate_definition": (
                    f"distinct tracks first entering the waiting zone in the last "
                    f"{int(snap.rate_window_s)}s of footage, scaled to pax/min"
                ),
                "mean_wait_dwell_s": snap.mean_wait_dwell_s,
                "performance": {
                    "detector_fps_actual": snap.detector_fps_actual,
                    "detector_fps_target": self.target_fps,
                    "detector_ms_per_frame": snap.detector_ms,
                    "detector_device": snap.detector_device,
                },
                "tracks": snap.tracks,
                "waiting_zone": {"bounds_frac": snap.zone_bounds_frac},
                "frame_size": list(snap.frame_size),
                "segmentation": dict(self._seg_state, live_overlay=self._seg_mask_overlay),
                "deep_pass": dict(self._deep_state),
                "updated_at": snap.updated_at,
                "age_s": round(time.time() - snap.updated_at, 2),
                "warnings": snap.warnings,
            }

    # -- detection worker --------------------------------------------------

    def _monitor_loop(self) -> None:
        cap = cv2.VideoCapture(self.clip_path)
        if not cap.isOpened():
            with self._lock:
                self._snapshot = None
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if fps else 0.0

        tracker_fps = self.target_fps
        tracker = IoUTracker(fps=tracker_fps, min_hits=2, max_misses=8)
        # With 2 fps sampling, the tracker sees the same face at a lower rate, so the
        # speed window is expressed in samples rather than frames.
        params = dict(self.params)
        params["speed_window_frames"] = max(2, int(round(self.params["speed_window_frames"] * self.target_fps / 25.0)))

        interval = 1.0 / self.target_fps
        sample_stride = max(1, int(round(fps / self.target_fps)))
        seen_ids: Dict[int, float] = {}   # track_id -> first zone-entry time (video seconds)
        frame_index = 0
        ema_ms = 0.0
        ema_seg_ms = 0.0
        ema_fps = 0.0
        last_report = time.time()
        reports = 0

        while not self._stop.is_set():
            cycle_start = time.time()
            ok = cap.grab()
            if not ok:
                if not self.loop_clip:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_index = 0
                seen_ids.clear()
                self._arrivals.clear()
                tracker = IoUTracker(fps=tracker_fps, min_hits=2, max_misses=8)
                continue

            if frame_index % sample_stride != 0:
                frame_index += 1
                continue

            ok, frame = cap.retrieve()
            if not ok:
                frame_index += 1
                continue

            video_time = frame_index / fps
            t0 = time.time()
            if self.segmenter is not None:
                # Segmentation model: boxes and masks from the same forward pass, so the
                # masks the user sees belong to the exact frame the counts come from.
                seg = self.segmenter.segment(frame)
                detect_ms = seg.latency_ms
                boxes = [
                    (
                        m["box"][0] * frame_w, m["box"][1] * frame_h,
                        m["box"][2] * frame_w, m["box"][3] * frame_h,
                    )
                    for m in seg.masks if m.get("box")
                ]
            else:
                result = self.detector.detect_frame(frame)
                detect_ms = (time.time() - t0) * 1000.0
                seg = None
                boxes = [tuple(d["box"]) for d in result["detections"]]
            ema_ms = detect_ms if ema_ms == 0 else 0.85 * ema_ms + 0.15 * detect_ms
            ema_seg_ms = detect_ms if ema_seg_ms == 0 else 0.8 * ema_seg_ms + 0.2 * detect_ms
            tracks = tracker.update(boxes, frame_index // sample_stride)
            records = assess_zone(
                self.zone_mask, tracks, params,
                fps=tracker_fps, frame_w=frame_w, frame_h=frame_h, stride=1,
            )
            summary = summarise_live(records, self.zone_mask, frame_w, frame_h)

            # Arrival events: a track is counted once, at the first sample where it is
            # inside the waiting zone. This is measured, not inferred.
            now_ids = {r["track_id"] for r in records if r["in_zone"]}
            for rec in records:
                if rec["in_zone"] and rec["track_id"] not in seen_ids:
                    seen_ids[rec["track_id"]] = video_time
                    self._arrivals.append(video_time)
            for tid in list(seen_ids):
                if tid not in now_ids and len(seen_ids) > 400:
                    # Keep the bookkeeping bounded on long runs.
                    seen_ids.pop(tid, None)

            # Roll the arrival window in footage time (loops make wall-clock wrong).
            while self._arrivals and (video_time - self._arrivals[0]) > RATE_WINDOW_S:
                self._arrivals.popleft()
            window_span = min(RATE_WINDOW_S, max(1.0, video_time if not self.loop_clip else RATE_WINDOW_S))
            rate = (len(self._arrivals) / window_span) * 60.0 if window_span > 0 else 0.0

            reports += 1
            elapsed = time.time() - last_report
            if elapsed >= 1.0:
                ema_fps = reports / elapsed
                reports = 0
                last_report = time.time()

            if seg is not None:
                with self._lock:
                    self._seg_state.update(
                        {
                            "status": "ok" if not seg.error else "error",
                            "error": seg.error,
                            "last_run_at": time.time(),
                            "last_duration_s": round(seg.latency_ms / 1000.0, 3),
                            "runs": self._seg_state.get("runs", 0) + 1,
                            "masks": len(seg.masks),
                            "person_count": seg.person_count,
                        }
                    )
                    self._seg_mask_overlay = {
                        "frame_size": [frame_w, frame_h],
                        "prompt": "person",
                        "masks": [
                            {"prompt": "person", "polygons": m["polygons"], "score": m.get("score")}
                            for m in seg.masks
                        ],
                    }

            with self._lock:
                self._snapshot = LiveSnapshot(
                    stop_id=self.stop_id,
                    clip=os.path.basename(self.clip_path),
                    running=True,
                    frame_index=frame_index,
                    video_time_s=video_time,
                    clip_duration_s=duration,
                    people_in_frame=summary["people_in_frame"],
                    people_in_waiting_zone=summary["people_in_waiting_zone"],
                    people_waiting=summary["people_waiting"],
                    arrival_rate_pax_per_min=round(rate, 2),
                    arrival_events=len(self._arrivals),
                    rate_window_s=RATE_WINDOW_S,
                    mean_wait_dwell_s=summary["mean_wait_dwell_s"],
                    detector_fps_actual=round(ema_fps, 2),
                    detector_ms=round(ema_ms, 1),
                    detector_device=getattr(self.detector, "device_label", "cpu"),
                    tracks=summary["tracks"],
                    zone_bounds_frac=zone_bounds(self.zone_mask),
                    frame_size=(frame_w, frame_h),
                    segmentation={},
                    updated_at=time.time(),
                )

            frame_index += 1
            sleep_for = interval - (time.time() - cycle_start)
            if sleep_for > 0:
                self._stop.wait(sleep_for)

        cap.release()

    # -- segmentation worker ----------------------------------------------

    def _deep_pass_loop(self) -> None:
        cap = cv2.VideoCapture(self.clip_path)
        if not cap.isOpened():
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        # Walk through the clip so successive segmentations show different moments.
        cursor = 0

        while not self._stop.is_set():
            cap.set(cv2.CAP_PROP_POS_FRAMES, cursor)
            ok, frame = cap.read()
            if not ok:
                cursor = 0
                continue

            started = time.time()
            try:
                seg = self.deep_pass_model.segment_frame(
                    frame, text_prompts=["waiting passenger"], conf_threshold=0.45
                )
                duration = time.time() - started
                with self._lock:
                    self._deep_state.update(
                        {
                            "status": "ok",
                            "last_run_at": time.time(),
                            "last_duration_s": round(duration, 2),
                            "runs": self._deep_state.get("runs", 0) + 1,
                            "masks": seg.get("total_segments", 0),
                            "frame_time_s": round(cursor / fps, 2),
                        }
                    )
            except Exception as exc:  # keep monitoring even if one pass fails
                with self._lock:
                    self._deep_state.update(
                        {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                    )
                duration = time.time() - started

            # Advance a few percent of the clip, then pause briefly.
            cursor = (cursor + max(1, int(0.05 * total))) % max(1, total)
            self._stop.wait(self.deep_pass_interval_s)

        cap.release()


class MonitorHub:
    """Owns the monitors for every stop and guarantees only one runs at a time."""

    def __init__(self, detector_factory, segmenter_factory=None, deep_pass_factory=None):
        self.detector_factory = detector_factory
        self.segmenter_factory = segmenter_factory
        self.deep_pass_factory = deep_pass_factory
        self.monitors: Dict[str, StopMonitor] = {}
        self._lock = threading.Lock()
        self._detector = None
        self._segmenter = None
        self._segmenter_attempted = False
        self._deep_pass = None
        self._deep_pass_attempted = False

    # Lazily created so the API can start without loading models.
    def detector(self):
        with self._lock:
            if self._detector is None:
                self._detector = self.detector_factory()
            return self._detector

    def segmenter(self):
        """The fast per-frame model (YOLO segmentation). Loaded on first use."""
        with self._lock:
            if self._segmenter is None and not self._segmenter_attempted:
                self._segmenter_attempted = True
                if self.segmenter_factory is not None:
                    try:
                        self._segmenter = self.segmenter_factory()
                    except Exception as exc:  # noqa: BLE001 - reported to the client
                        print(f"[monitor] segmentation unavailable: {type(exc).__name__}: {exc}")
                        self._segmenter = None
            return self._segmenter

    def deep_pass(self):
        """The slow optional model (SAM 3), only used when explicitly enabled."""
        with self._lock:
            if self._deep_pass is None and not self._deep_pass_attempted:
                self._deep_pass_attempted = True
                if self.deep_pass_factory is not None:
                    try:
                        self._deep_pass = self.deep_pass_factory()
                    except Exception as exc:  # noqa: BLE001 - reported to the client
                        print(f"[monitor] deep pass unavailable: {type(exc).__name__}: {exc}")
                        self._deep_pass = None
            return self._deep_pass

    def start(
        self,
        stop_id: str,
        clip_path: str,
        zone_mask: Optional[np.ndarray],
        fps: float = 5.0,
        with_segmentation: bool = True,
        deep_pass_interval_s: float = 0.0,
        params: Optional[Dict[str, Any]] = None,
    ) -> StopMonitor:
        # Load models BEFORE taking the lock: these accessors acquire the same
        # non-reentrant lock, so calling them while holding it deadlocks the request.
        detector = None if with_segmentation else self.detector()
        segmenter = self.segmenter() if with_segmentation else None
        if with_segmentation and segmenter is None:
            detector = self.detector()
        deep = self.deep_pass() if deep_pass_interval_s > 0 else None

        with self._lock:
            current = self.monitors.get(stop_id)
            if current and current.running:
                return current
            monitor = StopMonitor(
                stop_id=stop_id,
                clip_path=clip_path,
                detector=detector,
                zone_mask=zone_mask,
                params=params,
                fps=fps,
                segmenter=segmenter,
                deep_pass_model=deep,
                deep_pass_interval_s=deep_pass_interval_s,
            )
            self.monitors[stop_id] = monitor
        monitor.start()
        return monitor

    def stop(self, stop_id: str) -> bool:
        with self._lock:
            monitor = self.monitors.get(stop_id)
        if not monitor:
            return False
        monitor.stop()
        return True

    def stop_all(self) -> None:
        with self._lock:
            monitors = list(self.monitors.values())
        for monitor in monitors:
            monitor.stop()

    def snapshot(self, stop_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            monitor = self.monitors.get(stop_id)
        return monitor.snapshot() if monitor else None

    def active(self) -> List[str]:
        with self._lock:
            return [sid for sid, m in self.monitors.items() if m.running]
