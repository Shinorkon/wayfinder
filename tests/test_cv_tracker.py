"""
Computer vision unit tests.

These cover the parts that can be tested without footage: the tracker's association
behaviour and the measurement helpers. They guard the two failure modes that produced
bad numbers in practice - fragmenting a stationary person into multiple arrivals, and
counting a passing pedestrian as a queue member.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.cv.stop_analysis import DEFAULTS, _assess_confidence, _in_zone  # noqa: E402
from src.cv.tracker import IoUTracker, box_iou  # noqa: E402


def test_box_iou_basics():
    a = (0.0, 0.0, 10.0, 10.0)
    assert box_iou(a, a) == pytest.approx(1.0)
    assert box_iou(a, (20.0, 20.0, 30.0, 30.0)) == 0.0
    assert box_iou(a, (5.0, 0.0, 15.0, 10.0)) == pytest.approx(50.0 / 150.0)


def test_tracker_keeps_one_identity_for_a_jittering_person():
    """A stationary person with box jitter must not be counted as repeated arrivals."""
    tracker = IoUTracker(fps=5.0, min_hits=3)
    rng = np.random.default_rng(7)
    for step in range(40):
        jitter = rng.normal(0, 1.5, size=4)
        box = (100 + jitter[0], 200 + jitter[1], 140 + jitter[2], 320 + jitter[3])
        tracker.update([tuple(box)], step)

    confirmed = tracker.finalize()
    assert len(confirmed) == 1, f"expected a single track, got {len(confirmed)}"
    assert confirmed[0].hits >= 30


def test_tracker_separates_two_people_walking_apart():
    tracker = IoUTracker(fps=5.0, min_hits=2)
    for step in range(20):
        left = (100 - step * 2, 200, 140 - step * 2, 300)
        right = (400 + step * 2, 200, 440 + step * 2, 300)
        tracker.update([left, right], step)
    assert len(tracker.finalize()) == 2


def test_tracker_survives_a_short_occlusion():
    """A person hidden for a few samples then reappearing keeps their identity."""
    tracker = IoUTracker(fps=5.0, min_hits=2, max_misses=8)
    for step in range(10):
        tracker.update([(100.0, 200.0, 140.0, 300.0)], step)
    for step in range(10, 15):  # occlusion
        tracker.update([], step)
    for step in range(15, 25):
        tracker.update([(102.0, 201.0, 142.0, 301.0)], step)

    tracks = tracker.finalize()
    assert len(tracks) == 1, f"occlusion created {len(tracks)} identities instead of 1"


def test_zone_lookup_is_bounded_and_correct():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 3:6] = True
    # Centre of the mask, in pixel space of a 100x100 frame.
    assert _in_zone(mask, 40.0, 30.0, 100, 100) is True
    assert _in_zone(mask, 90.0, 90.0, 100, 100) is False
    # Out-of-range coordinates must clamp, not raise.
    assert _in_zone(mask, -5.0, 500.0, 100, 100) is False
    assert _in_zone(None, 10.0, 10.0, 100, 100) is False


def test_confidence_gate_rejects_short_clips():
    zone = {"fits_area_bounds": True}
    grade, reasons = _assess_confidence(14.8, waiting_tracks=10, mean_dwell_s=8.0, zone=zone)
    assert grade == "low"
    assert any("45s" in r for r in reasons)


def test_confidence_gate_rejects_thin_samples():
    zone = {"fits_area_bounds": True}
    grade, reasons = _assess_confidence(200.0, waiting_tracks=3, mean_dwell_s=9.0, zone=zone)
    assert grade == "low"
    assert any("waiting tracks" in r for r in reasons)


def test_confidence_gate_rejects_brief_dwell():
    zone = {"fits_area_bounds": True}
    grade, reasons = _assess_confidence(120.0, waiting_tracks=30, mean_dwell_s=1.4, zone=zone)
    assert grade == "low"
    assert any("mean dwell" in r for r in reasons)


def test_confidence_gate_accepts_long_dense_clip():
    zone = {"fits_area_bounds": True}
    grade, _ = _assess_confidence(600.0, waiting_tracks=60, mean_dwell_s=45.0, zone=zone)
    assert grade == "high"


def test_default_parameters_are_self_consistent():
    assert DEFAULTS["sample_stride"] >= 1
    assert DEFAULTS["min_hits"] <= DEFAULTS["max_misses"]
    assert DEFAULTS["stationary_enter_norm"] < DEFAULTS["stationary_exit_norm"], (
        "hysteresis band must be two-sided, with a lower enter threshold than exit"
    )
    assert 0 < DEFAULTS["zone_min_area_frac"] < DEFAULTS["zone_max_area_frac"] < 1
