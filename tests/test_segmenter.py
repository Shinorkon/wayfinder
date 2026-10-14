"""
Segmentation model tests.

The live dashboard's masks and boxes come from one forward pass of a YOLO segmentation
model. These tests pin the interface the rest of the system depends on, since the model
is swappable (SAM 3 is still available as a slow deep pass).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.cv.segmenter import LiveSegmentation, YoloSegmenter, _polygons_from_mask  # noqa: E402


@pytest.fixture(scope="module")
def segmenter():
    return YoloSegmenter()


def test_loads_fast_model_and_names_itself(segmenter):
    assert segmenter.name.startswith("yolov8n-seg")
    assert segmenter.is_sam3 is False
    assert segmenter.device_label in {"cpu", "cuda", "unknown"}


def test_returns_boxes_and_masks_from_one_pass(segmenter):
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    result = segmenter.segment(frame)
    assert isinstance(result, LiveSegmentation)
    assert result.error is None
    assert result.model == segmenter.name
    assert result.person_count == 0
    assert result.masks == []


def test_mask_polygons_are_normalised_and_bounded():
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[20:60, 40:120] = 1
    polygons = _polygons_from_mask(mask, min_area_frac=0.001)
    assert polygons, "expected at least one polygon"
    for polygon in polygons:
        for x, y in polygon:
            assert 0.0 <= x <= 1.0, x
            assert 0.0 <= y <= 1.0, y


def test_tiny_specks_are_discarded():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0:2, 0:2] = 1  # 4 px out of 10000
    assert _polygons_from_mask(mask, min_area_frac=0.001) == []


def test_empty_mask_yields_no_polygons():
    assert _polygons_from_mask(np.zeros((50, 50), dtype=np.uint8), 0.001) == []


def test_to_dict_shape_is_stable(segmenter):
    result = LiveSegmentation(model="test", device="cpu", latency_ms=12.5, person_count=0)
    payload = result.to_dict(640, 480)
    assert payload["model"] == "test"
    assert payload["frame_size"] == [640, 480]
    assert payload["latency_ms"] == 12.5
    assert "masks" in payload


def test_sam3_interface_shim_is_present(segmenter):
    """The offline renderer treats either model identically."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    envelope = segmenter.segment_frame(frame)
    assert set(envelope) >= {"image_size", "results", "total_segments"}
    assert envelope["image_size"] == [640, 480]
    assert envelope["results"][0]["prompt"] == "person"
    assert segmenter.mask_polygons(envelope) == []


def test_deep_pass_loader_prefers_fast_model():
    from src.cv.segmenter import load_segmenter

    model = load_segmenter()
    assert model is not None
    # Default must never be SAM 3: it cannot run per frame.
    assert getattr(model, "is_sam3", False) is False
