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


# ---------------------------------------------------------------------------
# Device and model selection
# ---------------------------------------------------------------------------

from src.cv.segmenter import (  # noqa: E402
    CPU_WEIGHTS,
    WEIGHT_TIERS,
    device_report,
    pick_weights,
)


def test_device_report_has_a_reason_when_cuda_is_unusable():
    report = device_report()
    assert "cuda_available" in report
    assert report["device"] in {"cpu", "cuda"}
    if not report["cuda_available"]:
        # A bare "False" is not actionable; the reason must say what to fix.
        assert report["reason"], "an unusable GPU must come with an explanation"


def test_device_report_names_a_real_gpu_when_cuda_works():
    report = device_report()
    if report["cuda_available"]:
        assert report["gpu_name"]
        assert report["total_vram_gb"] and report["total_vram_gb"] > 0
        assert report["free_vram_gb"] is not None


def test_cpu_selection_stays_small():
    weights, why = pick_weights({"cuda_available": False, "reason": "no device"})
    assert weights == CPU_WEIGHTS
    assert "CPU" in why


def test_gpu_selection_spends_available_vram():
    """A big GPU must get a bigger model, not the CPU default."""
    small, _ = pick_weights({"cuda_available": True, "free_vram_gb": 4.0, "gpu_name": "X", "total_vram_gb": 6.0})
    large, why = pick_weights({"cuda_available": True, "free_vram_gb": 11.0, "gpu_name": "RTX 4050", "total_vram_gb": 12.0})
    assert small != CPU_WEIGHTS
    assert large != CPU_WEIGHTS
    assert large != small, "more VRAM should select a larger model"
    assert "RTX 4050" in why


def test_gpu_selection_is_monotonic_in_vram():
    picks = []
    for free in (0.5, 1.0, 3.5, 4.0, 7.0, 11.0, 24.0):
        weights, _ = pick_weights(
            {"cuda_available": True, "free_vram_gb": free, "gpu_name": "g", "total_vram_gb": free + 1}
        )
        picks.append(weights)
    # A bigger card must never get a smaller model than a smaller card.
    order = [w for _m, w, _l in WEIGHT_TIERS] + [CPU_WEIGHTS]
    ranks = [order.index(p) if p in order else -1 for p in picks]
    assert all(r >= 0 for r in ranks), ranks
    assert ranks == sorted(ranks, reverse=True), f"selection not monotonic: {picks}"


def test_unknown_free_vram_still_returns_a_gpu_model():
    weights, _ = pick_weights({"cuda_available": True, "free_vram_gb": None, "gpu_name": "g"})
    assert weights != CPU_WEIGHTS


def test_all_ladder_models_are_small_enough_to_be_cheap():
    """
    Every model in the ladder is tens of millions of parameters. Documented here because
    the premise of the ladder is that VRAM capacity is never the constraint.
    """
    ladder = {w for _m, w, _l in WEIGHT_TIERS} | {CPU_WEIGHTS}
    assert "yolov8n-seg.pt" in ladder, "the CPU default must be in the ladder"
    assert "yolo11l-seg.pt" in ladder, "the largest GPU tier must be in the ladder"
    assert all(w.endswith("-seg.pt") for w in ladder)
    assert len(ladder) == len(WEIGHT_TIERS) + 1, "ladder must not repeat a model"


def test_segmenter_honours_an_explicit_device():
    model = YoloSegmenter(weights=CPU_WEIGHTS, device="cpu", half=False)
    assert model.device == "cpu"
    assert model.half is False
    assert model.warmup_ok is True


def test_precision_uses_the_current_api():
    """
    ultralytics 8.4 removed the `half=` predict argument. Passing it looked correct but
    silently ran FP32, so the code must use `quantize` and the two must agree.
    """
    cpu = YoloSegmenter(device="cpu", half=False)
    assert cpu.quantize == "fp32"
    assert cpu.half is False

    # Ask for half precision without a GPU present; the flag must drive the argument.
    requested = YoloSegmenter(device="cpu", half=True)
    assert requested.quantize == "fp16"


def test_no_deprecated_predict_arguments_are_used():
    import inspect

    from src.cv import segmenter as module

    source = inspect.getsource(module.YoloSegmenter.segment)
    assert "half=" not in source, "half= is deprecated and ignored; use quantize="
    assert "quantize=" in source
