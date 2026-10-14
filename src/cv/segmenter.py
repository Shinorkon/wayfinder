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

import os
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
        device: Optional[str] = None,
        half: Optional[bool] = None,
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
        self.device_note: Optional[str] = None

        # Resolve the accelerator explicitly, once, so every predict() call targets the
        # same device and the choice can be reported rather than guessed.
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:  # pragma: no cover
                device = "cpu"
        self.device = device
        # FP16 halves memory traffic on tensor cores and is safe for inference; it is not
        # used on CPU, where it is slower and unsupported for many ops.
        # NOTE: ultralytics removed the `half=` predict argument in 8.4; it is silently
        # ignored now, and precision is selected via `quantize`. Passing `half` would look
        # correct while quietly running FP32 on the GPU.
        self.half = (device == "cuda") if half is None else half
        self.quantize = "fp16" if self.half else "fp32"

        if device == "cuda":
            try:
                self.model.to("cuda")
            except Exception as exc:  # pragma: no cover - driver/VRAM issues
                print(f"[segmenter] could not move model to CUDA ({exc}); using CPU")
                self.device = "cpu"
                self.half = False

        # Warm up so the first real frame does not carry the CUDA/cuDNN init cost.
        try:
            import numpy as _np

            self.model.predict(
                source=_np.zeros((self.imgsz, self.imgsz, 3), dtype=_np.uint8),
                verbose=False,
                device=self.device,
                quantize=self.quantize,
            )
            self.warmup_ok = True
        except Exception:  # pragma: no cover
            self.warmup_ok = False

    @property
    def device_label(self) -> str:
        return self.device

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
                device=self.device,
                quantize=self.quantize,
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


# ---------------------------------------------------------------------------
# Device and model selection
# ---------------------------------------------------------------------------

# Weights ladder. Every model here is tens of millions of parameters, i.e. a rounding
# error against a modern laptop GPU's memory; the constraint is latency, not capacity.
# Timings are the CPU measurements from this repository, used only to pick a CPU default.
WEIGHT_TIERS: List[Tuple[float, str, str]] = [
    # (minimum free VRAM in GB, weights, human description for the UI)
    (10.0, "yolo11l-seg.pt", "yolo11l-seg (27.7M params)"),
    (6.0, "yolo11m-seg.pt", "yolo11m-seg (22.4M params)"),
    (3.0, "yolo11s-seg.pt", "yolo11s-seg (10.1M params)"),
    (0.0, "yolov8s-seg.pt", "yolov8s-seg (11.8M params)"),
]
CPU_WEIGHTS = "yolov8n-seg.pt"


def device_report() -> Dict[str, Any]:
    """
    Describes the accelerator situation, including *why* the GPU is unusable when it is.

    A machine can have a perfectly healthy NVIDIA driver and still be unusable from inside
    a container that was not given the device nodes. Reporting that distinction matters:
    "no GPU" and "GPU present but not passed through" call for different fixes.
    """
    report: Dict[str, Any] = {
        "torch": None,
        "cuda_available": False,
        "device": "cpu",
        "gpu_name": None,
        "total_vram_gb": None,
        "free_vram_gb": None,
        "cuda_devices": 0,
        "torch_cuda_build": None,
        "reason": None,
        # Whether the host has an NVIDIA GPU at all, independent of whether this process
        # can use it. The distinction is what makes the failure actionable.
        "gpu_present": False,
        "gpu_name_from_host": None,
    }
    report["gpu_present"], report["gpu_name_from_host"] = _host_gpu()
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        report["reason"] = f"torch unavailable: {exc}"
        return report

    report["torch"] = torch.__version__
    report["torch_cuda_build"] = torch.version.cuda

    if not torch.cuda.is_available():
        report["reason"] = _explain_no_cuda(torch)
        return report

    try:
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        free, total = torch.cuda.mem_get_info(index)
        report.update(
            {
                "cuda_available": True,
                "device": "cuda",
                "cuda_devices": torch.cuda.device_count(),
                "gpu_name": props.name,
                "total_vram_gb": round(total / 1024**3, 2),
                "free_vram_gb": round(free / 1024**3, 2),
            }
        )
    except Exception as exc:  # pragma: no cover
        report["reason"] = f"CUDA reported available but querying it failed: {exc}"
    return report


def _host_gpu() -> Tuple[bool, Optional[str]]:
    """
    Detects an NVIDIA GPU on the host even when CUDA is unusable from here, by reading
    the driver's proc interface (which is often visible inside containers that were not
    given the device nodes).
    """
    base = "/proc/driver/nvidia/gpus"
    try:
        entries = os.listdir(base)
    except OSError:
        return False, None
    if not entries:
        return False, None
    name = None
    try:
        with open(os.path.join(base, entries[0], "information"), encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model:"):
                    name = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return True, name


def _explain_no_cuda(torch_module) -> str:
    """Best-effort diagnosis of an unusable GPU, without pretending to be certain."""
    build = getattr(torch_module.version, "cuda", None)
    if not build:
        return "this torch build has no CUDA support (CPU-only wheel)"

    gpu_present, _name = _host_gpu()
    nodes = [
        n for n in ("nvidia0", "nvidiactl", "nvidia-uvm") if os.path.exists(f"/dev/{n}")
    ]
    if gpu_present and not nodes:
        return (
            "an NVIDIA GPU is present and the driver is loaded, but no /dev/nvidia* device "
            "nodes are available to this process - the container/sandbox was not given the "
            "GPU device. Expose it (docker --gpus all / device cgroup, or recreate the nodes "
            "with CAP_MKNOD) and the same code will select the GPU automatically."
        )
    if not gpu_present:
        return "no NVIDIA GPU detected on this host"
    return "CUDA is not usable from this process (driver/library mismatch or device busy)"


def pick_weights(report: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    """
    Chooses weights for the detected hardware. Returns (weights, why).

    On CPU we stay deliberately small, because per-frame latency is the binding
    constraint there. On a GPU the model is essentially free, so we spend the headroom
    on accuracy instead.
    """
    report = report or device_report()
    if not report["cuda_available"]:
        return CPU_WEIGHTS, (
            f"{CPU_WEIGHTS} chosen for CPU: per-frame latency is the constraint "
            f"({report.get('reason') or 'no CUDA device'})"
        )

    free = report.get("free_vram_gb")
    if free is None:
        return "yolov8s-seg.pt", "CUDA available but free VRAM unknown; using a small model"

    for minimum, weights, _label in WEIGHT_TIERS:
        if free >= minimum:
            return weights, (
                f"{weights} chosen for {report.get('gpu_name')} "
                f"({free:.1f} GB free of {report.get('total_vram_gb')} GB)"
            )
    return "yolov8s-seg.pt", "CUDA available; using the smallest GPU model"


def load_segmenter(prefer: Optional[str] = None, weights: Optional[str] = None) -> Any:
    """
    Returns the best available segmenter for this machine.

    ``prefer="sam3"`` forces the slow model (deep-pass comparison only). Otherwise the
    weights are chosen from the detected device, with a fallback ladder so a GPU that
    cannot fit the chosen model still ends up running something.
    """
    if prefer == "sam3":
        from src.cv.sam3_masker import SAM3Masker

        return SAM3Masker.get_instance()

    report = device_report()
    chosen, why = (weights, "explicitly requested") if weights else pick_weights(report)
    print(f"[segmenter] {why}")

    ladder = [chosen] + [
        w for _m, w, _l in WEIGHT_TIERS if w != chosen
    ] + [CPU_WEIGHTS]
    seen = set()
    last_error: Optional[Exception] = None
    for candidate in ladder:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            segmenter = YoloSegmenter(weights=candidate)
            segmenter.device_note = why
            return segmenter
        except Exception as exc:  # noqa: BLE001 - try the next size down
            last_error = exc
            print(f"[segmenter] {candidate} failed to load ({type(exc).__name__}: {exc}); trying smaller")

    print(f"[segmenter] no YOLO segmentation model loaded ({last_error}); falling back to SAM 3")
    from src.cv.sam3_masker import SAM3Masker

    return SAM3Masker.get_instance()
