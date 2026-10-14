"""
Wayfinder API server.

Contract:
  * Every passenger number comes from src/intelligence/telemetry.py, which either
    reports a measured value from data/analysis/ or reports that no measurement
    exists. The server never synthesises telemetry.
  * Every response that carries demand data also carries its provenance, so the
    client cannot display a number without knowing where it came from.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.cv.live_monitor import MonitorHub
from src.cv.segmenter import device_report, load_segmenter, pick_weights
from src.cv.stop_analysis import zone_mask_from_rows
from src.intelligence.deepseek_agent import DeepSeekDispatchAgent
from src.intelligence.recommendation import live_recommendation, plain_recommendation
from src.intelligence.telemetry import (
    DEFAULT_SCENARIO,
    SCENARIOS,
    TelemetryService,
    load_analysis_index,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NETWORK_PATH = os.path.join(BASE_DIR, "data", "routes", "male_rtl_network.json")
FOOTAGE_DIR = os.path.join(BASE_DIR, "data", "footage")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
WEB_DIR = os.path.join(BASE_DIR, "web")

with open(NETWORK_PATH, "r", encoding="utf-8") as f:
    network_data = json.load(f)

analysis_index = load_analysis_index()

app = FastAPI(
    title="Wayfinder RTL Intelligence API",
    description=(
        "Bus-stop demand measurement and fleet frequency modelling for the Greater Malé "
        "RTL network. Arrival rates are measured from public reference footage; wait times "
        "are modelled. This is a prototype, not an operational dispatch system."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

telemetry = TelemetryService(network_data, analysis_index, scenario=DEFAULT_SCENARIO)
deepseek_agent = DeepSeekDispatchAgent()


# ---------------------------------------------------------------------------
# Continuous monitoring
# ---------------------------------------------------------------------------

def _segmenter_factory():
    """
    The fast per-frame model: a YOLO segmentation network.

    Returns boxes and masks from one forward pass, which is what makes per-frame masks
    affordable. load_segmenter() picks the weights from the detected device (free VRAM on
    a GPU, smallest weights on CPU) and records why, so the UI can report it.
    """
    return load_segmenter()


def _deep_pass_factory():
    """Optional slow SAM 3, used only when explicitly requested for comparison."""
    from src.cv.sam3_masker import SAM3Masker

    return SAM3Masker.get_instance()


monitor_hub = MonitorHub(
    detector_factory=lambda: __import__("src.cv.detector", fromlist=["PersonDetector"]).PersonDetector(),
    segmenter_factory=_segmenter_factory,
    deep_pass_factory=_deep_pass_factory,
)

# Waiting-zone raster per stop, reused from the offline analysis so live and offline
# measurements mean the same thing. Built once at import from the stored artifacts.
ZONE_MASKS: Dict[str, Any] = {}
if analysis_index:
    for row in analysis_index.get("stops", []):
        zone = ((row.get("measured") or {}).get("waiting_zone")) or row.get("waiting_zone") or {}
        mask = zone_mask_from_rows(zone.get("mask_rows"))
        if mask is not None:
            ZONE_MASKS[row["id"]] = mask


def _stop_by_id(stop_id: str) -> Optional[Dict[str, Any]]:
    return next((s for s in network_data["stops"] if s["id"] == stop_id), None)


def _clip_path(stop: Dict[str, Any]) -> Optional[str]:
    name = stop.get("video_filename")
    if not name:
        return None
    path = os.path.join(FOOTAGE_DIR, name)
    return path if os.path.exists(path) else None


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@app.get("/api/device")
def device() -> Dict[str, Any]:
    """
    Reports the accelerator actually available, the model chosen for it, and - when a GPU
    is present but unusable - the specific reason, so it can be fixed rather than guessed.
    """
    report = device_report()
    weights, why = pick_weights(report)
    return {
        **report,
        "chosen_weights": weights,
        "why": why,
        "note": (
            "Model size is chosen from free VRAM. Every model in the ladder is under 30M "
            "parameters, so GPU memory is never the constraint on a modern laptop GPU; "
            "per-frame latency is."
        ),
    }


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "analysis_loaded": telemetry.has_analysis(),
        "stops": len(network_data.get("stops", [])),
        "scenario": telemetry.scenario,
    }


@app.get("/api/meta")
def meta() -> Dict[str, Any]:
    """Everything the client needs to describe the data honestly."""
    cov = (analysis_index or {}).get("coverage", {})
    return {
        "network": network_data.get("network"),
        "data_status": network_data.get("data_status"),
        "network_geometry": [
            {
                "id": r["id"],
                "name": r["name"],
                "color": r.get("color"),
                "waypoints": r.get("waypoints", []),
                "osrm_geometry": r.get("osrm_geometry", []),
            }
            for r in network_data.get("routes", [])
        ],
        "analysis": {
            "available": telemetry.has_analysis(),
            "generated_at": (analysis_index or {}).get("generated_at"),
            "generator": (analysis_index or {}).get("generator"),
            "coverage": cov,
        },
        "scenarios": telemetry.list_scenarios(),
        "current_scenario": telemetry.scenario_meta,
        "method": {
            "perception": (
                "YOLOv8n person detection per frame plus IoU/constant-velocity tracking "
                "across frames"
            ),
            "measurement": (
                "Arrival rate = distinct tracks entering the derived waiting zone and then "
                "standing still, per minute of footage"
            ),
            "wait_time": "Welch's formula E[W] = H/2 x (1 + CV^2)",
            "detector_ms_note": "measured live by the monitor; see the performance block in /api/live/{stop_id}",
            "live": {
                "detector_target_fps": 5.0,
                "device": device_report().get("device"),
                "device_reason": device_report().get("reason"),
                "segmentation_model": pick_weights()[0],
                "segmentation": (
                    "Masks and boxes come from one YOLO segmentation forward pass, roughly "
                    "40 ms per frame on CPU, so masks refresh with the counts. SAM 3 is "
                    "available as an optional deep pass but needs about 14 s per frame on "
                    "this CPU and accepts only 1008x1008 input, so it cannot run per frame."
                ),
                "model_comparison_ms_per_frame": {
                    "yolov8n-seg": 39,
                    "yolo11n-seg": 112,
                    "yolov8s-seg": 237,
                    "sam3_1008px": 14000,
                },
            },
            "limitations": [
                "Clip footage is public reference material from other cities, not RTL cameras.",
                "Track identity is IoU-based; dense crowds inflate arrival counts.",
                "Wait times are modelled from schedule parameters, not observed on the network.",
                "No official RTL demand, fleet or timetable data is used anywhere.",
            ],
        },
        "footage_note": (
            "All footage is public reference video from other cities, used to exercise the "
            "detector. It is not RTL and not Malé. Each clip's source and licence is recorded "
            "in data/footage/SOURCES.json."
        ),
        "disclaimer": (
            "Prototype for a datathon. Demand figures are measurements of public reference "
            "footage scaled by declared scenario factors, or are marked unavailable. Nothing "
            "here is official RTL operational data."
        ),
    }


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


@app.get("/api/network")
def get_network() -> Dict[str, Any]:
    return network_data


@app.get("/api/routes/{route_id}/geometry")
def get_route_geometry(route_id: str) -> Dict[str, Any]:
    route = next((r for r in network_data["routes"] if r["id"] == route_id), None)
    if not route:
        raise HTTPException(status_code=404, detail=f"route {route_id} not found")
    return {
        "id": route["id"],
        "name": route["name"],
        "color": route.get("color"),
        "stops": route.get("stops", []),
        "geometry": route.get("osrm_geometry") or route.get("waypoints", []),
        "distance_km": route.get("distance_km"),
        "travel_time_min": route.get("travel_time_min"),
        "source": route.get("source"),
        "default_headway_min": route.get("default_headway_min"),
    }


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


@app.get("/api/stops")
def get_stops() -> Dict[str, Any]:
    states = telemetry.compute_all()
    return {
        "scenario": telemetry.scenario_meta,
        "stops": [s.to_dict() for s in states],
    }


@app.get("/api/stops/{stop_id}")
def get_stop(stop_id: str) -> Dict[str, Any]:
    stop = next((s for s in network_data["stops"] if s["id"] == stop_id), None)
    if not stop:
        raise HTTPException(status_code=404, detail=f"stop {stop_id} not found")
    return telemetry.compute_stop(stop).to_dict()


@app.get("/api/telemetry")
def get_network_telemetry() -> Dict[str, Any]:
    return telemetry.network_summary(telemetry.compute_all())


class ScenarioRequest(BaseModel):
    scenario: str = Field(..., description=f"one of {sorted(SCENARIOS)}")


@app.post("/api/scenario")
def set_scenario(req: ScenarioRequest) -> Dict[str, Any]:
    try:
        telemetry.set_scenario(req.scenario)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "status": "ok",
        "scenario": telemetry.scenario_meta,
        "telemetry": telemetry.network_summary(telemetry.compute_all()),
    }


# ---------------------------------------------------------------------------
# Optional advisory layer (DeepSeek)
# ---------------------------------------------------------------------------


class ReasonRequest(BaseModel):
    stop_id: str
    api_key: Optional[str] = None


# ---------------------------------------------------------------------------
# Live monitoring
# ---------------------------------------------------------------------------


class MonitorRequest(BaseModel):
    stop_id: str
    fps: float = Field(5.0, ge=0.5, le=25.0, description="detection samples per second")
    segmentation: bool = Field(True, description="per-frame masks from the segmentation model")
    deep_pass_interval_s: float = Field(
        0.0,
        ge=0.0,
        le=600.0,
        description=(
            "seconds between optional SAM 3 deep passes. 0 disables it. SAM 3 needs "
            "roughly 14 s per frame on CPU, so this is for comparison, not monitoring."
        ),
    )


@app.get("/api/live/{stop_id}")
def live_state(stop_id: str) -> Dict[str, Any]:
    """Current monitor state for a stop, plus the plain-language recommendation."""
    stop = _stop_by_id(stop_id)
    if not stop:
        raise HTTPException(status_code=404, detail=f"stop {stop_id} not found")

    state = telemetry.compute_stop(stop)
    live = monitor_hub.snapshot(stop_id)
    recommendation = plain_recommendation(state.dispatch, state.provenance)
    if live:
        recommendation = live_recommendation(
            live, state.provenance.get("measured"), state.dispatch, state.provenance
        )

    return {
        "stop_id": stop_id,
        "stop_name": stop.get("name"),
        "live": live,
        "monitoring": live is not None and live.get("running", False),
        "device": device_report(),
        "recommendation": recommendation,
        "provenance": state.provenance,
        "active_monitors": monitor_hub.active(),
    }


@app.post("/api/live/start")
def live_start(req: MonitorRequest) -> Dict[str, Any]:
    """Starts continuous monitoring of one stop. Only the newest stop keeps running."""
    stop = _stop_by_id(req.stop_id)
    if not stop:
        raise HTTPException(status_code=404, detail=f"stop {req.stop_id} not found")
    clip = _clip_path(stop)
    if not clip:
        raise HTTPException(
            status_code=409,
            detail=(
                f"No footage is available for {req.stop_id}, so it cannot be monitored. "
                "Wayfinder reports this rather than simulating a feed."
            ),
        )

    # One monitored stop at a time: the detector is a shared resource and pretending to
    # watch every camera at once would be a claim this hardware cannot support.
    for other in monitor_hub.active():
        if other != req.stop_id:
            monitor_hub.stop(other)

    segmenter = monitor_hub.segmenter() if req.segmentation else None
    monitor = monitor_hub.start(
        stop_id=req.stop_id,
        clip_path=clip,
        zone_mask=ZONE_MASKS.get(req.stop_id),
        fps=req.fps,
        with_segmentation=req.segmentation,
        deep_pass_interval_s=req.deep_pass_interval_s,
    )
    return {
        "status": "started",
        "stop_id": req.stop_id,
        "fps": req.fps,
        "segmentation_model": getattr(segmenter, "name", None),
        "segmentation_device": getattr(segmenter, "device", None),
        "device_reason": getattr(segmenter, "device_note", None),
        "segmentation": segmenter is not None,
        "deep_pass": req.deep_pass_interval_s > 0,
        "zone_calibrated": req.stop_id in ZONE_MASKS,
    }


@app.post("/api/live/stop")
def live_stop(req: MonitorRequest) -> Dict[str, Any]:
    stopped = monitor_hub.stop(req.stop_id)
    return {"status": "stopped" if stopped else "not_running", "stop_id": req.stop_id}


@app.get("/api/live")
def live_all() -> Dict[str, Any]:
    """Every active monitor, newest state only. Used by the map view."""
    out = []
    for stop_id in monitor_hub.active():
        snap = monitor_hub.snapshot(stop_id)
        if snap:
            out.append(
                {
                    "stop_id": stop_id,
                    "people_in_frame": snap["people_in_frame"],
                    "people_waiting": snap["people_waiting"],
                    "arrival_rate_pax_per_min": snap["arrival_rate_pax_per_min"],
                    "age_s": snap["age_s"],
                }
            )
    return {"monitors": out}


@app.post("/api/advisory")
def advisory(req: ReasonRequest) -> Dict[str, Any]:
    """
    Optional narrative commentary on a stop's computed state.

    The deterministic engine has already produced every number. This endpoint can add
    an LLM narrative, clearly labelled as advisory, and it is told explicitly not to
    invent figures. If no API key is configured the response says so.
    """
    stop = next((s for s in network_data["stops"] if s["id"] == req.stop_id), None)
    if not stop:
        raise HTTPException(status_code=404, detail=f"stop {req.stop_id} not found")

    state = telemetry.compute_stop(stop)
    dispatch = state.dispatch

    if not req.api_key and not deepseek_agent.api_key:
        return {
            "stop_id": req.stop_id,
            "available": False,
            "reason": "No DeepSeek API key configured. All figures on this screen come from the deterministic engine.",
            "engine_output": {
                "directive_summary": dispatch.get("directive_summary"),
                "reasoning_steps": dispatch.get("reasoning_steps", []),
                "warnings": state.warnings,
            },
        }

    if req.api_key:
        deepseek_agent.set_api_key(req.api_key)

    narrative = deepseek_agent.reason(
        stop_name=stop["name"],
        arrival_rate_pax_per_min=state.arrival_rate_pax_per_min,
        queue_pax=state.queue_pax,
        default_headway_min=dispatch.get("default_headway_min", 15),
        recommended_headway_min=dispatch.get("recommended_headway_min", 15),
        route_name=dispatch.get("primary_route") or "unassigned",
        scenario=telemetry.scenario,
        measured_rate=state.arrival_rate_base_pax_per_min,
        confidence=state.provenance.get("confidence", "none"),
    )
    return {
        "stop_id": req.stop_id,
        "available": narrative.get("source") == "deepseek-api-live",
        "advisory": narrative,
        "engine_output": {
            "directive_summary": dispatch.get("directive_summary"),
            "reasoning_steps": dispatch.get("reasoning_steps", []),
            "warnings": state.warnings,
        },
    }


# ---------------------------------------------------------------------------
# Static mounts
# ---------------------------------------------------------------------------

if os.path.exists(FOOTAGE_DIR):
    app.mount("/footage", StaticFiles(directory=FOOTAGE_DIR), name="footage")
if os.path.exists(OUTPUT_DIR):
    app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

if os.path.exists(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
