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

from src.intelligence.deepseek_agent import DeepSeekDispatchAgent
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
# Metadata
# ---------------------------------------------------------------------------


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
    """Everything a client needs to describe the data honestly."""
    cov = (analysis_index or {}).get("coverage", {})
    return {
        "network": network_data.get("network"),
        "data_status": network_data.get("data_status"),
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
                "YOLOv8n person detection (per frame) + IoU/constant-velocity tracking "
                "(temporal) + dwell-density waiting-zone occupancy"
            ),
            "measurement": (
                "Arrival rate = distinct tracks entering the derived waiting zone per minute, "
                "restricted to tracks that then remained stationary"
            ),
            "wait_time": "Welch's formula E[W] = H/2 x (1 + CV^2)",
            "limitations": [
                "Clip footage is public reference material from other cities, not RTL cameras.",
                "Track identity is IoU-based; dense crowds inflate arrival counts.",
                "Wait times are modelled from schedule parameters, not observed on the network.",
                "No official RTL demand, fleet or timetable data is used anywhere.",
            ],
        },
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
