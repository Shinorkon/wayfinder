"""
Wayfinder telemetry layer.

This module is the single source of truth for every passenger number the dashboard
shows. Two provenance kinds exist and nothing else:

  measured   arrival rate and queue come from scripts/analyze_footage.py
  simulated  arrival rate is a measured base rate multiplied by an explicitly
             declared, per-scenario factor. The factor and the base rate are both
             returned to the client, so the UI can show exactly what was scaled.

There is no random number generation anywhere in this path.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.intelligence.dispatch_engine import FleetDispatchOptimizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Scenarios: declared demand multipliers applied to MEASURED base rates.
# ---------------------------------------------------------------------------

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "observed": {
        "label": "Observed (footage)",
        "description": "Measured arrival rates straight from the analysed clips. No scaling applied.",
        "multiplier": 1.0,
        "simulated": False,
    },
    "morning_peak": {
        "label": "Morning peak",
        "description": (
            "Simulated commute surge. Measured base rates are multiplied by 1.6 at "
            "residential stops and 1.2 elsewhere. This is a demand model, not a measurement."
        ),
        "multiplier": 1.2,
        "stop_multipliers": {
            "hulhumale_hiyaa_1": 1.6,
            "hulhumale_hiyaa_2": 1.6,
            "hulhumale_amin_avenue": 1.6,
            "hulhumale_ferry_terminal": 1.4,
        },
        "simulated": True,
    },
    "monsoon": {
        "label": "Monsoon downpour",
        "description": (
            "Simulated weather event. Boarding slows and shelter occupancy rises, so the "
            "measured base rates are multiplied by 1.35 uniformly. This is a demand model, "
            "not a measurement."
        ),
        "multiplier": 1.35,
        "simulated": True,
    },
    "ferry_arrival": {
        "label": "Ferry arrival",
        "description": (
            "Simulated ferry disembarkation pulse. Base rates are multiplied by 1.2, with 2.5x "
            "at the ferry terminal and Carnival Terminal interchange. This is a demand model, "
            "not a measurement."
        ),
        "multiplier": 1.2,
        "stop_multipliers": {
            "hulhumale_ferry_terminal": 2.5,
            "male_carnival": 2.5,
        },
        "simulated": True,
    },
    "off_peak": {
        "label": "Off-peak",
        "description": (
            "Simulated quiet period. Base rates are multiplied by 0.4 to represent off-peak "
            "demand. This is a demand model, not a measurement."
        ),
        "multiplier": 0.4,
        "simulated": True,
    },
}

DEFAULT_SCENARIO = "observed"

# Arrival rates above this are treated as a tracking artefact rather than demand. A
# single bus stop boarding more than this many passengers per minute continuously would
# be a rail interchange, not a shelter.
PLAUSIBLE_MAX_ARRIVAL_PAX_PER_MIN = 25.0


@dataclass
class StopTelemetry:
    stop: Dict[str, Any]
    arrival_rate_pax_per_min: float
    arrival_rate_base_pax_per_min: Optional[float]
    scenario_multiplier: float
    queue_pax: float
    queue_source: str
    queue_baseline_measured: Optional[float]
    provenance: Dict[str, Any]
    dispatch: Dict[str, Any]
    served: bool
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.stop["id"],
            "name": self.stop["name"],
            "zone": self.stop.get("zone"),
            "lat": self.stop["lat"],
            "lng": self.stop["lng"],
            "description": self.stop.get("description"),
            "capacity_threshold": self.stop.get("capacity_threshold"),
            "video_filename": self.stop.get("video_filename"),
            "keyframes": self.stop.get("keyframes", {}),
            "keyframe_timestamp_s": self.stop.get("keyframe_timestamp_s"),
            "detections_in_keyframe": self.stop.get("detections_in_keyframe"),
            "segmentation": self.stop.get("segmentation", {"available": False}),
            "footage_provenance": self.stop.get("footage_provenance"),
            "served_by_routes": self.dispatch.get("routes_serving", []),
            "arrival_rate_pax_per_min": round(self.arrival_rate_pax_per_min, 2),
            "arrival_rate_base_pax_per_min": (
                round(self.arrival_rate_base_pax_per_min, 2)
                if self.arrival_rate_base_pax_per_min is not None
                else None
            ),
            "scenario_multiplier": self.scenario_multiplier,
            "queue_pax": round(self.queue_pax, 1),
            "queue_source": self.queue_source,
            "queue_baseline_measured": (
                round(self.queue_baseline_measured, 1)
                if self.queue_baseline_measured is not None
                else None
            ),
            "provenance": self.provenance,
            "dispatch": self.dispatch,
            "served": self.served,
            "warnings": self.warnings,
        }


class TelemetryService:
    """Builds stop telemetry from the offline analysis index."""

    def __init__(
        self,
        network: Dict[str, Any],
        analysis_index: Optional[Dict[str, Any]] = None,
        scenario: str = DEFAULT_SCENARIO,
    ):
        self.network = network
        self.optimizer = FleetDispatchOptimizer(network)
        self.index = analysis_index
        self.scenario = scenario if scenario in SCENARIOS else DEFAULT_SCENARIO

        # index by stop id
        self.measured_by_stop: Dict[str, Dict[str, Any]] = {}
        if analysis_index:
            for s in analysis_index.get("stops", []):
                self.measured_by_stop[s["id"]] = s

    # -- configuration -----------------------------------------------------

    @property
    def scenario_meta(self) -> Dict[str, Any]:
        meta = SCENARIOS[self.scenario]
        return {
            "key": self.scenario,
            "label": meta["label"],
            "description": meta["description"],
            "simulated": meta["simulated"],
        }

    def set_scenario(self, scenario: str) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(
                f"unknown scenario '{scenario}'; expected one of {sorted(SCENARIOS)}"
            )
        self.scenario = scenario

    def list_scenarios(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": k,
                "label": v["label"],
                "description": v["description"],
                "simulated": v["simulated"],
                "multiplier": v["multiplier"],
            }
            for k, v in SCENARIOS.items()
        ]

    # -- computation -------------------------------------------------------

    def _stop_multiplier(self, stop_id: str) -> float:
        meta = SCENARIOS[self.scenario]
        return float(meta.get("stop_multipliers", {}).get(stop_id, meta["multiplier"]))

    def has_analysis(self) -> bool:
        return bool(self.measured_by_stop)

    def compute_stop(self, stop: Dict[str, Any]) -> StopTelemetry:
        row = self.measured_by_stop.get(stop["id"], {})
        measured = row.get("measured")
        provenance_row = row.get("footage_provenance")
        # The network config does not know about analysed footage, so enrich the stop
        # record with the analysis index row before it is serialised.
        enriched_stop = {
            **stop,
            "keyframes": row.get("keyframes", {}),
            "keyframe_timestamp_s": row.get("keyframe_timestamp_s"),
            "detections_in_keyframe": row.get("detections_in_keyframe"),
            "segmentation": row.get("segmentation", {"available": False}),
            "footage_provenance": provenance_row,
        }
        warnings: List[str] = []

        if measured:
            base_rate = float(measured["arrival_rate_pax_per_min"])
            confidence = measured["confidence"]
            provenance = {
                "kind": "measured",
                "confidence": confidence,
                "confidence_reasons": measured["confidence_reasons"],
                "clip": (provenance_row or {}).get("clip"),
                "clip_title": (provenance_row or {}).get("title"),
                "clip_city": (provenance_row or {}).get("city"),
                "clip_source": (provenance_row or {}).get("source"),
                "clip_license": (provenance_row or {}).get("license"),
                "disclosure": (provenance_row or {}).get("disclosure"),
                "redistributable": (provenance_row or {}).get("redistributable"),
                "analysed_span_s": measured["analysed_span_s"],
                "waiting_tracks": measured["waiting_tracks"],
                "unique_tracks": measured["unique_tracks"],
                "mean_dwell_s": measured["mean_dwell_s"],
                "queue_series": measured["queue_series"],
                "limitations": measured["limitations"],
                "analysis_method": measured["analysis_method"],
                "note": (
                    "Arrival rate measured from public reference footage of a different city. "
                    "It is a real measurement of that footage, not of this Maldivian stop."
                ),
            }
            if confidence == "low":
                warnings.append(
                    "Measured arrival rate is low confidence; dispatch recommendation is restricted."
                )
        else:
            base_rate = None
            confidence = "none"
            provenance = {
                "kind": "unavailable",
                "note": (
                    "No analysed footage for this stop, so no arrival rate can be measured. "
                    "No number is shown rather than an invented one."
                ),
            }
            warnings.append("No footage analysed for this stop.")

        multiplier = self._stop_multiplier(stop["id"])
        simulated = bool(SCENARIOS[self.scenario]["simulated"])

        if base_rate is None:
            rate = None
            queue = None
            queue_source = "unavailable"
            queue_baseline = None
            dispatch_dict = self._unavailable_dispatch(stop)
        elif not self.optimizer.routes_serving(stop["id"]):
            # Measured demand exists, but no route serves the stop, so there is no
            # timetable to optimise. Say that instead of borrowing another route's.
            rate = base_rate * multiplier
            queue = (measured["queue_mean_pax"] * multiplier) if measured else 0.0
            queue_source = "measured mean standing queue x scenario factor"
            queue_baseline = float(measured["queue_mean_pax"]) if measured else None
            dispatch_dict = self._unrouted_dispatch(stop)
            warnings.append("No configured route serves this stop; no headway recommendation issued.")
        else:
            rate = base_rate * multiplier
            queue_baseline = float(measured["queue_mean_pax"])
            queue_baseline_scaled = queue_baseline * multiplier
            dispatch = self.optimizer.evaluate_stop(
                stop_id=stop["id"],
                arrival_rate_pax_per_min=rate,
                current_queue_pax=queue_baseline_scaled,
                queue_source=(
                    f"measured mean standing queue x scenario factor ({multiplier:g})"
                    if simulated
                    else "measured mean standing queue"
                ),
                measurement_confidence=confidence,
            )
            # Queue evolution over one service interval: observed standing queue plus
            # net accumulation. Deterministic, no randomness.
            balance = dispatch.net_clearance_pax_per_min or 0.0
            horizon = float(dispatch.recommended_headway_min)
            queue = max(0.0, queue_baseline_scaled + balance * horizon)
            queue_source = "measured mean standing queue + modelled growth over one service interval"
            warnings.extend(dispatch.warnings)
            dispatch_dict = dispatch.to_dict()

            # Plausibility guard: a short clip whose tracker fragmented one person into
            # several tracks can produce an arrival rate that is arithmetically correct
            # and operationally absurd. Say so rather than letting it stand unqualified.
            plausible = PLAUSIBLE_MAX_ARRIVAL_PAX_PER_MIN
            if dispatch_dict["arrival_rate_pax_per_min"] > plausible and confidence == "low":
                warning = (
                    f"Measured arrival rate ({dispatch_dict['arrival_rate_pax_per_min']:.1f} pax/min) "
                    f"exceeds what a single stop plausibly receives. This is almost certainly "
                    f"fragmented tracking in a short clip, not real demand."
                )
                warnings.append(warning)
                dispatch_dict.setdefault("warnings", []).append(warning)
                if dispatch_dict["status"] == "CAPACITY_FAILURE":
                    dispatch_dict["status"] = "MEASUREMENT_SUSPECT"

            if provenance["kind"] == "measured":
                provenance["scenario_scaling"] = {
                    "applied": simulated,
                    "factor": multiplier,
                    "base_arrival_rate_pax_per_min": round(base_rate, 2),
                    "scenario": self.scenario,
                    "explanation": SCENARIOS[self.scenario]["description"],
                }

        return StopTelemetry(
            stop=enriched_stop,
            arrival_rate_pax_per_min=rate if rate is not None else 0.0,
            arrival_rate_base_pax_per_min=base_rate,
            scenario_multiplier=multiplier,
            queue_pax=queue if queue is not None else 0.0,
            queue_source=queue_source,
            queue_baseline_measured=queue_baseline,
            provenance=provenance,
            dispatch=dispatch_dict,
            served=bool(dispatch_dict.get("routes_serving")),
            warnings=warnings,
        )

    def _unavailable_dispatch(self, stop: Dict[str, Any]) -> Dict[str, Any]:
        serving = self.optimizer.routes_serving(stop["id"])
        return {
            "stop_id": stop["id"],
            "stop_name": stop.get("name"),
            "status": "NO_DATA",
            "routes_serving": serving,
            "primary_route": serving[0] if serving else None,
            "directive_summary": "No measured arrival rate available for this stop; no recommendation issued.",
            "reasoning_steps": [
                "No analysed footage is mapped to this stop, so the arrival rate is unknown.",
                "Wayfinder does not substitute an assumed rate for a measured one.",
            ],
            "warnings": ["No footage analysed for this stop."],
        }

    def _unrouted_dispatch(self, stop: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "stop_id": stop["id"],
            "stop_name": stop.get("name"),
            "status": "NO_ROUTE",
            "routes_serving": [],
            "primary_route": None,
            "capacity_ok": True,
            "directive_summary": (
                "This stop is not served by any configured route, so no headway or fleet "
                "recommendation can be computed."
            ),
            "reasoning_steps": [
                "Demand has been measured for this stop.",
                "No route in the network configuration lists this stop, so there is no schedule to optimise.",
                (stop.get("modelling_note") or "").strip() or "Place the stop on a route to enable recommendations.",
            ],
            "warnings": ["No configured route serves this stop."],
        }

    # -- aggregates --------------------------------------------------------

    def compute_all(self) -> List[StopTelemetry]:
        return [self.compute_stop(s) for s in self.network["stops"]]

    def network_summary(self, stop_states: List[StopTelemetry]) -> Dict[str, Any]:
        with_data = [s for s in stop_states if s.provenance["kind"] == "measured"]
        simulated = bool(SCENARIOS[self.scenario]["simulated"])

        total_waiting = sum(s.queue_pax for s in with_data)
        total_inflow = sum(s.arrival_rate_pax_per_min for s in with_data)

        baseline_waits = [s.dispatch.get("baseline_wait_min", 0.0) for s in with_data]
        optimized_waits = [s.dispatch.get("optimized_wait_min", 0.0) for s in with_data]
        mean_baseline = sum(baseline_waits) / len(baseline_waits) if baseline_waits else 0.0
        mean_optimized = sum(optimized_waits) / len(optimized_waits) if optimized_waits else 0.0
        delta_pct = (
            round((mean_baseline - mean_optimized) / mean_baseline * 100.0, 1)
            if mean_baseline > 0
            else 0.0
        )

        # Fleet: sum the buses each route needs for its worst stop, not the sum of
        # per-stop requirements (which would triple-count the same vehicle).
        fleet_by_route: Dict[str, int] = {}
        for s in with_data:
            route = s.dispatch.get("primary_route")
            if not route:
                continue
            fleet_by_route[route] = max(
                fleet_by_route.get(route, 0), int(s.dispatch.get("buses_required", 0))
            )
        fleet_total = sum(fleet_by_route.values())
        configured_total = sum(v.get("count", 0) for v in self.network.get("fleet_composition", {}).values())

        status_counts: Dict[str, int] = {}
        for s in stop_states:
            key = s.dispatch.get("status", "UNKNOWN")
            status_counts[key] = status_counts.get(key, 0) + 1

        capacity_failures = [
            s.stop["id"] for s in with_data if not s.dispatch.get("capacity_ok", True)
        ]
        shortfalls = {
            s.stop["id"]: s.dispatch.get("fleet_shortfall", 0)
            for s in with_data
            if s.dispatch.get("fleet_shortfall")
        }

        return {
            "scenario": self.scenario_meta,
            "data_status": {
                "stops_total": len(stop_states),
                "stops_with_measurements": len(with_data),
                "stops_without_measurements": len(stop_states) - len(with_data),
                "arrival_rates_simulated": simulated,
                "wait_times_modelled": True,
                "disclosure": (
                    "Arrival rates are measured from public reference footage of other cities and "
                    "then scaled by a declared scenario factor. Wait times are modelled from schedule "
                    "parameters, not observed. No official RTL operational data is used."
                    if simulated
                    else
                    "Arrival rates are measured from public reference footage of other cities. Wait "
                    "times are modelled from schedule parameters, not observed. No official RTL "
                    "operational data is used."
                ),
            },
            "kpis": {
                "total_waiting_pax": round(total_waiting, 1),
                "total_inflow_pax_per_min": round(total_inflow, 2),
                "mean_baseline_wait_min": round(mean_baseline, 1),
                "mean_optimized_wait_min": round(mean_optimized, 1),
                "wait_change_pct": delta_pct,
                "wait_change_direction": "reduction" if delta_pct > 0 else ("increase" if delta_pct < 0 else "none"),
                "wait_formula": "Welch's formula E[W] = H/2 x (1 + CV^2), averaged over stops with data",
                "buses_required_by_route": fleet_by_route,
                "buses_required_total": fleet_total,
                "buses_configured_total": configured_total,
                "status_counts": status_counts,
                "capacity_failure_stops": capacity_failures,
                "fleet_shortfall_stops": shortfalls,
            },
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


def load_analysis_index(path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    path = path or os.path.join(ROOT, "data", "analysis", "index.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
