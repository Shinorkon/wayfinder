"""
Wayfinder fleet frequency optimizer.

The model, stated explicitly so it can be checked:

  Demand side (measured, not assumed)
      lambda  = measured passenger arrival rate at a stop (pax/min), from
                src/cv/stop_analysis.py
      q0      = observed queue length at the stop (pax)

  Service side (schedule parameters)
      c       = seats per vehicle for the assigned fleet type
      H       = headway in minutes between consecutive buses of the route
      P       = passenger capacity actually offered = c * (1 - reserve_fraction)
      mu      = service rate = P / H   (pax/min the stop can board)

  Bottleneck
      demand_rate  = lambda * (1 + route_bunching_penalty)
      net          = mu - demand_rate      (pax/min)
      net <= 0  ->  the stop cannot clear its queue at this headway. That is
                    reported as a capacity failure, NOT clamped to a nice number.

  Waiting time (Welch's formula for average wait at a bus stop)
      E[W] = H/2 * (1 + CV^2)
      CV = 0.5 by default (an assumption, exposed as a parameter, not buried)

  Time to clear the observed queue
      T = q0 / net   if net > 0 else infinity

Fleet
      buses_required = ceil(round_trip_minutes / H), capped by the fleet that
      actually exists on the route and by depot headroom. If the required fleet
      exceeds what is available, that shortfall is reported instead of being
      silently satisfied by a phantom double-decker.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

STATUS_ORDER = ["NORMAL", "ELEVATED", "SURGE", "CAPACITY_FAILURE"]

DEFAULT_SERVICE = {
    # Fraction of a vehicle's seats deliberately held back so a bus that is running
    # slightly late still has room. 0.15 means we plan to fill 85% of seats.
    "reserve_fraction": 0.15,
    # Extra demand multiplier to account for the fact that observed headway variance
    # bunches buses, so the effective arrival rate at a stop can exceed the mean.
    "route_bunching_penalty": 0.0,
    # Coefficient of variation of bus arrival intervals used in Welch's formula.
    "headway_cv": 0.5,
    # Extra minutes per stop for boarding/alighting dwell; affects round-trip time.
    "dwell_minutes_per_stop": 0.25,
}


@dataclass
class DispatchRecommendation:
    stop_id: str
    stop_name: str
    routes_serving: List[str]
    primary_route: str
    status: str
    # Inputs
    arrival_rate_pax_per_min: float
    current_queue_pax: float
    current_queue_source: str
    # Schedule
    default_headway_min: float
    recommended_headway_min: float
    recommended_headway_source: str
    headway_change_pct: float
    # Capacity
    bus_type: str
    bus_capacity_pax: int
    effective_capacity_pax: float
    offered_service_pax_per_min: float
    demand_rate_pax_per_min: float
    net_clearance_pax_per_min: Optional[float]
    utilization: Optional[float]
    capacity_ok: bool
    # Waiting time
    baseline_wait_min: float
    optimized_wait_min: float
    wait_reduction_pct: float
    wait_reduction_source: str
    baseline_formula: str
    optimized_formula: str
    # Fleet
    cycle_time_min: float
    buses_required: int
    buses_assigned: int
    extra_buses_available: int
    extra_buses_to_inject: int
    fleet_shortfall: int
    fleet_available: int
    # Queue dynamics
    queue_clear_time_min: Optional[float]
    queue_clear_note: str
    # Narrative
    directive_summary: str
    reasoning_steps: List[str]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FleetDispatchOptimizer:
    """Computes headway and fleet recommendations for Greater Malé RTL stops."""

    def __init__(self, network_config: Dict[str, Any], service_params: Optional[Dict[str, Any]] = None):
        self.config = network_config
        self.service = dict(DEFAULT_SERVICE)
        if network_config.get("service_parameters"):
            self.service.update(network_config["service_parameters"])
        if service_params:
            self.service.update(service_params)

        self.routes: Dict[str, Dict[str, Any]] = {r["id"]: r for r in network_config.get("routes", [])}
        self.stops: Dict[str, Dict[str, Any]] = {s["id"]: s for s in network_config.get("stops", [])}
        self.fleet: Dict[str, Dict[str, Any]] = network_config.get("fleet_composition", {
            "double_decker": {"capacity": 72, "count": 6},
            "standard_electric": {"capacity": 50, "count": 9},
            "minibus": {"capacity": 25, "count": 3},
        })
        self.depot_reserve: int = int(network_config.get("depot_reserve", 0))

        # stop_id -> [route_id, ...] built once, so attribution is deterministic
        # instead of depending on dict iteration order.
        self.stop_routes: Dict[str, List[str]] = {sid: [] for sid in self.stops}
        for route in self.routes.values():
            for sid in route.get("stops", []):
                self.stop_routes.setdefault(sid, [])
                if route["id"] not in self.stop_routes[sid]:
                    self.stop_routes[sid].append(route["id"])

        # Buses currently allocated per route (from config, else derived from fleet).
        self.route_fleet: Dict[str, int] = {}
        for route in self.routes.values():
            if "buses_assigned" in route:
                self.route_fleet[route["id"]] = int(route["buses_assigned"])

    # -- helpers -----------------------------------------------------------

    def routes_serving(self, stop_id: str) -> List[str]:
        return self.stop_routes.get(stop_id, [])

    def default_buses_for(self, route: Dict[str, Any]) -> int:
        if route["id"] in self.route_fleet:
            return self.route_fleet[route["id"]]
        cycle = self.cycle_time_min(route)
        return max(1, math.ceil(cycle / float(route.get("default_headway_min", 15))))

    def cycle_time_min(self, route: Dict[str, Any]) -> float:
        """Round-trip time including terminal turnaround and per-stop dwell."""
        one_way = float(route.get("travel_time_min", 25.0))
        n_stops = max(1, len(route.get("stops", [])))
        dwell = float(self.service["dwell_minutes_per_stop"]) * n_stops
        turnaround = float(route.get("terminal_turnaround_min", 3.0))
        return 2.0 * (one_way + dwell) + turnaround

    def _bus_for_load(self, demand_per_hour: float, headway_min: float) -> str:
        """Smallest vehicle whose planned capacity covers demand per departure."""
        per_departure = demand_per_hour * (headway_min / 60.0)
        for key in ("minibus", "standard_electric", "double_decker"):
            spec = self.fleet.get(key)
            if not spec:
                continue
            planned = spec["capacity"] * (1.0 - float(self.service["reserve_fraction"]))
            if planned >= per_departure:
                return key
        return "double_decker"

    # -- main entry point --------------------------------------------------

    def evaluate_stop(
        self,
        stop_id: str,
        arrival_rate_pax_per_min: float,
        current_queue_pax: float,
        queue_source: str = "unknown",
        route_id: Optional[str] = None,
        measurement_confidence: str = "unknown",
    ) -> DispatchRecommendation:
        stop = self.stops.get(stop_id)
        if not stop:
            raise ValueError(f"Stop {stop_id} not found in RTL network")

        serving = self.routes_serving(stop_id)
        if route_id is None:
            if not serving:
                raise ValueError(f"Stop {stop_id} is not served by any configured route")
            route_id = serving[0]
        elif route_id not in self.routes:
            raise ValueError(f"Route {route_id} not configured")

        route = self.routes[route_id]
        warnings: List[str] = []
        low_confidence = measurement_confidence == "low"
        if low_confidence:
            warnings.append(
                "Arrival rate is low-confidence measured data, so no headway compression is "
                "recommended. The reported headway is the existing schedule."
            )

        default_headway = float(route.get("default_headway_min", 15))
        min_headway = float(route.get("min_headway_min", 5))
        max_headway = float(route.get("max_headway_min", 30))
        threshold = float(stop.get("capacity_threshold", 25))
        lambda_ = max(0.0, float(arrival_rate_pax_per_min))
        q0 = max(0.0, float(current_queue_pax))

        reserve = float(self.service["reserve_fraction"])
        demand_rate = lambda_ * (1.0 + float(self.service["route_bunching_penalty"]))

        baseline_buses = self.default_buses_for(route)
        cycle = self.cycle_time_min(route)

        def wait_minutes(headway: float) -> float:
            cv = float(self.service["headway_cv"])
            return (headway / 2.0) * (1.0 + cv * cv)

        baseline_wait = wait_minutes(default_headway)
        baseline_formula = (
            f"Welch: E[W] = H/2 x (1 + CV^2) with H={default_headway:g} min, "
            f"CV={self.service['headway_cv']}"
        )

        # ---- Choose a headway ------------------------------------------------
        # Goal: offer enough capacity to clear both the standing queue and the
        # continuing arrivals, without exceeding the fleet that exists.
        demand_per_hour = demand_rate * 60.0
        bus_type = (
            self._bus_for_load(demand_per_hour, default_headway) if demand_per_hour > 0 else "minibus"
        )
        capacity = int(self.fleet[bus_type]["capacity"])
        planned_capacity = capacity * (1.0 - reserve)

        # Spare vehicles that can be pulled onto this route. Explicit depot reserve in
        # the network config wins; otherwise we may redeploy the rest of the class.
        class_total = int(self.fleet[bus_type].get("count", baseline_buses))
        if self.depot_reserve > 0:
            extra_available = self.depot_reserve
        else:
            extra_available = max(0, class_total - baseline_buses)
        max_service_buses = baseline_buses + extra_available
        fleet_floor_headway = cycle / max(1, max_service_buses)

        # Headway that clears the standing queue within half the current interval:
        #   P/H - lambda > 0  =>  H < P / (lambda + q0/T_target)
        target_clear_min = max(5.0, default_headway / 2.0)
        required_service = demand_rate + q0 / target_clear_min
        ideal_headway = (planned_capacity / required_service) if required_service > 0 else max_headway

        recommended = min(max(min_headway, fleet_floor_headway, ideal_headway), max_headway)
        recommended_source = "computed"
        if low_confidence:
            # Data quality gate: with an unevaluable arrival rate we do not compress the
            # schedule. Report the existing timetable instead of acting on noise.
            recommended = default_headway
            recommended_source = "held at schedule (low-confidence measurement)"
        elif q0 <= max(2.0, threshold * 0.25) and demand_per_hour < 60.0:
            # Genuinely quiet: spacing out is the correct (and cheaper) answer.
            recommended = min(max_headway, max(recommended, default_headway))
            recommended_source = "off-peak spacing"
        if not low_confidence and abs(recommended - default_headway) < 0.5:
            recommended = default_headway
            recommended_source = "schedule already adequate"

        headway_change_pct = round((recommended - default_headway) / default_headway * 100.0, 1)

        # ---- Fleet ------------------------------------------------------------
        buses_required = max(1, math.ceil(cycle / recommended))
        extra_buses = max(0, buses_required - baseline_buses)
        max_injectable = extra_available
        extra_to_inject = min(extra_buses, max_injectable)
        fleet_shortfall = max(0, extra_buses - extra_to_inject)
        if fleet_shortfall:
            warnings.append(
                f"Requires {buses_required} buses but only {baseline_buses + extra_to_inject} "
                f"can be assigned to {route_id} (shortfall {fleet_shortfall})."
            )

        # ---- Capacity check ---------------------------------------------------
        offered_service = planned_capacity / recommended
        net = offered_service - demand_rate
        utilization = (demand_rate / offered_service) if offered_service > 0 else None
        capacity_ok = net > 0

        # The queue only clears if the net service rate is positive.
        if net > 0 and q0 > 0:
            clear_time = q0 / net
            clear_note = (
                f"q0 / (P/H - lambda) = {q0:.1f} / ({planned_capacity:.1f}/{recommended:g} "
                f"- {demand_rate:.2f}) = {clear_time:.1f} min"
            )
        elif q0 <= 0:
            clear_time = 0.0
            clear_note = "no standing queue observed"
        else:
            # Cannot honestly produce a finite number here.
            clear_time = None
            clear_note = (
                "queue cannot be cleared at any headway while demand exceeds offered capacity: "
                f"{demand_rate:.2f} pax/min arriving vs {offered_service:.2f} pax/min boarding"
            )
            warnings.append(clear_note)

        # ---- Status -----------------------------------------------------------
        # A capacity failure can only be asserted when the demand measurement is worth
        # acting on. With a low-confidence rate, the schedule is held and the stop is
        # flagged for a better measurement instead of being declared over capacity.
        measured_capacity_failure = not capacity_ok
        if not capacity_ok and low_confidence:
            warnings.append(
                "The measured arrival rate implies demand above the stop's boarding capacity. "
                "Because that measurement is low-confidence, this is flagged for review rather "
                "than reported as a confirmed capacity failure."
            )
        if not capacity_ok and low_confidence:
            status = "MEASUREMENT_SUSPECT"
        elif not capacity_ok:
            status = "CAPACITY_FAILURE"
        elif q0 >= threshold or (utilization is not None and utilization >= 0.95):
            status = "SURGE"
        elif q0 >= threshold * 0.65 or (utilization is not None and utilization >= 0.75):
            status = "ELEVATED"
        else:
            status = "NORMAL"

        optimized_wait = wait_minutes(recommended)
        optimized_formula = (
            f"Welch: E[W] = H/2 x (1 + CV^2) with H={recommended:g} min, "
            f"CV={self.service['headway_cv']}"
        )
        reduction = ((baseline_wait - optimized_wait) / baseline_wait * 100.0) if baseline_wait > 0 else 0.0
        reduction = round(reduction, 1)
        wait_source = "modelled from schedule parameters (Welch's formula)"
        if reduction < 0:
            warnings.append(
                f"Recommended headway increases modelled wait by {abs(reduction):.1f}%; "
                "this is an energy/fleet trade-off, not a service improvement."
            )

        # ---- Narrative, generated from the computed values only ---------------
        reasoning = [
            f"Measured arrival rate at {stop['name']}: {demand_rate:.2f} pax/min "
            f"({'low-confidence' if measurement_confidence == 'low' else measurement_confidence} measurement).",
            f"Observed queue: {q0:.1f} pax ({queue_source}).",
            f"Vehicle {bus_type.replace('_', ' ')} offers {planned_capacity:.1f} usable seats "
            f"({capacity} seats less {reserve * 100:.0f}% reserve).",
            f"At headway {recommended:g} min the stop boards {offered_service:.2f} pax/min; "
            f"arrivals are {demand_rate:.2f} pax/min, so net clearance is {net:+.2f} pax/min.",
            f"Round trip is {cycle:.1f} min, so {buses_required} buses are needed "
            f"(baseline allocation {baseline_buses}).",
        ]
        if recommended_source == "off-peak spacing":
            reasoning.append(
                "Demand is below the scheduled service, so the headway is extended to avoid "
                "running empty buses. Modelled wait increases; that is the intended trade-off."
            )

        directive = (
            f"[{status}] {route['name']}: headway {default_headway:g} -> {recommended:g} min "
            f"({headway_change_pct:+.1f}%), {buses_required} buses "
            f"({extra_to_inject} added), modelled wait {baseline_wait:.1f} -> {optimized_wait:.1f} min "
            f"({reduction:+.1f}%)."
        )

        return DispatchRecommendation(
            stop_id=stop_id,
            stop_name=stop.get("name", stop_id),
            routes_serving=serving,
            primary_route=route_id,
            status=status,
            arrival_rate_pax_per_min=round(demand_rate, 2),
            current_queue_pax=round(q0, 1),
            current_queue_source=queue_source,
            default_headway_min=default_headway,
            recommended_headway_min=round(recommended, 1),
            recommended_headway_source=recommended_source,
            headway_change_pct=headway_change_pct,
            bus_type=bus_type,
            bus_capacity_pax=capacity,
            effective_capacity_pax=round(planned_capacity, 1),
            offered_service_pax_per_min=round(offered_service, 2),
            demand_rate_pax_per_min=round(demand_rate, 2),
            net_clearance_pax_per_min=round(net, 2) if capacity_ok else None,
            utilization=round(utilization, 3) if utilization is not None else None,
            capacity_ok=capacity_ok,
            baseline_wait_min=round(baseline_wait, 1),
            optimized_wait_min=round(optimized_wait, 1),
            wait_reduction_pct=reduction,
            wait_reduction_source=wait_source,
            baseline_formula=baseline_formula,
            optimized_formula=optimized_formula,
            cycle_time_min=round(cycle, 1),
            buses_required=buses_required,
            buses_assigned=baseline_buses + extra_to_inject,
            extra_buses_available=max_injectable,
            extra_buses_to_inject=extra_to_inject,
            fleet_shortfall=fleet_shortfall,
            fleet_available=max_service_buses,
            queue_clear_time_min=round(clear_time, 1) if clear_time is not None else None,
            queue_clear_note=clear_note,
            directive_summary=directive,
            reasoning_steps=reasoning,
            warnings=warnings,
        )
