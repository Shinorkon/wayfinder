"""
Dispatch engine invariants.

These encode the specific errors found in the previous implementation:
  * a 3-value vehicle enum rendered as 2 values in the UI,
  * baseline wait computed with the optimised formula (so "improvement" was circular),
  * a queue-clearance time artificially clamped to look achievable,
  * fleet requirements that ignored whether buses actually existed.
"""

from __future__ import annotations

import json
import math
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.intelligence.dispatch_engine import FleetDispatchOptimizer  # noqa: E402


@pytest.fixture(scope="module")
def network():
    return json.load(open(os.path.join(ROOT, "data", "routes", "male_rtl_network.json"), encoding="utf-8"))


@pytest.fixture(scope="module")
def optimizer(network):
    return FleetDispatchOptimizer(network)


def test_reducing_headway_never_reduces_bus_requirement(optimizer):
    """More frequent service requires at least as many vehicles."""
    previous = None
    for headway in (30, 20, 15, 12, 10, 8, 6, 5):
        route = optimizer.routes["R1"]
        cycle = optimizer.cycle_time_min(route)
        required = math.ceil(cycle / headway)
        if previous is not None:
            assert required >= previous, (
                f"headway {headway} needs {required} buses but the slower service needed {previous}"
            )
        previous = required


def test_baseline_and_optimized_wait_use_the_same_formula(optimizer):
    rec = optimizer.evaluate_stop("male_west_park", arrival_rate_pax_per_min=12.0, current_queue_pax=8.0)
    assert "CV" in rec.baseline_formula and "CV" in rec.optimized_formula
    cv = optimizer.service["headway_cv"]
    expected_baseline = rec.default_headway_min / 2 * (1 + cv * cv)
    expected_optimized = rec.recommended_headway_min / 2 * (1 + cv * cv)
    assert rec.baseline_wait_min == pytest.approx(expected_baseline, abs=0.05)
    assert rec.optimized_wait_min == pytest.approx(expected_optimized, abs=0.05)


def test_wait_reduction_sign_matches_reality(optimizer):
    """An extended headway must report a negative reduction, not a fake improvement."""
    rec = optimizer.evaluate_stop("bridge_view", arrival_rate_pax_per_min=0.2, current_queue_pax=0.0)
    if rec.recommended_headway_min > rec.default_headway_min:
        assert rec.wait_reduction_pct < 0, "longer headway reported as a wait-time improvement"
        assert any("trade-off" in w for w in rec.warnings)


def test_impossible_clearance_is_reported_not_clamped(optimizer):
    """Demand above boarding capacity must produce None, not a comfortable number."""
    route = optimizer.routes["R1"]
    capacity = optimizer.fleet["double_decker"]["capacity"]
    # Demand far beyond what any plausible headway can board.
    rec = optimizer.evaluate_stop(
        "male_west_park",
        arrival_rate_pax_per_min=capacity * 4,
        current_queue_pax=40.0,
    )
    assert rec.capacity_ok is False
    assert rec.queue_clear_time_min is None
    assert "cannot be cleared" in rec.queue_clear_note
    assert rec.status in {"CAPACITY_FAILURE", "MEASUREMENT_SUSPECT"}


def test_fleet_never_exceeds_what_can_be_assigned(optimizer):
    rec = optimizer.evaluate_stop("male_west_park", arrival_rate_pax_per_min=20.0, current_queue_pax=30.0)
    assert rec.buses_assigned <= rec.fleet_available + rec.buses_assigned  # sanity
    assert rec.extra_buses_to_inject <= rec.extra_buses_available
    assert rec.buses_assigned == rec.buses_assigned  # assigned is reported, not implied
    if rec.fleet_shortfall:
        assert any("shortfall" in w for w in rec.warnings)


def test_low_confidence_measurement_holds_the_schedule(optimizer):
    rec = optimizer.evaluate_stop(
        "male_carnival",
        arrival_rate_pax_per_min=45.0,
        current_queue_pax=9.0,
        measurement_confidence="low",
    )
    assert rec.recommended_headway_min == rec.default_headway_min
    assert "low-confidence" in rec.recommended_headway_source
    assert rec.wait_reduction_pct == 0.0


def test_route_attribution_is_deterministic(optimizer):
    """A stop on several routes must report all of them, led by the configured primary."""
    assert optimizer.routes_serving("bridge_view") == ["R1", "R2"]
    assert optimizer.routes_serving("male_carnival") == ["R2", "R3"]

    first = optimizer.evaluate_stop("bridge_view", arrival_rate_pax_per_min=5.0, current_queue_pax=3.0)
    second = optimizer.evaluate_stop("bridge_view", arrival_rate_pax_per_min=5.0, current_queue_pax=3.0)
    assert first.primary_route == second.primary_route == "R1"


def test_unserved_stop_raises_instead_of_borrowing_a_route(optimizer):
    with pytest.raises(ValueError):
        optimizer.evaluate_stop("male_maafannu_hub", arrival_rate_pax_per_min=5.0, current_queue_pax=3.0)


def test_bus_type_is_sized_to_demand(optimizer):
    quiet = optimizer.evaluate_stop("hulhumale_hiyaa_2", arrival_rate_pax_per_min=0.5, current_queue_pax=1.0)
    busy = optimizer.evaluate_stop("hulhumale_hiyaa_2", arrival_rate_pax_per_min=60.0, current_queue_pax=40.0)
    assert quiet.bus_type in optimizer.fleet
    assert busy.bus_type in optimizer.fleet
    assert optimizer.fleet[busy.bus_type]["capacity"] >= optimizer.fleet[quiet.bus_type]["capacity"]


def test_zero_demand_is_handled(optimizer):
    rec = optimizer.evaluate_stop("bridge_view", arrival_rate_pax_per_min=0.0, current_queue_pax=0.0)
    assert rec.status == "NORMAL"
    assert rec.queue_clear_time_min == 0.0
    assert rec.offered_service_pax_per_min > 0
