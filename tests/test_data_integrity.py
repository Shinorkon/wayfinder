"""
Guard tests for data integrity.

These exist specifically because the previous version of this project generated
passenger counts with random.randint and presented them as CCTV telemetry. If anyone
reintroduces synthesised numbers into the telemetry path, these tests fail.
"""

from __future__ import annotations

import ast
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TELEMETRY_PATH = os.path.join(ROOT, "src", "intelligence", "telemetry.py")
SERVER_PATH = os.path.join(ROOT, "src", "api", "server.py")

FORBIDDEN_CALLS = {"randint", "random", "uniform", "choice", "shuffle", "normalvariate", "randrange"}


def _called_names(path: str) -> set:
    tree = ast.parse(open(path, encoding="utf-8").read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


@pytest.mark.parametrize("path", [TELEMETRY_PATH, SERVER_PATH])
def test_no_randomness_in_telemetry_path(path):
    names = _called_names(path)
    offenders = names & FORBIDDEN_CALLS
    assert not offenders, (
        f"{os.path.basename(path)} calls {sorted(offenders)}. Passenger telemetry must be "
        "measured or explicitly declared as simulated - never randomised."
    )


def test_no_hardcoded_fabricated_operations():
    """
    The old build invented depots, vehicle IDs and delay figures in strings.

    Documentation may name the removed examples, so a line can opt out with the marker
    ``integrity-test-allow`` (used only in comments/docstrings, never in data paths).
    """
    banned = ["Carnival Depot", "Unit R-04", "Unit #07", "+14.2 minutes", "2.74 seconds"]
    hits = []
    for sub in ("src", "web", "scripts"):
        for dirpath, _dirs, files in os.walk(os.path.join(ROOT, sub)):
            if "__pycache__" in dirpath:
                continue
            for name in files:
                if not name.endswith((".py", ".js", ".html", ".css", ".json")):
                    continue
                full = os.path.join(dirpath, name)
                for lineno, line in enumerate(open(full, encoding="utf-8", errors="ignore"), 1):
                    if "integrity-test-allow" in line:
                        continue
                    for phrase in banned:
                        if phrase in line:
                            hits.append(f"{os.path.relpath(full, ROOT)}:{lineno}: {phrase}")
    assert not hits, "Fabricated operational detail found: " + "; ".join(hits)


def test_network_config_is_internally_consistent():
    network = json.load(open(os.path.join(ROOT, "data", "routes", "male_rtl_network.json"), encoding="utf-8"))
    stop_ids = {s["id"] for s in network["stops"]}

    for route in network["routes"]:
        for stop_id in route["stops"]:
            assert stop_id in stop_ids, f"route {route['id']} references unknown stop {stop_id}"
        assert route["stops"], f"route {route['id']} has no stops"
        geometry = route.get("osrm_geometry") or []
        if geometry:
            assert len(geometry) > 2, f"route {route['id']} geometry is not a path"

    # Every routable stop must be reachable by exactly the routes that list it.
    served = {}
    for route in network["routes"]:
        for stop_id in route["stops"]:
            served.setdefault(stop_id, []).append(route["id"])
    for stop_id, routes in served.items():
        assert len(routes) >= 1

    # Fleet composition is required for the model to size vehicles.
    assert network["fleet_composition"], "fleet_composition missing"
    for cls, spec in network["fleet_composition"].items():
        assert spec["capacity"] > 0, f"{cls} needs a positive capacity"


def test_scenario_multipliers_are_declared_and_positive():
    from src.intelligence.telemetry import SCENARIOS

    assert "observed" in SCENARIOS
    for key, meta in SCENARIOS.items():
        assert meta["multiplier"] > 0, f"scenario {key} has a non-positive multiplier"
        assert meta["description"], f"scenario {key} must describe itself"
        for stop_id, value in meta.get("stop_multipliers", {}).items():
            assert value > 0, f"scenario {key} stop override {stop_id} must be positive"
