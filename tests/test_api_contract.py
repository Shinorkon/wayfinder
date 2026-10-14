"""
API contract tests.

The dashboard's honesty depends on these guarantees:
  * every stop response separates measured values from modelled ones,
  * unavailable data is reported as unavailable rather than filled in,
  * scenario scaling is declared, with the unscaled base rate still exposed.
"""

from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.api.server import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_and_meta(client):
    assert client.get("/api/health").json()["status"] == "ok"

    meta = client.get("/api/meta").json()
    assert meta["analysis"]["coverage"]["stops_total"] > 0
    assert meta["scenarios"], "scenarios must be advertised by the API, not hardcoded in the UI"
    assert meta["method"]["limitations"], "the API must state its own limitations"
    assert "not official" in meta["disclaimer"].lower() or "official" in meta["disclaimer"].lower()


def test_every_stop_carries_provenance(client):
    payload = client.get("/api/stops").json()
    assert payload["stops"], "no stops returned"

    for stop in payload["stops"]:
        provenance = stop["provenance"]
        assert provenance["kind"] in {"measured", "unavailable"}
        if provenance["kind"] == "measured":
            assert provenance["confidence"] in {"high", "medium", "low"}
            assert provenance["limitations"], "a measurement must ship with its limitations"
            assert provenance["analysis_method"]
            # Measured stops must expose an unscaled base rate as well as the model input.
            assert stop["arrival_rate_base_pax_per_min"] is not None
            assert stop["queue_baseline_measured"] is not None
        else:
            assert stop["arrival_rate_pax_per_min"] in (0.0, None)
            assert stop["dispatch"]["status"] in {"NO_DATA", "NO_ROUTE"}


def test_measured_stops_expose_footage_provenance_and_overlays(client):
    """
    A stop with analysed footage must ship its footage attribution and rendered layers.

    Regression guard: these fields used to be dropped while serialising, which left the
    UI claiming "no footage" for a stop that had measurements.
    """
    payload = client.get("/api/stops").json()
    measured = [s for s in payload["stops"] if s["provenance"]["kind"] == "measured"]
    assert measured, "no measured stops to check"

    for stop in measured:
        provenance = stop["footage_provenance"]
        assert provenance, f"{stop['id']} is measured but ships no footage provenance"
        assert provenance.get("source"), f"{stop['id']} footage has no source"
        assert provenance.get("license"), f"{stop['id']} footage has no licence"
        assert provenance.get("disclosure"), f"{stop['id']} footage has no disclosure"
        assert stop["keyframes"], f"{stop['id']} is measured but has no keyframe layers"
        for layer, url in stop["keyframes"].items():
            assert url.startswith("/output/keyframes/"), (
                f"{stop['id']} keyframe url '{url}' is not server-absolute, so the client "
                "cannot resolve it reliably"
            )


def test_unknown_stop_is_404(client):
    assert client.get("/api/stops/nope_not_here").status_code == 404


def test_scenario_scaling_is_declared(client):
    observed = client.get("/api/stops").json()
    base_by_stop = {s["id"]: s["arrival_rate_base_pax_per_min"] for s in observed["stops"]}

    response = client.post("/api/scenario", json={"scenario": "ferry_arrival"})
    assert response.status_code == 200
    scaled = client.get("/api/stops").json()

    assert scaled["scenario"]["simulated"] is True
    changed = False
    for stop in scaled["stops"]:
        base = base_by_stop[stop["id"]]
        if base is None:
            continue
        assert stop["arrival_rate_base_pax_per_min"] == pytest.approx(base)
        if stop["scenario_multiplier"] != 1.0:
            changed = True
            assert stop["arrival_rate_pax_per_min"] == pytest.approx(base * stop["scenario_multiplier"], rel=1e-3)
    assert changed, "ferry scenario applied no multiplier anywhere"

    # Restore for other tests.
    client.post("/api/scenario", json={"scenario": "observed"})


def test_invalid_scenario_is_rejected(client):
    assert client.post("/api/scenario", json={"scenario": "make_it_busy"}).status_code == 400


def test_telemetry_declares_what_is_modelled(client):
    telemetry = client.get("/api/telemetry").json()
    status = telemetry["data_status"]
    assert status["wait_times_modelled"] is True
    assert isinstance(status["arrival_rates_simulated"], bool)
    assert "footage" in status["disclosure"].lower()

    kpis = telemetry["kpis"]
    assert kpis["buses_required_total"] >= 0
    assert kpis["buses_required_total"] <= kpis["buses_configured_total"] * 3
    assert kpis["wait_formula"]
    # Fleet is summed per route, so it can never exceed the sum of per-route maxima.
    assert sum(kpis["buses_required_by_route"].values()) == kpis["buses_required_total"]


def test_advisory_is_optional_and_grounded(client):
    stop_id = client.get("/api/stops").json()["stops"][0]["id"]
    payload = client.post("/api/advisory", json={"stop_id": stop_id}).json()

    assert "engine_output" in payload
    assert payload["engine_output"]["directive_summary"]
    if not payload["available"]:
        # No API key configured: must say so rather than pretending an LLM answered.
        assert "No DeepSeek API key" in payload["reason"]


def test_route_geometry_endpoint(client):
    payload = client.get("/api/routes/R1/geometry").json()
    assert payload["id"] == "R1"
    assert len(payload["geometry"]) > 10
    assert client.get("/api/routes/ZZ/geometry").status_code == 404
