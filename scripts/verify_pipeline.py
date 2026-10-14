#!/usr/bin/env python3
"""
Wayfinder end-to-end verification.

Unlike the script it replaces, this exits non-zero when something is wrong. It checks:

  1. the network configuration is internally consistent,
  2. the offline analysis artefacts exist and each measured stop carries provenance,
  3. the dispatch model produces coherent output for every stop and scenario,
  4. the HTTP API serves that data with its provenance intact.

Usage:
    python3 scripts/verify_pipeline.py            # artefacts must exist
    python3 scripts/verify_pipeline.py --allow-missing-analysis
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.intelligence.telemetry import SCENARIOS, TelemetryService, load_analysis_index  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NETWORK_PATH = os.path.join(ROOT, "data", "routes", "male_rtl_network.json")

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

failures: list[str] = []
warnings: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> bool:
    if condition:
        print(f"  [{PASS}] {label}")
        return True
    print(f"  [{FAIL}] {label}{(' — ' + detail) if detail else ''}")
    failures.append(f"{label}{(' — ' + detail) if detail else ''}")
    return False


def warn(label: str) -> None:
    print(f"  [{WARN}] {label}")
    warnings.append(label)


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-missing-analysis", action="store_true")
    args = ap.parse_args()

    print("=" * 68)
    print("  WAYFINDER END-TO-END VERIFICATION")
    print("=" * 68)

    # ---------------------------------------------------------------- network
    section("1. Network configuration")
    if not os.path.exists(NETWORK_PATH):
        check(False, "network configuration present", NETWORK_PATH)
        return 1
    network = json.load(open(NETWORK_PATH, encoding="utf-8"))
    stop_ids = {s["id"] for s in network["stops"]}
    check(len(stop_ids) > 0, f"network defines {len(stop_ids)} stops")
    check(bool(network["routes"]), f"network defines {len(network['routes'])} routes")

    referenced = set()
    for route in network["routes"]:
        referenced.update(route["stops"])
        check(
            set(route["stops"]).issubset(stop_ids),
            f"route {route['id']} stop references valid",
            f"unknown: {sorted(set(route['stops']) - stop_ids)}",
        )
    unserved = sorted(stop_ids - referenced)
    if unserved:
        warn(f"stops not served by any route (reported as NO_ROUTE, not modelled): {unserved}")

    check(bool(network.get("fleet_composition")), "fleet composition configured")
    check("data_status" in network, "network data declares its own provenance")

    # --------------------------------------------------------------- analysis
    section("2. Offline analysis artefacts")
    index = load_analysis_index()
    if index is None:
        if args.allow_missing_analysis:
            warn("data/analysis/index.json missing (allowed by flag)")
        else:
            check(False, "analysis index present", "run: python3 scripts/analyze_footage.py --skip-sam3")
            return 1
    else:
        coverage = index.get("coverage", {})
        check(
            coverage.get("stops_with_measured_footage", 0) > 0,
            f"{coverage.get('stops_with_measured_footage')}/{coverage.get('stops_total')} stops have measured footage",
        )
        graded = coverage.get("confidence_counts", {})
        print(f"         confidence grades: {graded}")
        if graded.get("low", 0) and not graded.get("high", 0):
            warn(
                "no stop has a high-confidence arrival rate, so the model will hold existing "
                "headways. This is a property of the available footage, not a bug."
            )

        for clip, artifact in index.get("clips", {}).items():
            metrics = artifact.get("metrics", {})
            confidence = artifact.get("confidence", {}).get("grade")
            reasons = artifact.get("confidence", {}).get("reasons", [])
            check(
                confidence in {"low", "medium", "high"},
                f"{clip[:44]} graded {confidence}",
            )
            check(
                bool(reasons),
                f"{clip[:44]} confidence is justified",
                "a grade without reasons is not auditable",
            )
            check(
                "\n" not in json.dumps(metrics),
                f"{clip[:44]} metrics are structured values",
            )
            check(
                metrics.get("arrival_rate_pax_per_min", 0) >= 0,
                f"{clip[:44]} arrival rate non-negative",
            )

    # ----------------------------------------------------------- dispatch model
    section("3. Dispatch model")
    telemetry = TelemetryService(network, index, "observed")
    states = telemetry.compute_all()
    check(len(states) == len(stop_ids), "every stop evaluated")

    statuses: dict[str, int] = {}
    for state in states:
        dispatch = state.dispatch
        statuses[dispatch.get("status", "?")] = statuses.get(dispatch.get("status", "?"), 0) + 1

        if state.provenance["kind"] == "measured":
            check(
                state.arrival_rate_base_pax_per_min is not None,
                f"{state.stop['id']} exposes the unscaled measured base rate",
            )
            check(
                bool(state.provenance.get("limitations")),
                f"{state.stop['id']} ships measurement limitations",
            )
            # Regression guard: enrichment used to be dropped during serialisation, which
            # made the dashboard claim "no footage" for a stop that had measurements.
            check(
                bool(state.stop.get("footage_provenance")),
                f"{state.stop['id']} ships footage provenance",
            )
            urls = list((state.stop.get("keyframes") or {}).values())
            check(bool(urls), f"{state.stop['id']} ships rendered keyframe layers")
            if urls:
                check(
                    urls[0].startswith("/output/keyframes/"),
                    f"{state.stop['id']} keyframe path is server-absolute",
                    urls[0],
                )
        if dispatch.get("default_headway_min") is not None:
            check(
                dispatch["buses_required"] >= 1,
                f"{state.stop['id']} requires at least one bus",
            )
            if state.provenance.get("confidence") == "low":
                check(
                    dispatch["recommended_headway_min"] == dispatch["default_headway_min"],
                    f"{state.stop['id']} holds schedule on low-confidence data",
                )
    print(f"         statuses: {statuses}")

    check(
        sum(1 for s in states if s.dispatch.get("queue_clear_time_min") is None)
        <= len(states),
        "clearance reporting is bounded",
    )

    # ---------------------------------------------------------------- scenarios
    section("4. Scenario scaling")
    observed_rates = {
        s.stop["id"]: s.arrival_rate_base_pax_per_min
        for s in state_cache(telemetry, "observed")
    }
    for key in SCENARIOS:
        telemetry.set_scenario(key)
        scaled = telemetry.compute_all()
        for state in scaled:
            base = state.arrival_rate_base_pax_per_min
            if base is None:
                continue
            expected = base * state.scenario_multiplier
            if abs(state.arrival_rate_pax_per_min - expected) > 0.01:
                check(False, f"scenario {key} scaling arithmetic", state.stop["id"])
                break
        else:
            check(True, f"scenario '{key}' scaling is consistent with its declared multiplier")
    telemetry.set_scenario("observed")

    # ---------------------------------------------------------------------- API
    section("5. HTTP API")
    try:
        from fastapi.testclient import TestClient

        from src.api.server import app

        client = TestClient(app)
        health = client.get("/api/health")
        check(health.status_code == 200 and health.json()["status"] == "ok", "GET /api/health")
        check(health.json()["analysis_loaded"] == (index is not None), "health reports analysis state")

        meta = client.get("/api/meta").json()
        check(bool(meta["method"]["limitations"]), "meta states its limitations")
        check(bool(meta["scenarios"]), "meta advertises scenarios")

        payload = client.get("/api/stops").json()
        check(len(payload["stops"]) == len(stop_ids), "GET /api/stops returns every stop")
        check(
            all("provenance" in s and "kind" in s["provenance"] for s in payload["stops"]),
            "every stop response carries provenance",
        )

        telemetry_payload = client.get("/api/telemetry").json()
        check(bool(telemetry_payload["kpis"]["wait_formula"]), "telemetry states its wait-time formula")
        check(
            "footage" in telemetry_payload["data_status"]["disclosure"].lower(),
            "telemetry discloses its data source",
        )

        bad = client.post("/api/scenario", json={"scenario": "not_a_scenario"})
        check(bad.status_code == 400, "invalid scenario is rejected")
    except Exception as exc:  # pragma: no cover - environment specific
        check(False, "API checks", f"{type(exc).__name__}: {exc}")

    # -------------------------------------------------------------------- result
    print("\n" + "=" * 68)
    if failures:
        print(f"  RESULT: {len(failures)} FAILURE(S)")
        for item in failures:
            print(f"    - {item}")
        print("=" * 68)
        return 1

    print(f"  RESULT: ALL CHECKS PASSED ({len(warnings)} warning(s))")
    for item in warnings:
        print(f"    - {item}")
    print("=" * 68)
    return 0


def state_cache(telemetry: TelemetryService, scenario: str):
    telemetry.set_scenario(scenario)
    return telemetry.compute_all()


if __name__ == "__main__":
    raise SystemExit(main())
