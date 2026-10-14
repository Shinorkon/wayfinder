"""
Wayfinder advisory layer (optional).

The authoritative output for any stop is produced by src/intelligence/dispatch_engine.py.
This module can add narrative commentary on top of that, and nothing else.

Rules enforced here:
  * Every figure given to the model is supplied by the deterministic engine. The model
    is instructed not to invent numbers, vehicles, depots or causes.
  * If no API key is configured, or the call fails, the returned narrative is explicitly
    labelled as coming from the deterministic engine - never from an LLM.
  * There is no hardcoded filler of the kind this file used to contain (invented delay
    figures and invented vehicle registrations). See tests/test_data_integrity.py,
    which fails the build if that style of content returns.  # integrity-test-allow
    That kind of text is indistinguishable from a hallucination and is not permitted.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import httpx

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

SYSTEM_PROMPT = """You are an advisory analyst for the RTL bus network in Greater Malé, Maldives.

You will be given a stop's computed state. Every number you receive was calculated by a
deterministic queueing model that has already been run. Your only job is to explain it to
an operations supervisor in plain language.

Hard rules:
1. Use ONLY the numbers provided. Never invent a figure, a vehicle, a depot, a registration,
   a road, a cause, or a statistic that is not in the input.
2. If a number is missing or marked low-confidence, say so plainly instead of estimating it.
3. If the input is insufficient to justify your claim, say that instead of filling the gap.
4. Keep it short: 2-4 sentences for the summary, at most 4 advisory points.
5. You are advisory only. Never describe an action as already dispatched, ordered, or done.

Respond with valid JSON matching exactly:
{
  "summary": "2-4 sentence plain-language explanation of the computed state",
  "advisory_points": ["short point", "..."],
  "data_caveats": ["any caveat about the data quality you were given"]
}"""


class DeepSeekDispatchAgent:
    """Adds advisory commentary to already-computed dispatch output."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")

    def set_api_key(self, api_key: str) -> None:
        self.api_key = api_key

    # -- public API --------------------------------------------------------

    def reason(
        self,
        stop_name: str,
        arrival_rate_pax_per_min: float,
        queue_pax: float,
        default_headway_min: float,
        recommended_headway_min: float,
        route_name: str,
        scenario: str,
        measured_rate: Optional[float] = None,
        confidence: str = "unknown",
        **extra: Any,
    ) -> Dict[str, Any]:
        payload = {
            "stop_name": stop_name,
            "route": route_name,
            "scenario": scenario,
            "arrival_rate_pax_per_min": round(arrival_rate_pax_per_min, 2),
            "measured_base_rate_pax_per_min": (
                round(measured_rate, 2) if measured_rate is not None else None
            ),
            "measurement_confidence": confidence,
            "queue_pax": round(queue_pax, 1),
            "default_headway_min": default_headway_min,
            "recommended_headway_min": recommended_headway_min,
        }
        payload.update(extra)

        if self.api_key:
            try:
                return self._call_api(payload)
            except Exception as exc:  # network, auth, malformed JSON - all non-fatal
                print(f"[advisory] DeepSeek call failed ({exc}); falling back to engine text.")
        return self._engine_text(payload)

    # -- implementations ---------------------------------------------------

    def _call_api(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, indent=2)},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }
        with httpx.Client(timeout=20.0) as client:
            resp = client.post(DEEPSEEK_API_URL, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        parsed["source"] = "deepseek-api-live"
        parsed["grounding"] = payload
        return parsed

    def _engine_text(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Deterministic narrative built only from the supplied numbers."""
        rate = payload["arrival_rate_pax_per_min"]
        queue = payload["queue_pax"]
        base = payload["default_headway_min"]
        rec = payload["recommended_headway_min"]
        conf = payload["measurement_confidence"]
        measured = payload["measured_base_rate_pax_per_min"]

        caveats = []
        if conf == "low":
            caveats.append(
                "The arrival rate for this stop is a low-confidence measurement, so the "
                "headway recommendation is indicative only."
            )
        if conf == "none":
            caveats.append("No arrival rate has been measured for this stop.")
        caveats.append(
            "Arrival rates are measured from public reference footage of other cities, "
            "scaled by a declared scenario factor."
        )

        direction = "unchanged"
        if rec < base:
            direction = "shortened"
        elif rec > base:
            direction = "lengthened"

        summary = (
            f"{payload['stop_name']} is modelled to receive {rate:.2f} pax/min with an observed "
            f"standing queue of {queue:.1f} passengers. The current timetable headway is {base:g} minutes "
            f"and the computed headway is {rec:g} minutes ({direction}). "
            f"Arrival rate provenance: {conf}-confidence measurement"
            + (f" of {measured:.2f} pax/min before scenario scaling." if measured is not None else ".")
        )

        points = [
            f"Demand used by the model: {rate:.2f} pax/min.",
            f"Standing queue used by the model: {queue:.1f} pax.",
            f"Headway change under consideration: {base:g} -> {rec:g} min.",
            "This text is generated deterministically from the computed state; configure a"
            " DeepSeek API key for language-model commentary.",
        ]
        return {
            "source": "deterministic-engine-text",
            "summary": summary,
            "advisory_points": points,
            "data_caveats": caveats,
            "grounding": payload,
        }
