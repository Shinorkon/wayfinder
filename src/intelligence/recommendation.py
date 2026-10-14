"""
Plain-language recommendations.

The dashboard is meant to be read by an operator in a few seconds, so this module
converts the dispatch model's numbers into one sentence and one action.

It does not compute anything new and it does not soften anything: every statement
here is a restatement of a value that src/intelligence/dispatch_engine.py already
produced. Where the model reported a capacity failure, a fleet shortfall, or a
low-confidence measurement, the sentence says so.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

ACTION_HOLD = "hold"
ACTION_ADD_BUS = "add_bus"
ACTION_SHORTEN_HEADWAY = "shorten_headway"
ACTION_SPACE_OUT = "space_out"
ACTION_REVIEW_DATA = "review_data"
ACTION_NONE = "none"


def _fmt(value: Optional[float], digits: int = 0, suffix: str = "") -> str:
    if value is None:
        return "unknown"
    return f"{value:.{digits}f}{suffix}"


def _advice(dispatch: Dict[str, Any], provenance: Dict[str, Any]) -> Dict[str, Any]:
    """
    The advice clause and headline for one stop, with no leading observation.

    Kept separate so the offline view and the live view can each introduce it with the
    numbers they actually have. Returns sentence containing only the advice.
    """
    status = dispatch.get("status", "UNKNOWN")
    confidence = provenance.get("confidence", "none")

    if status == "NO_DATA":
        return {
            "headline": "No data",
            "sentence": "This stop has no analysed footage, so there is nothing to measure yet.",
            "action": ACTION_NONE,
            "numbers": {},
        }

    if status == "NO_ROUTE":
        return {
            "headline": "No route",
            "sentence": "No configured bus route serves this stop, so no schedule can be adjusted.",
            "action": ACTION_NONE,
            "numbers": {},
        }

    queue = dispatch.get("current_queue_pax")
    rate = dispatch.get("arrival_rate_pax_per_min")
    base = dispatch.get("default_headway_min")
    rec = dispatch.get("recommended_headway_min")
    buses_needed = dispatch.get("buses_required")
    extra = dispatch.get("extra_buses_to_inject") or 0
    shortfall = dispatch.get("fleet_shortfall") or 0
    bus_type = (dispatch.get("bus_type") or "").replace("_", " ")

    numbers = {
        "waiting_passengers": queue,
        "arrivals_per_min": rate,
        "headway_now_min": base,
        "headway_recommended_min": rec,
        "buses_required": buses_needed,
    }

    # Data quality first: an unevaluable measurement outranks any scheduling advice.
    if confidence == "low" or status == "MEASUREMENT_SUSPECT":
        return {
            "headline": "Keep schedule (weak data)",
            "sentence": (
                "That measurement is too weak to change the timetable safely, so the schedule "
                "is left as it is. More footage of this stop would settle it."
            ),
            "action": ACTION_REVIEW_DATA,
            "numbers": numbers,
        }

    if not dispatch.get("capacity_ok", True):
        return {
            "headline": "Over capacity",
            "sentence": (
                f"More people are arriving than the buses can pick up at the current "
                f"{_fmt(base, 0)}-minute headway. Rescheduling alone cannot fix this; the stop "
                "needs more capacity."
            ),
            "action": ACTION_ADD_BUS,
            "numbers": numbers,
        }

    if rec is not None and base is not None and rec < base - 0.5:
        sentence = f"Run buses every {_fmt(rec, 0)} minutes instead of {_fmt(base, 0)}"
        if extra:
            sentence += f", which needs {extra} extra {bus_type or 'bus'}{'es' if extra > 1 else ''}"
        sentence += "."
        if shortfall:
            sentence += (
                f" Only {buses_needed - shortfall} buses are available, so service cannot "
                "fully meet this."
            )
        return {
            "headline": f"Run every {_fmt(rec, 0)} min",
            "sentence": sentence,
            "action": ACTION_SHORTEN_HEADWAY,
            "numbers": numbers,
        }

    if rec is not None and base is not None and rec > base + 0.5:
        return {
            "headline": f"Space out to {_fmt(rec, 0)} min",
            "sentence": (
                f"Buses can run every {_fmt(rec, 0)} minutes instead of {_fmt(base, 0)} to save "
                "fuel. Passengers will wait a little longer."
            ),
            "action": ACTION_SPACE_OUT,
            "numbers": numbers,
        }

    return {
        "headline": "No change needed",
        "sentence": f"The current {_fmt(base, 0)}-minute service handles that.",
        "action": ACTION_HOLD,
        "numbers": numbers,
    }


def plain_recommendation(dispatch: Dict[str, Any], provenance: Dict[str, Any]) -> Dict[str, Any]:
    """Advice for one stop, introduced by what the recorded analysis measured."""
    result = _advice(dispatch, provenance)
    numbers = result.get("numbers") or {}

    if result["action"] == ACTION_NONE:
        return result

    queue = numbers.get("waiting_passengers")
    rate = numbers.get("arrivals_per_min")
    lead = f"{_fmt(queue, 0)} people are waiting and {_fmt(rate, 1)} arrive per minute."
    return {**result, "sentence": f"{lead} {result['sentence']}"}


def live_recommendation(
    live: Dict[str, Any],
    measured: Optional[Dict[str, Any]],
    dispatch: Dict[str, Any],
    provenance: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merges what the monitor is seeing right now with the scheduled recommendation.

    The live counts come from the continuous monitor; the headway advice comes from the
    offline measurement, because a headway decision needs minutes of data, not seconds.
    The two are never blended into one number.
    """
    advice = _advice(dispatch, provenance)
    waiting_now = live.get("people_waiting") if live else None
    in_frame = live.get("people_in_frame") if live else None
    rate_now = live.get("arrival_rate_pax_per_min") if live else None
    recorded = (advice.get("numbers") or {}).get("arrivals_per_min")

    numbers = dict(advice.get("numbers") or {})
    numbers.update(
        {
            "waiting_now": waiting_now,
            "people_in_frame_now": in_frame,
            "arrivals_per_min_now": rate_now,
            # The live rate leads the dashboard when the monitor is producing one; the
            # recorded average is kept separately so the two are never conflated.
            "arrivals_per_min_displayed": rate_now if rate_now else recorded,
            "arrivals_per_min_recorded_average": recorded,
        }
    )

    if advice["action"] == ACTION_NONE:
        # The advice is a data-availability statement; there is nothing to introduce.
        sentence = advice["sentence"]
        if live:
            sentence = (
                f"The camera is tracking {_fmt(in_frame, 0)} people, "
                f"{_fmt(waiting_now, 0)} of them waiting. {sentence}"
            )
        return {**advice, "sentence": sentence, "numbers": numbers}

    # Lead paragraph: what is happening now (live) or what was recorded (offline).
    if live:
        lead = (
            f"The camera is tracking {_fmt(in_frame, 0)} people, "
            f"{_fmt(waiting_now, 0)} of them waiting"
        )
        if rate_now:
            lead += f", with about {_fmt(rate_now, 1)} arriving per minute"
        lead += "."
        if recorded:
            lead += (
                f" The recorded analysis of this stop averaged {_fmt(recorded, 1)} per minute."
            )
    else:
        lead = (
            f"{_fmt(numbers.get('waiting_passengers'), 0)} people are waiting and "
            f"{_fmt(recorded, 1)} arrive per minute."
        )

    return {
        "headline": advice["headline"],
        "sentence": f"{lead} {advice['sentence']}",
        "action": advice["action"],
        "numbers": numbers,
    }


    return {
        "headline": base["headline"],
        "sentence": sentence,
        "action": base["action"],
        "numbers": numbers,
    }
