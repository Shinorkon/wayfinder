"""
Recommendation wording tests.

Added after two real defects in this module: a missing import that only showed up at
runtime, and a sentence builder that produced duplicated and ungrammatical text. The
wordings are user-facing, so they are asserted rather than eyeballed.
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.intelligence.recommendation import (  # noqa: E402
    ACTION_ADD_BUS,
    ACTION_HOLD,
    ACTION_NONE,
    ACTION_REVIEW_DATA,
    ACTION_SHORTEN_HEADWAY,
    ACTION_SPACE_OUT,
    live_recommendation,
    plain_recommendation,
)

BASE_DISPATCH = {
    "status": "NORMAL",
    "capacity_ok": True,
    "current_queue_pax": 6.0,
    "arrival_rate_pax_per_min": 8.0,
    "default_headway_min": 15.0,
    "recommended_headway_min": 15.0,
    "buses_required": 5,
    "extra_buses_to_inject": 0,
    "fleet_shortfall": 0,
    "bus_type": "standard_electric",
}
MEASURED = {"kind": "measured", "confidence": "medium"}

LIVE = {
    "people_in_frame": 12,
    "people_waiting": 9,
    "arrival_rate_pax_per_min": 10.0,
    "segmentation": {},
}


def dispatch(**overrides):
    return {**BASE_DISPATCH, **overrides}


def test_hold_sentence_is_well_formed():
    result = plain_recommendation(dispatch(), MEASURED)
    assert result["action"] == ACTION_HOLD
    assert result["sentence"].startswith("6 people are waiting and 8.0 arrive per minute.")
    assert result["sentence"].endswith("The current 15-minute service handles that.")


def test_low_confidence_holds_schedule_and_names_the_reason():
    result = plain_recommendation(dispatch(), {"kind": "measured", "confidence": "low"})
    assert result["action"] == ACTION_REVIEW_DATA
    assert "too weak" in result["sentence"]


def test_shorten_headway_mentions_extra_buses():
    result = plain_recommendation(
        dispatch(recommended_headway_min=8.0, extra_buses_to_inject=2), MEASURED
    )
    assert result["action"] == ACTION_SHORTEN_HEADWAY
    assert "every 8 minutes instead of 15" in result["sentence"]
    assert "2 extra standard electric" in result["sentence"]


def test_fleet_shortfall_is_disclosed():
    result = plain_recommendation(
        dispatch(recommended_headway_min=8.0, extra_buses_to_inject=2, fleet_shortfall=1),
        MEASURED,
    )
    assert "cannot" in result["sentence"]


def test_over_capacity_says_rescheduling_is_not_enough():
    result = plain_recommendation(dispatch(capacity_ok=False), MEASURED)
    assert result["action"] == ACTION_ADD_BUS
    assert "needs more capacity" in result["sentence"]


def test_space_out_is_labelled_as_a_tradeoff():
    result = plain_recommendation(dispatch(recommended_headway_min=20.0), MEASURED)
    assert result["action"] == ACTION_SPACE_OUT
    assert "wait a little longer" in result["sentence"]


def test_no_footage_and_no_route_are_distinct():
    no_data = plain_recommendation({"status": "NO_DATA"}, {"kind": "unavailable"})
    assert no_data["action"] == ACTION_NONE
    assert "nothing to measure" in no_data["sentence"]

    no_route = plain_recommendation({"status": "NO_ROUTE"}, MEASURED)
    assert "no schedule can be adjusted" in no_route["sentence"]


def test_live_view_states_each_arrival_rate_exactly_once():
    """The live view must not restate the recorded average as if it were the live rate."""
    result = live_recommendation(LIVE, None, dispatch(), MEASURED)
    sentence = result["sentence"]

    assert sentence.count("10.0") == 1, sentence
    assert sentence.count("8.0") == 1, sentence
    assert "recorded analysis of this stop averaged 8.0" in sentence
    assert result["numbers"]["arrivals_per_min_displayed"] == 10.0
    assert result["numbers"]["arrivals_per_min_recorded_average"] == 8.0


def test_live_view_without_a_recorded_rate_does_not_invent_one():
    result = live_recommendation(LIVE, None, {"status": "NO_ROUTE"}, {"kind": "measured"})
    assert "averaged" not in result["sentence"]
    assert "unknown" not in result["sentence"]
    assert "no schedule can be adjusted" in result["sentence"]


def test_offline_view_leads_with_the_recorded_numbers():
    result = plain_recommendation(dispatch(), MEASURED)
    assert "camera is tracking" not in result["sentence"]
    assert result["numbers"]["arrivals_per_min"] == 8.0


def test_every_action_returns_a_non_empty_sentence():
    scenarios = [
        (dispatch(), MEASURED),
        (dispatch(), {"kind": "measured", "confidence": "low"}),
        (dispatch(capacity_ok=False), MEASURED),
        (dispatch(recommended_headway_min=8.0), MEASURED),
        (dispatch(recommended_headway_min=20.0), MEASURED),
        ({"status": "NO_DATA"}, {"kind": "unavailable"}),
        ({"status": "NO_ROUTE"}, {"kind": "measured"}),
    ]
    for dispatch_record, provenance in scenarios:
        result = plain_recommendation(dispatch_record, provenance)
        assert result["sentence"].strip()
        assert result["headline"].strip()
        assert result["action"]


def test_no_sentence_has_double_spaces_or_repeated_lead():
    """Guards the exact formatting defect seen during development."""
    for live in (None, LIVE):
        if live:
            result = live_recommendation(live, None, dispatch(), MEASURED)
        else:
            result = plain_recommendation(dispatch(), MEASURED)
        sentence = result["sentence"]
        assert "  " not in sentence, sentence
        assert "per minute. The recorded demand" not in sentence, sentence
        assert sentence.count("people are waiting") <= 1, sentence
