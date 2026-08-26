"""Pytest suite for core interoperability logic."""
from __future__ import annotations

import pandas as pd
import pytest

from config import CRITERIA, READINESS_HIGH_MIN, READINESS_MEDIUM_MIN
from core.barriers import classify_barriers
from core.integration import classify_integration_readiness
from core.scoring import apply_scoring_rules, score_api_availability
from core.validation import (
    coerce_criteria_columns,
    validate_systems_df,
    weighted_readiness_class,
)
from data.loaders import load_systems
from data.pipeline import build_framework

# Post-audit values (2026-08): SYS01, SYS02, SYS07, SYS14 were corrected by the
# scorecard-validation audit — see their "SCORING AUDIT" notes in
# data/systems.csv and CONTEXT.md's "Scorecard scoring conventions" section
# for the evidence-cited reasoning behind each change. SYS14's readiness_class
# moving Medium -> Low is an intended consequence of that correction, not a
# regression.
EXPECTED_SCORES = {
    "SYS01": (17.0, "High"),
    "SYS02": (12.0, "Medium"),
    "SYS03": (8.0, "Low"),
    "SYS04": (8.0, "Low"),
    "SYS05": (9.0, "Low"),
    "SYS06": (20.0, "High"),
    "SYS07": (6.0, "Low"),
    "SYS08": (13.0, "Medium"),
    "SYS09": (11.0, "Medium"),
    "SYS10": (7.0, "Low"),
    "SYS11": (9.0, "Low"),
    "SYS12": (18.0, "High"),
    "SYS13": (11.0, "Medium"),
    "SYS14": (8.0, "Low"),
}


def test_readiness_thresholds() -> None:
    assert weighted_readiness_class(15) == "High"
    assert weighted_readiness_class(14) == "Medium"
    assert weighted_readiness_class(10) == "Medium"
    assert weighted_readiness_class(9) == "Low"
    assert weighted_readiness_class(READINESS_HIGH_MIN) == "High"
    assert weighted_readiness_class(READINESS_MEDIUM_MIN) == "Medium"


def test_invalid_score_coercion() -> None:
    df = pd.DataFrame({
        "system_id": ["T1"],
        "system_name": ["Test"],
        **{c: [1] for c in CRITERIA},
    })
    df.loc[0, "api_availability"] = 5
    result = validate_systems_df(df)
    assert any("invalid value" in e for e in result.errors)


def test_coerce_criteria_valid() -> None:
    df = pd.DataFrame({
        "system_id": ["T1"],
        "system_name": ["T"],
        "api_availability": [2],
    })
    for c in CRITERIA[1:]:
        df[c] = 1
    out, msgs = coerce_criteria_columns(df)
    assert not msgs
    assert out.loc[0, "api_availability"] == 2


def test_api_availability_vectornet_scores_two() -> None:
    row = pd.Series({
        "access_method": "Open data portal on GBIF infrastructure; FAIR-compliant",
        "data_type": "Tick occurrence records",
        "notes": "VectorNet portal",
    })
    assert score_api_availability(row) == 2


def test_api_availability_no_api_scores_zero() -> None:
    row = pd.Series({
        "access_method": "Static map publications; no API",
        "data_type": "",
        "notes": "",
    })
    assert score_api_availability(row) == 0


def test_barriers_produces_five_dimensions() -> None:
    snap = build_framework()
    cols = ["technical_barrier", "semantic_barrier", "legal_barrier", "governance_barrier", "accessibility_barrier"]

    for c in cols:
        assert c in snap.barriers.columns


def test_integration_ranking_order() -> None:
    snap = build_framework()
    assert "rank" in snap.integration.columns
    assert snap.integration["rank"].tolist() == list(range(1, len(snap.integration) + 1))


@pytest.mark.parametrize(
    "sid,expected_total,expected_class",
    [(sid, total, cls) for sid, (total, cls) in EXPECTED_SCORES.items()],
)
def test_system_scores_match_inventory(sid, expected_total, expected_class) -> None:
    snap = build_framework()
    row = snap.systems.loc[snap.systems["system_id"] == sid].iloc[0]
    assert float(row["total_score"]) == expected_total
    assert row["readiness_class"] == expected_class


def test_scoring_rules_stay_in_valid_range() -> None:
    # apply_scoring_rules is a keyword-heuristic fallback for missing scores
    # (see prepare_systems_dataframe's auto_score_if_empty path) — it is not
    # expected to reproduce researcher-assigned scores from free-text
    # metadata that wasn't written to trigger specific keywords. This test
    # only checks the rule engine stays well-formed, not that it agrees with
    # the manually researched EXPECTED_SCORES.
    raw = load_systems()
    rescored = apply_scoring_rules(raw.copy(), overwrite=True)
    for _, row in rescored.iterrows():
        for c in CRITERIA:
            assert row[c] in (0, 1, 2)


def test_barriers_and_integration_consistency() -> None:
    snap = build_framework()
    merged = snap.integration
    low = merged[merged["integration_class"] == "Low integration ready"]
    for _, row in low.iterrows():
        assert float(row["total_score"]) < READINESS_MEDIUM_MIN or row["barrier_level"] == "High"


def test_hard_gate_caps_high_scoring_zero_api_system() -> None:
    """A system that would otherwise be 'High integration ready' is capped at
    Medium if it scores 0 on api_availability (core.integration.HARD_GATE_CRITERIA)."""
    df = pd.DataFrame({
        "system_id": ["GATE1"],
        "system_name": ["Synthetic no-API system"],
        "readiness_class": ["High"],
        "total_score": [20.0],
        "criteria_scored": [10],
        "access_method": [""],
        "standards_used": [""],
        "license_type": [""],
        "governance_body": [""],
        "notes": [""],
        **{c: [2] for c in CRITERIA},
    })
    df.loc[0, "api_availability"] = 0
    result = classify_integration_readiness(df, classify_barriers(df))
    row = result.loc[result["system_id"] == "GATE1"].iloc[0]
    assert bool(row["hard_gate_failed"])
    assert row["integration_class"] != "High integration ready"


def test_subscores_sum_to_total_score() -> None:
    from core.validation import governance_subscore, technical_subscore

    snap = build_framework()
    complete = snap.systems.loc[snap.systems["score_complete"]]
    for _, row in complete.iterrows():
        assert technical_subscore(row) + governance_subscore(row) == float(row["total_score"])
