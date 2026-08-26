"""
Turns each system's total score + barrier severity into a single three-way
"integration readiness" verdict, and produces the enriched hover-text table
the Map tab renders. This is the closest thing the framework has to a final
"should we integrate this system next" recommendation — see
classify_integration_readiness for the exact thresholds.
"""

from __future__ import annotations

import pandas as pd

from core.barriers import classify_barriers
from config import READINESS_HIGH_MIN, READINESS_MEDIUM_MIN

INTEGRATION_CLASSES = ["High integration ready", "Medium integration ready", "Low integration ready"]

INTEGRATION_COLORS = {
    "High integration ready": "#27ae60",
    "Medium integration ready": "#f39c12",
    "Low integration ready": "#c0392b",
}

# Hard-gate rule (post-audit addition): criteria whose 0 score is treated as a
# genuinely blocking gap, named here explicitly rather than left as a side
# effect of the barrier-severity heuristic in core.barriers. A system that
# fails this gate cannot be "High integration ready" regardless of total_score
# or barrier_level — starting with api_availability, since zero programmatic
# access to a system's data makes "High" indefensible no matter how strong
# everything else scores. Extend this list only with criteria that are
# similarly non-negotiable for integration, not merely low-scoring.
HARD_GATE_CRITERIA = ["api_availability"]


def _fails_hard_gate(row: pd.Series) -> bool:
    """True if this system scores 0 on any HARD_GATE_CRITERIA criterion."""
    return any(
        pd.to_numeric(row.get(c), errors="coerce") == 0
        for c in HARD_GATE_CRITERIA
        if c in row.index
    )


def classify_integration_readiness(
    systems_df: pd.DataFrame,
    barriers_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Produce integration decision table per system. This is the primary,
    headline classification for the app (see ui/tabs/overview.py) — total_score
    and readiness_class are supporting figures, not the final verdict.

    Rules:
      - Low integration ready:  total_score < 10 OR barrier_level == High
      - High integration ready: total_score >= 15 AND barrier_level == Low
                                 AND the system passes the hard gate below
      - Medium integration ready: all other cases

    A high score alone isn't enough for "High" — a barrier can veto it (e.g. a
    well-documented system that's still legally restricted stays capped at
    Medium), and conversely a single High barrier is enough to force "Low"
    regardless of score, since one severe barrier (e.g. no licence at all)
    blocks integration outright no matter how good everything else is.

    Hard gate: on top of the barrier veto above, HARD_GATE_CRITERIA is checked
    explicitly — a system scoring 0 on any of those criteria is capped below
    "High integration ready" even if its score and barrier_level would
    otherwise qualify it. This is a named, stated rule (not an emergent
    property of the barrier heuristic), because relying on barriers alone to
    catch every blocking gap isn't guaranteed to generalise as new systems are
    added.
    """
    if systems_df is None or systems_df.empty:
        return pd.DataFrame()

    if barriers_df is None or barriers_df.empty:
        barriers_df = classify_barriers(systems_df)

    merge_cols = ["system_id"]
    barrier_cols = [
        "technical_barrier",
        "semantic_barrier",
        "legal_barrier",
        "governance_barrier",
        "accessibility_barrier",
        "organisational_barrier",
        "barrier_level",
        "barrier_count",
        "barrier_summary",
    ]
    available_barrier = [c for c in barrier_cols if c in barriers_df.columns]

    gate_cols = [c for c in HARD_GATE_CRITERIA if c in systems_df.columns]
    out = systems_df[["system_id", "system_name"] + gate_cols].copy()
    score_cols = [c for c in ("total_score", "readiness_class", "criteria_scored") if c in systems_df.columns]
    for col in score_cols:
        out[col] = systems_df[col].values

    out = out.merge(barriers_df[merge_cols + available_barrier], on="system_id", how="left")

    out["total_score"] = pd.to_numeric(out.get("total_score"), errors="coerce").fillna(0)
    # A system with no matching barriers row (shouldn't normally happen, since
    # barriers_df is derived from the same systems_df) is treated as Medium
    # risk rather than Low or High, since we genuinely don't know its barriers.
    out["barrier_level"] = out.get("barrier_level", "Medium").fillna("Medium")
    out["integration_class"] = out.apply(_integration_class_row, axis=1)

    out["hard_gate_failed"] = out.apply(_fails_hard_gate, axis=1)
    downgrade = out["hard_gate_failed"] & (out["integration_class"] == "High integration ready")
    out.loc[downgrade, "integration_class"] = "Medium integration ready"

    # Sort key: integration class dominates (3/2/1), with total_score as a tiebreaker
    # within the same class (divided by 100 so it can never flip two systems into a
    # different class-ordering — it only orders systems that already tied on class).
    out["integration_rank"] = (
        out["integration_class"].map(
            {"High integration ready": 3, "Medium integration ready": 2, "Low integration ready": 1}
        ).fillna(0)
        + out["total_score"].fillna(0) / 100
    )

    out = out.sort_values("integration_rank", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def _integration_class_row(row: pd.Series) -> str:
    """Per-row implementation of the score+barrier rule documented on classify_integration_readiness (before the hard gate, applied separately)."""
    score = float(row.get("total_score", 0) or 0)
    barrier = str(row.get("barrier_level", "Medium"))

    if score < READINESS_MEDIUM_MIN or barrier == "High":
        return "Low integration ready"
    if score >= READINESS_HIGH_MIN and barrier == "Low":
        return "High integration ready"
    return "Medium integration ready"


def enrich_map_dataframe(scatter_df: pd.DataFrame, integration_df: pd.DataFrame) -> pd.DataFrame:
    """Merge integration/barrier fields into map scatter data for rich tooltips."""
    if scatter_df is None or scatter_df.empty:
        return pd.DataFrame()

    if integration_df is None or integration_df.empty:
        out = scatter_df.copy()
        out["barrier_level"] = "Unknown"
        out["integration_class"] = "Unknown"
        return out

    extra = [
        c
        for c in (
            "readiness_class",
            "barrier_level",
            "barrier_summary",
            "integration_class",
            "technical_barrier",
            "semantic_barrier",
            "legal_barrier",
            "governance_barrier",
            "accessibility_barrier",
            "organisational_barrier",
        )
        if c in integration_df.columns
    ]
    # scatter_df already carries its own readiness_class from the geo-build step
    # (core.geo.build_map_dataframe); drop integration_df's copy to avoid a
    # _x/_y column-name collision on merge.
    if "readiness_class" in scatter_df.columns:
        extra = [c for c in extra if c != "readiness_class"]

    merged = scatter_df.merge(integration_df[["system_id"] + extra], on="system_id", how="left")

    merged["barrier_level"] = merged["barrier_level"].fillna("Unknown")
    merged["integration_class"] = merged["integration_class"].fillna("Unknown")
    if "readiness_class" in merged.columns:
        merged["readiness_class"] = merged["readiness_class"].fillna("Unknown")
    else:
        merged["readiness_class"] = "Unknown"

    merged["hover_text"] = merged.apply(_map_hover_text, axis=1)
    return merged


def _map_hover_text(row: pd.Series) -> str:
    """HTML hover tooltip shown on the map for a single system's marker."""
    return (
        f"<b>{row.get('system_name', '')}</b><br>"
        f"Score: {row.get('total_score', '—')}/20<br>"
        f"Readiness: {row.get('readiness_class', '—')}<br>"
        f"Barriers: {row.get('barrier_level', '—')}<br>"
        f"Integration: {row.get('integration_class', '—')}<br>"
        f"Region: {row.get('countries_covered', '—')}"
    )
