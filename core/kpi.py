"""KPI calculations for dashboard overview."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class OverviewKPIs:
    """
    Summary statistics for the overview tab.

    high_integration_ready/medium_.../low_... (barrier- and hard-gate-adjusted,
    from core.integration.classify_integration_readiness) are the headline
    counts — see ui/tabs/overview.py. high_readiness/medium_.../low_...
    (score-only, from readiness_class) are kept as a supporting figure so the
    unadjusted score view stays visible, not because it's the primary verdict.
    """

    total_systems: int = 0
    average_score: float = 0.0
    highest_score: float = 0.0
    lowest_score: float = 0.0
    high_readiness: int = 0
    medium_readiness: int = 0
    low_readiness: int = 0
    high_integration_ready: int = 0
    medium_integration_ready: int = 0
    low_integration_ready: int = 0
    scored_systems: int = 0


def compute_overview_kpis(systems_df: pd.DataFrame, integration_df: pd.DataFrame | None = None) -> OverviewKPIs:
    kpis = OverviewKPIs()
    if systems_df is None or systems_df.empty:
        return kpis

    kpis = OverviewKPIs(total_systems=len(systems_df))
    scores = pd.to_numeric(systems_df.get("total_score"), errors="coerce").dropna()
    scored = int(scores.shape[0])

    if not scores.empty:
        integration_counts = {"High integration ready": 0, "Medium integration ready": 0, "Low integration ready": 0}
        if integration_df is not None and not integration_df.empty and "integration_class" in integration_df.columns:
            vc = integration_df["integration_class"].value_counts()
            integration_counts.update({k: int(vc.get(k, 0)) for k in integration_counts})

        return OverviewKPIs(
            total_systems=len(systems_df),
            average_score=round(float(scores.mean()), 1),
            highest_score=round(float(scores.max()), 1),
            lowest_score=round(float(scores.min()), 1),
            high_readiness=int((systems_df["readiness_class"] == "High").sum()) if "readiness_class" in systems_df.columns else 0,
            medium_readiness=int((systems_df["readiness_class"] == "Medium").sum()) if "readiness_class" in systems_df.columns else 0,
            low_readiness=int((systems_df["readiness_class"] == "Low").sum()) if "readiness_class" in systems_df.columns else 0,
            high_integration_ready=integration_counts["High integration ready"],
            medium_integration_ready=integration_counts["Medium integration ready"],
            low_integration_ready=integration_counts["Low integration ready"],
            scored_systems=scored,
        )

    return kpis


def readiness_breakdown_df(systems_df: pd.DataFrame) -> pd.DataFrame:
    if systems_df is None or systems_df.empty or "readiness_class" not in systems_df.columns:
        return pd.DataFrame(columns=["readiness_class", "count"])
    counts = systems_df["readiness_class"].fillna("Unknown").value_counts().reset_index()
    counts.columns = ["readiness_class", "count"]
    return counts
