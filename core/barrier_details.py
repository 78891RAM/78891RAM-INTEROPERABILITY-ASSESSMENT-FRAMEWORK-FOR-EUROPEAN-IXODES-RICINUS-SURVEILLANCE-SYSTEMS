"""
Expands the five barrier-severity columns from core.barriers into one row per
(system, barrier) pair with a human-readable reason and a fix recommendation —
this is the table that actually renders on the Barriers tab. classify_barriers
only tells you the severity; this module explains *why* and *what to do*.
"""

from __future__ import annotations

import pandas as pd

from core.barriers import BARRIER_TYPES, classify_barriers
from config import CRITERIA_LABELS

BARRIER_LABELS = {
    "technical_barrier": "Technical",
    "semantic_barrier": "Semantic",
    "legal_barrier": "Legal",
    "governance_barrier": "Governance",
    "accessibility_barrier": "Accessibility",
    "organisational_barrier": "Governance",  # legacy alias, see core.barriers.LEGACY_BARRIER_COLUMN
}

# Which criterion feeds which barrier, for building the "reason" text below —
# mirrors core.barriers' *_CRITERIA groupings but keyed the other way round
# (criterion -> barrier) since here we're explaining a barrier from its criteria,
# not deriving a barrier's severity from them.
CRITERION_TO_BARRIER: dict[str, str] = {
    "api_availability": "technical_barrier",
    "schema_completeness": "technical_barrier",
    "spatial_resolution": "technical_barrier",
    "update_frequency": "technical_barrier",
    "semantic_alignment": "semantic_barrier",
    "license_openness": "legal_barrier",
    "governance_gdpr_clarity": "governance_barrier",
    "documentation_quality": "governance_barrier",
}

# Fix recommendation text per barrier category and severity level.
RECOMMENDATIONS: dict[str, dict[str, str]] = {
    "technical_barrier": {
        "High": "Expose a documented REST API or machine-readable data feed.",
        "Medium": "Improve API documentation and schema completeness.",
        "Low": "Maintain current technical access pathways.",
    },
    "semantic_barrier": {
        "High": "Adopt Darwin Core or a shared EU tick surveillance vocabulary.",
        "Medium": "Map local fields to standard ontologies (e.g. GBIF, NUTS).",
        "Low": "Continue semantic alignment monitoring.",
    },
    "legal_barrier": {
        "High": "Publish an open licence and clarify data-sharing terms.",
        "Medium": "Document licence scope and GDPR constraints.",
        "Low": "Licence posture is adequate for integration.",
    },
    "governance_barrier": {
        "High": "Establish a single governance body and GDPR documentation.",
        "Medium": "Clarify institutional roles and data stewardship.",
        "Low": "Governance structure supports integration.",
    },
    "accessibility_barrier": {
        "High": "Remove access restrictions; enable public or partner API access.",
        "Medium": "Publish FAIR metadata and access procedures.",
        "Low": "Access model supports responsible sharing.",
    },
}


def _reason_for_barrier(row: pd.Series, barrier_col: str, systems_row: pd.Series) -> str:
    """
    Build a reason string from criterion scores and metadata.

    Prefers naming the specific low-scoring criteria that drive this barrier
    (e.g. "API Availability=0"); falls back to a generic "Derived severity: X"
    only when no individual criterion explains it (severity came purely from
    the keyword-text signal in classify_barriers, not from a 0/1 score).
    """
    parts = []
    for crit, bcol in CRITERION_TO_BARRIER.items():
        if bcol != barrier_col:
            continue
        if crit not in systems_row.index:
            continue
        val = systems_row.get(crit)
        if not pd.notna(val):
            continue
        if int(val) <= 1:
            label = CRITERIA_LABELS.get(crit, crit)
            parts.append(f"{label}={int(val)}")

    if barrier_col == "accessibility_barrier":
        access = str(systems_row.get("access_method", "") or "")
        licence = str(systems_row.get("license_type", "") or "")
        if any(k in access.lower() for k in ("restricted", "not publicly", "on request")):
            parts.append("access restrictions in metadata")
        if any(k in licence.lower() for k in ("restricted", "closed", "on request")):
            parts.append("licence restrictions in metadata")

    severity = row.get(barrier_col, "Unknown")
    if not parts and severity in ("High", "Medium"):
        parts.append(f"Derived severity: {severity}")

    if parts:
        return "; ".join(parts)
    return "No significant barrier drivers identified"


def build_barrier_detail_table(systems_df: pd.DataFrame, barriers_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Expand barriers into one row per system × barrier category.

    Only High/Medium severities get a row — a Low-severity barrier isn't a
    barrier worth reporting, so this table is naturally shorter than
    len(systems) x len(BARRIER_TYPES).

    Columns: system_id, system_name, barrier, severity, reason, recommendation
    """
    if systems_df is None or systems_df.empty:
        return pd.DataFrame(columns=["system_id", "system_name", "barrier", "severity", "reason", "recommendation"])

    if barriers_df is None or barriers_df.empty:
        barriers_df = classify_barriers(systems_df)

    systems_idx = systems_df.set_index("system_id")
    rows = []

    detail_cols = [c for c in BARRIER_TYPES if c in barriers_df.columns]
    if "organisational_barrier" in barriers_df.columns and "governance_barrier" not in detail_cols:
        detail_cols.append("organisational_barrier")

    for _, brow in barriers_df.iterrows():
        sid = str(brow["system_id"])
        srow = systems_idx.loc[sid] if sid in systems_idx.index else pd.Series(dtype=object)
        if isinstance(srow, pd.DataFrame):
            # set_index("system_id") only produces duplicate rows if the same
            # system_id appears twice in systems_df, which validate_systems_df
            # is supposed to have already de-duplicated — this is a defensive
            # fallback (take the first match) in case that guarantee ever slips.
            srow = srow.iloc[0]

        for bcol in detail_cols:
            severity = str(brow.get(bcol, "Unknown"))
            if severity not in ("High", "Medium"):
                continue
            label = BARRIER_LABELS.get(bcol, bcol)
            rec_map = RECOMMENDATIONS.get("governance_barrier" if bcol == "organisational_barrier" else bcol, {})
            rows.append({
                "system_id": sid,
                "system_name": str(brow.get("system_name", "")),
                "barrier": label,
                "severity": severity,
                "reason": _reason_for_barrier(brow, bcol, srow),
                "recommendation": rec_map.get(severity, "Review interoperability posture."),
            })

    return pd.DataFrame(rows)


def barrier_frequency_df(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Count barrier occurrences by category for pie/bar charts."""
    if detail_df is None or detail_df.empty:
        return pd.DataFrame(columns=["barrier", "count"])
    return (
        detail_df.groupby("barrier", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
    )


def barrier_severity_distribution(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Severity counts across all barrier records."""
    if detail_df is None or detail_df.empty:
        return pd.DataFrame(columns=["severity", "count"])
    return (
        detail_df.groupby("severity", as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
