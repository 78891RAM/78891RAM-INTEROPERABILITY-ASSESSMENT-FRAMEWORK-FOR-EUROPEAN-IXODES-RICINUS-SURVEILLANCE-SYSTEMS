"""
Classifies each system's interoperability barriers across five dimensions.

Each barrier's severity comes from two independent signals that are then
combined (see _combine_severity): the *lowest* of the relevant criterion
scores for that dimension (a single weak score is enough to flag a barrier —
averaging would let one bad score hide behind good ones), and a keyword scan
of the system's free-text metadata for phrases that describe a barrier even
when the numeric scores don't fully capture it (e.g. "restricted" access
noted in prose but not reflected in any single 0-2 score).
"""

from __future__ import annotations

import pandas as pd

BARRIER_TYPES = ["technical_barrier", "semantic_barrier", "legal_barrier", "governance_barrier", "accessibility_barrier"]

# "organisational_barrier" predates the current five-category split and is kept only
# as a duplicate of governance_barrier so any older code/URLs that still key on it
# keep working — do not add new logic against it, use governance_barrier instead.
LEGACY_BARRIER_COLUMN = "organisational_barrier"

# Which of the 10 scorecard criteria feed each barrier dimension. A dimension's
# score-derived severity is the worst (lowest) of its listed criteria, so a
# criterion can appear under more than one dimension — e.g. api_availability
# affects both "can you access the data at all" (accessibility) and "can a
# machine consume it" (technical).
TECHNICAL_CRITERIA = ["api_availability", "schema_completeness", "spatial_resolution", "update_frequency"]
SEMANTIC_CRITERIA = ["semantic_alignment"]
LEGAL_CRITERIA = ["license_openness"]
GOVERNANCE_CRITERIA = ["governance_gdpr_clarity", "documentation_quality"]
ACCESSIBILITY_CRITERIA = ["license_openness", "api_availability"]
ORG_CRITERIA = GOVERNANCE_CRITERIA  # kept for the same backward-compatibility reason as LEGACY_BARRIER_COLUMN

# Keyword lists used to detect each barrier from free text, independent of the
# numeric scores. Kept short and specific deliberately — broad words like "data"
# would false-positive on nearly every row.
TECHNICAL_TEXT = ("no api", "not machine-readable", "not publicly", "static map", "one-off", "not a live")
SEMANTIC_TEXT = ("no single", "no formal", "no shared", "internal", "fragmented", "inconsistent")
LEGAL_TEXT = ("restricted", "closed", "on request", "gdpr", "privacy")
ACCESS_TEXT = ("restricted", "not publicly", "on request", "not openly", "not downloadable", "no api")
GOVERNANCE_TEXT = ("fragmented", "no single", "no permanent", "varies", "coordination network", "one-off", "regional")


def _severity_from_scores(scores: pd.Series) -> str:
    """Worst-case severity from a row's criterion scores: min <=0 -> High, <=1 -> Medium, else Low."""
    valid = scores.dropna()
    if valid.empty:
        return "Unknown"
    minimum = float(valid.min())
    if minimum <= 0:
        return "High"
    if minimum <= 1:
        return "Medium"
    return "Low"


def _text_hits(text: str, keywords: tuple[str, ...]) -> bool:
    """True if any keyword appears in `text` (case-insensitive)."""
    if not text or (isinstance(text, float) and pd.isna(text)):
        return False
    lower = str(text).lower()
    return any(k in lower for k in keywords)


def _combine_severity(score_severity: str, text_flag: bool) -> str:
    """
    Merge the numeric-score severity with the text-keyword flag.

    A text hit can only push severity up, never down — free text describing a
    barrier is treated as corroborating evidence, not grounds to override a
    score that already says the barrier exists. An Unknown score (no relevant
    criteria present) with a text hit becomes Medium rather than High, since
    prose alone is weaker evidence than an actual low score.
    """
    if score_severity == "High" or text_flag:
        return "High"
    if score_severity == "Medium":
        return "Medium"
    if score_severity == "Unknown" and text_flag:
        return "Medium"
    return score_severity if score_severity != "Unknown" else "Low"


def classify_barriers(df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-system barrier table: one severity per dimension, plus a rollup."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["system_id", "system_name"] + BARRIER_TYPES + [LEGACY_BARRIER_COLUMN, "barrier_level", "barrier_count"])

    out = df[["system_id", "system_name"]].copy() if "system_name" in df.columns else df[["system_id"]].copy()

    # Combined bag of text fields for the governance-barrier keyword scan, which
    # (unlike the other four dimensions) isn't tied to one specific metadata column.
    meta_text = (
        df.get("access_method", "").fillna("").astype(str)
        + " "
        + df.get("standards_used", "").fillna("").astype(str)
        + " "
        + df.get("license_type", "").fillna("").astype(str)
        + " "
        + df.get("governance_body", "").fillna("").astype(str)
        + " "
        + df.get("notes", "").fillna("").astype(str)
    )

    tech_scores = df[[c for c in TECHNICAL_CRITERIA if c in df.columns]]
    sem_scores = df[[c for c in SEMANTIC_CRITERIA if c in df.columns]]
    leg_scores = df[[c for c in LEGAL_CRITERIA if c in df.columns]]
    gov_scores = df[[c for c in GOVERNANCE_CRITERIA if c in df.columns]]
    acc_scores = df[[c for c in ACCESSIBILITY_CRITERIA if c in df.columns]]

    tech_from_scores = tech_scores.apply(_severity_from_scores, axis=1)
    sem_from_scores = sem_scores.apply(_severity_from_scores, axis=1)
    leg_from_scores = leg_scores.apply(_severity_from_scores, axis=1)
    gov_from_scores = gov_scores.apply(_severity_from_scores, axis=1)
    acc_from_scores = acc_scores.apply(_severity_from_scores, axis=1)

    access_text = df.get("access_method", "").fillna("").astype(str) + " " + df.get("license_type", "").fillna("").astype(str)

    out["technical_barrier"] = [
        _combine_severity(ts, _text_hits(t, TECHNICAL_TEXT))
        for ts, t in zip(tech_from_scores, df.get("access_method", ""))
    ]

    out["semantic_barrier"] = [
        _combine_severity(ss, _text_hits(t, SEMANTIC_TEXT))
        for ss, t in zip(sem_from_scores, df.get("standards_used", ""))
    ]

    out["legal_barrier"] = [
        _combine_severity(ls, _text_hits(t, LEGAL_TEXT))
        for ls, t in zip(leg_from_scores, df.get("license_type", ""))
    ]

    out["governance_barrier"] = [
        _combine_severity(gs, _text_hits(t, GOVERNANCE_TEXT))
        for gs, t in zip(gov_from_scores, meta_text)
    ]

    out["accessibility_barrier"] = [
        _combine_severity(as_, _text_hits(t, ACCESS_TEXT))
        for as_, t in zip(acc_from_scores, access_text)
    ]

    out[LEGACY_BARRIER_COLUMN] = out["governance_barrier"]

    severity_cols = BARRIER_TYPES
    out["barrier_count"] = (out[severity_cols] == "High").sum(axis=1)
    # Overall readiness-blocking level: 2+ High barriers is itself High, exactly
    # one is Medium, none is Low — deliberately blunt (count, not a weighted
    # score) so it's easy to explain in the dissertation write-up.
    out["barrier_level"] = out["barrier_count"].map(lambda n: "High" if n >= 2 else "Medium" if n == 1 else "Low")

    out["barrier_summary"] = out.apply(_barrier_summary_row, axis=1)

    return out


def _barrier_summary_row(row: pd.Series) -> str:
    """One-line "Technical: High; Legal: Medium" style summary for table display."""
    parts = []
    for col in BARRIER_TYPES:
        label = col.replace("_barrier", "").title()
        if row.get(col) in ("High", "Medium"):
            parts.append(f"{label}: {row[col]}")
    if parts:
        return "; ".join(parts)
    return "No significant barriers identified"


def barriers_summary_chart_df(barriers_df: pd.DataFrame) -> pd.DataFrame:
    """Long-format (system x barrier_type x severity) table for the grouped bar chart."""
    if barriers_df.empty:
        return pd.DataFrame()

    id_cols = ["system_id", "system_name"] if "system_name" in barriers_df.columns else ["system_id"]

    melted = barriers_df.melt(
        id_vars=id_cols,
        value_vars=BARRIER_TYPES,
        var_name="barrier_type",
        value_name="severity",
    )

    melted["barrier_type"] = melted["barrier_type"].str.replace("_barrier", "").str.title()

    severity_order = {"High": 3, "Medium": 2, "Low": 1, "Unknown": 0}
    melted["severity_rank"] = melted["severity"].map(severity_order).fillna(0)

    return melted
