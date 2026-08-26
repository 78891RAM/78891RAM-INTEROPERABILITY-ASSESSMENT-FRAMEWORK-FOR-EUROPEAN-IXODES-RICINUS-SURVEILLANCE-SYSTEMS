"""
Turns a system's scorecard + metadata into a list of concrete "what to fix" items.

Two independent sources feed the recommendation list for each system:
1. _RULES — a direct lookup from (criterion, score) to a fix suggestion. Any
   criterion scored 0 or 1 fires its matching rule; a 2 is considered adequate
   and produces nothing.
2. _metadata_recommendations — signals that don't map to a single criterion
   score (e.g. "standards are documented but don't mention FAIR"), read
   straight from the free-text metadata columns.

Recommendations are deliberately generated from what's already in
data/systems.csv, never invented — there is no LLM call or free-text
generation here, only pattern lookups against real scores/metadata, so every
recommendation traces back to a specific field a reader can go check.
"""

from __future__ import annotations

import pandas as pd

from config import CRITERIA, CRITERIA_LABELS

# (criterion, score-as-string, message). Only scores "0" and "1" appear here —
# a 2 needs no fix, so there is intentionally no rule for it. Score is compared
# as a string against int(val) below purely because that was the original
# authoring format; the comparison itself is numeric.
_RULES: list[tuple[str, str, str]] = [
    ("api_availability", "0", "System lacks API — recommend REST API or open data portal."),
    ("api_availability", "1", "API partially available — recommend documented machine-readable endpoints."),
    ("schema_completeness", "0", "Schema not standardised — recommend Darwin Core or GBIF-aligned schema."),
    ("schema_completeness", "1", "Schema partially complete — recommend full metadata schema publication."),
    ("semantic_alignment", "0", "Semantic alignment weak — recommend shared EU tick surveillance vocabulary."),
    ("semantic_alignment", "1", "Partial semantic alignment — recommend ontology mapping to standards."),
    ("license_openness", "0", "Licence restrictions — recommend open licence and FAIR metadata."),
    ("license_openness", "1", "Partial licence openness — clarify redistribution terms."),
    ("documentation_quality", "0", "Documentation insufficient — recommend FAIR metadata and API docs."),
    ("documentation_quality", "1", "Documentation partial — expand integration guides."),
    ("governance_gdpr_clarity", "0", "Governance unclear — recommend GDPR and stewardship documentation."),
    ("governance_gdpr_clarity", "1", "Governance partially documented — clarify data controller roles."),
    ("data_quality", "0", "Data quality controls weak — recommend validation and QA procedures."),
    ("update_frequency", "0", "Infrequent updates — recommend scheduled refresh cadence."),
    ("spatial_resolution", "0", "Coarse spatial resolution — recommend point-level or NUTS-standard georeferencing."),
    ("temporal_coverage", "0", "Limited temporal coverage — recommend continuous monitoring window."),
]


def _metadata_recommendations(row: pd.Series) -> list[str]:
    """Supplement score-based rules with metadata text signals (no fabrication)."""
    recs = []
    standards = str(row.get("standards_used", "") or "").lower()
    access = str(row.get("access_method", "") or "").lower()

    if "fair" not in standards and standards.strip():
        recs.append("Standards documented but FAIR principles not cited — recommend FAIR metadata.")
    elif not standards.strip():
        recs.append("No standards documented — recommend FAIR metadata and Darwin Core.")

    if any(k in access for k in ("not publicly", "restricted", "on request", "no api")):
        recs.append("Access restrictions noted — recommend public API or documented partner access.")

    if "no api" in access or "no public api" in access:
        recs.append("Recommend REST API for programmatic access.")

    return recs


def generate_recommendations(systems_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per recommendation linked to a system.

    Columns: system_id, system_name, priority, recommendation
    """
    if systems_df is None or systems_df.empty:
        return pd.DataFrame(columns=["system_id", "system_name", "priority", "recommendation"])

    rows = []
    for _, row in systems_df.iterrows():
        sid = str(row.get("system_id", ""))
        name = str(row.get("system_name", ""))

        for crit, score_str, message in _RULES:
            if crit not in row.index:
                continue
            val = row[crit]
            if pd.isna(val):
                continue
            if int(val) == int(score_str):
                # A 0 is a hard gap (High priority); a 1 is a partial gap (Medium).
                priority = "High" if int(score_str) == 0 else "Medium"
                rows.append(
                    {
                        "system_id": sid,
                        "system_name": name,
                        "priority": priority,
                        "recommendation": message,
                        "criterion": CRITERIA_LABELS.get(crit, crit),
                    }
                )

        for msg in _metadata_recommendations(row):
            rows.append(
                {
                    "system_id": sid,
                    "system_name": name,
                    "priority": "Medium",
                    "recommendation": msg,
                    "criterion": "Metadata",
                }
            )

        # If neither source produced anything for this system, still give it a row —
        # otherwise a fully-scored system would just be silently absent from the
        # Recommendations tab, which reads as a bug ("where did SYS06 go?") rather
        # than "this system has no outstanding gaps". Scoped to `sid` so it only
        # looks at rows just added for *this* system, not the whole accumulated list.
        if not any(r["system_id"] == sid for r in rows):
            rows.append(
                {
                    "system_id": sid,
                    "system_name": name,
                    "priority": "Low",
                    "recommendation": "No critical gaps identified from current scores and metadata.",
                    "criterion": "—",
                }
            )

    return pd.DataFrame(rows)
