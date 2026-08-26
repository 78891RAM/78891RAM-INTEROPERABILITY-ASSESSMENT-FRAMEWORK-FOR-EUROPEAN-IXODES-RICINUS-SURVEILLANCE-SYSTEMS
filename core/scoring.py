# system-level metadata only — never biological records
"""
Keyword-heuristic auto-scorer for the 10-criteria interoperability scorecard.

This is a FALLBACK, not the primary scoring path. The primary path is a human
(or an LLM doing directed research) reading each system's real documentation
and hand-assigning 0-2 scores directly into data/systems.csv, because the
criteria are genuinely judgment calls — no keyword rule can reliably tell
"open licence" from "licence available on request" without reading the actual
terms. apply_scoring_rules() only runs when a row's criteria are entirely
empty (see core.preparation.prepare_systems_dataframe's auto_score_if_empty
path) and exists to give reviewers a rough first-pass score to react to,
rather than leaving newly-added systems completely unscored while someone
gets round to researching them properly.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from config import CRITERIA
from core.validation import total_score, weighted_readiness_class


def _text(row: pd.Series, *fields: str) -> str:
    """Concatenate the given metadata columns (lower-cased) into one search string."""
    parts = []
    for field in fields:
        val = row.get(field, "")
        if pd.notna(val):
            parts.append(str(val).lower())
    return " ".join(parts)


def score_api_availability(row: pd.Series) -> int:
    """0 = no programmatic access; 1 = downloadable but not a live API; 2 = REST/GBIF-style API."""
    text = _text(row, "access_method", "data_type", "notes")

    # Checked first and wins outright: a system can mention "open data portal" in passing
    # while still being, in substance, a static map with no machine access — the negative
    # phrases here are more specific than the positive ones, so they take priority.
    if any(
        k in text
        for k in (
            "no api",
            "no public api",
            "not publicly downloadable",
            "not openly downloadable",
            "static map",
            "no direct data",
            "n/a (network",
            "not a live data feed",
            "not a continuously operating public system",
            "coordination network",
        )
    ):
        return 0
    if any(k in text for k in ("gbif", "rest api", "open api", "open data portal", "fair-compliant")):
        return 2
    if any(k in text for k in ("downloadable datasets", "structured downloadable", "mixed:", "some open reports")):
        return 1
    return 0


def score_schema_completeness(row: pd.Series) -> int:
    """0 = no defined schema; 1 = partial/internal schema; 2 = a named standard schema (e.g. Darwin Core)."""
    text = _text(row, "standards_used", "data_type", "access_method", "notes")

    if any(k in text for k in ("darwin core", "fair principles", "all key fields")):
        return 2
    if any(k in text for k in ("protocol", "nuts", "internal", "partial", "methodology", "pcr")):
        return 1
    if any(k in text for k in ("not machine-readable", "map-only", "n/a", "coordination network", "no single")):
        return 0
    # Default to 1 (not 0): most citizen-reporting apps have *some* implicit
    # schema (species/date/location) even when no standard is named.
    return 1


def score_semantic_alignment(row: pd.Series) -> int:
    """0 = no shared vocabulary; 1 = partial/internal mapping; 2 = a recognised ontology/standard."""
    text = _text(row, "standards_used", "notes")

    if any(k in text for k in ("darwin core", "fair", "snomed", "dama protocol", "stockholm paradigm")):
        return 2
    if any(k in text for k in ("nuts", "partial", "protocol", "internal", "method")):
        return 1
    return 0


def score_update_frequency(row: pd.Series) -> int:
    """0 = one-off/static; 1 = periodic (weekly-annual); 2 = continuous/real-time."""
    text = _text(row, "update_freq_desc", "notes")

    if any(k in text for k in ("one-off", "single study", "snapshot", "project-based", "not a recurring", "not continuous")):
        return 0
    if any(k in text for k in ("n/a (network", "n/a (network; not dataset)")):
        return 0
    if any(k in text for k in ("continuous", "rolling", "real-time", "daily")):
        return 2
    if any(k in text for k in ("periodic", "varies", "seasonal", "annual", "weekly", "monthly")):
        return 1
    # Default to 1 rather than 0: absence of an update-frequency description is
    # not itself evidence the system is static — most ongoing platforms simply
    # don't publish a formal refresh cadence.
    return 1


def score_spatial_resolution(row: pd.Series) -> int:
    """0 = country-level only / not a dataset; 1 = regional (NUTS2/3); 2 = point-level (GPS)."""
    text = _text(row, "spatial_res_desc", "notes")

    if any(k in text for k in ("point", "gps")):
        return 2
    if any(k in text for k in ("nuts3", "regional", "province", "nuts2")):
        return 1
    if any(k in text for k in ("n/a", "country-level only", "not a dataset")):
        return 0
    return 1


def score_temporal_coverage(row: pd.Series) -> int:
    """0 = single snapshot study; 1 = fixed/limited window; 2 = long-running, still active."""
    text = _text(row, "temporal_cov_desc", "notes")

    if any(k in text for k in ("present", "2005", "2006", "2008", "2010", "2017", "10+")):
        # A long start date alone isn't enough — a decade-old but since-concluded
        # study (e.g. a one-off academic sampling campaign) still scores 0.
        if "one-off" in text or "single study" in text:
            return 0
        return 2
    if any(k in text for k in ("2022-2026", "2015-2021", "varies")):
        return 1
    if any(k in text for k in ("single study", "one-off", "n/a")):
        return 0
    return 1


def score_license_openness(row: pd.Series) -> int:
    """
    0 = closed/restricted/no licence found; 1 = mixed or available-on-request; 2 = an explicit open licence.

    Unstated-evidence convention (formalised post-audit, applies to this criterion
    and any other hedged by "not published" / "not confirmed" / "typically" language):
    a licence that is unstated, unconfirmed, or merely inferred from platform
    convention (e.g. "GBIF-hosted, typically CC0/CC-BY") scores 0, the same as an
    explicitly closed licence — NOT the middle "1" tier. Only a licence that is
    actually documented (named, linked, or stated in the source) earns credit.
    Do not infer openness from what a hosting platform usually does; score only
    what is written down for this specific system. This rule was applied to
    correct an inconsistency between SYS01 and SYS08, which had textually
    parallel "unstated, GBIF-adjacent" evidence but different scores before the
    fix — see their notes fields in data/systems.csv for the correction record.
    """
    text = _text(row, "license_type", "notes")

    if any(k in text for k in ("open (fair", "cc-by", "cc-licensed", "open (maps", "open (published")):
        return 2
    if any(k in text for k in ("mixed", "on request", "summary statistics openly", "restricted (research")):
        return 1
    if any(k in text for k in ("restricted", "closed", "n/a")):
        return 0
    # Unstated-evidence rule (see docstring): an unstated or merely-inferred
    # licence scores 0, not a soft-gap 1 — only a documented licence earns
    # credit. (Prior behaviour defaulted this to 1; changed post-audit to
    # keep the auto-scorer consistent with the hand-scoring convention.)
    return 0


def score_data_quality(row: pd.Series) -> int:
    """0 = no QA process evident; 1 = plausible research rigor; 2 = explicit validation/peer review."""
    text = _text(row, "notes", "standards_used", "governance_body")

    if any(k in text for k in ("expert-validated", "peer-reviewed", "validated", "roc auc", "50,000")):
        return 2
    if any(k in text for k in ("academic", "research", "varies", "methodology")):
        return 1
    if any(k in text for k in ("n/a", "not a data", "coordination network")):
        return 0
    return 1


def score_documentation_quality(row: pd.Series) -> int:
    """0 = essentially undocumented; 1 = some reports/papers; 2 = thorough, citable documentation."""
    text = _text(row, "notes", "access_method")

    if any(k in text for k in ("gbif", "peer-reviewed", "nature", "sci reports", "published", "methodology paper")):
        return 2
    if any(k in text for k in ("reports", "papers", "cost", "study")):
        return 1
    return 1


def score_governance_gdpr_clarity(row: pd.Series) -> int:
    """0 = no accountable body identifiable; 1 = academic/multi-partner consortium; 2 = a named public authority."""
    text = _text(row, "governance_body", "license_type", "notes")

    if any(k in text for k in ("efsa", "ecdc", "ukhsa", "rivm", "inrae", "cost association", "szent istván", "national institute")):
        return 2
    if any(k in text for k in ("academic", "consortium", "rki", "fragmented", "varies", "pragmatick")):
        return 1
    if any(k in text for k in ("no single", "no permanent", "n/a")):
        return 0
    return 1


# Maps each of config.CRITERIA to the function that scores it. apply_scoring_rules
# iterates this dict rather than calling the ten functions by name, so adding a new
# criterion only means adding one function + one entry here.
SCORING_RULES: dict[str, Callable[[pd.Series], int]] = {
    "api_availability": score_api_availability,
    "schema_completeness": score_schema_completeness,
    "semantic_alignment": score_semantic_alignment,
    "update_frequency": score_update_frequency,
    "spatial_resolution": score_spatial_resolution,
    "temporal_coverage": score_temporal_coverage,
    "license_openness": score_license_openness,
    "data_quality": score_data_quality,
    "documentation_quality": score_documentation_quality,
    "governance_gdpr_clarity": score_governance_gdpr_clarity,
}


def apply_scoring_rules(df: pd.DataFrame, overwrite: bool = True) -> pd.DataFrame:
    """
    Apply rule-based scoring to each system row.

    When overwrite=False, only fills missing criterion values.
    """
    result = df.copy()

    for idx, row in result.iterrows():
        # Rows whose name is blank or still holds a spreadsheet template placeholder
        # (e.g. "[ADD SYSTEM NAME]") are unfilled template rows carried over from the
        # source workbook, not real systems — leave every score as NA rather than
        # scoring a system that doesn't actually exist yet.
        if pd.isna(row.get("system_name")) or str(row.get("system_name", "")).startswith("[ADD"):
            for crit in CRITERIA:
                result.at[idx, crit] = pd.NA
            result.at[idx, "total_score"] = pd.NA
            result.at[idx, "readiness_class"] = ""
            continue

        for crit, rule_fn in SCORING_RULES.items():
            computed = rule_fn(row)
            if overwrite:
                result.at[idx, crit] = computed
            elif pd.isna(result.at[idx, crit]):
                result.at[idx, crit] = computed

    result["total_score"] = result.apply(
        lambda r: total_score(r) if r[CRITERIA].notna().all() else float("nan"), axis=1
    )
    result["readiness_class"] = result["total_score"].apply(
        lambda t: weighted_readiness_class(t) if pd.notna(t) else ""
    )

    return result
