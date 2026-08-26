from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.scoring import apply_scoring_rules
from config import COUNTRY_COLUMN, CRITERIA
from core.geo import MapBuildResult, resolve_iso3_list
from core.validation import (
    GOVERNANCE_SUBSCORE_CRITERIA,
    TECHNICAL_SUBSCORE_CRITERIA,
    ValidationResult,
    count_invalid_criteria,
    validate_systems_df,
    weighted_readiness_class,
)


@dataclass
class DataQualityReport:
    """Aggregated data-quality findings for sidebar debug panel."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)
    row_count: int = 0
    complete_score_rows: int = 0
    partial_score_rows: int = 0
    missing_system_id_rows: int = 0
    missing_country_rows: int = 0
    invalid_score_cells: int = 0
    duplicate_system_ids: list[str] = field(default_factory=list)
    invalid_criterion_entries: list[str] = field(default_factory=list)
    map_result: MapBuildResult | None = None

    @property
    def has_issues(self) -> bool:
        return bool(self.errors or self.warnings)

    @property
    def unmapped_map_systems(self) -> list[str]:
        if self.map_result:
            return self.map_result.unmapped_system_ids
        return []

    def summary_lines(self) -> list[str]:
        lines = [
            f"Rows loaded: {self.row_count}",
            f"Complete scores: {self.complete_score_rows}",
            f"Partial scores: {self.partial_score_rows}",
            f"Missing system_id: {self.missing_system_id_rows}",
            f"Missing country: {self.missing_country_rows}",
            f"Invalid score cells: {self.invalid_score_cells}",
        ]
        if self.map_result:
            lines.extend([
                f"Map — loaded: {self.map_result.systems_loaded}",
                f"Map — mapped: {self.map_result.systems_mapped}",
                f"Map — skipped: {self.map_result.systems_skipped}",
            ])
        if self.duplicate_system_ids:
            lines.append(f"Duplicate IDs: {', '.join(self.duplicate_system_ids)}")
        if self.unmapped_map_systems:
            lines.append(f"Unmapped for map: {', '.join(self.unmapped_map_systems)}")
        lines.extend(f"ERROR: {e}" for e in self.errors)
        lines.extend(f"WARN: {w}" for w in self.warnings)
        lines.extend(f"INFO: {i}" for i in self.info)
        return lines


def coerce_criteria_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    crit_frame = pd.DataFrame(index=out.index)
    for col in CRITERIA:
        if col not in out.columns:
            crit_frame[col] = np.nan
            continue
        numeric = pd.to_numeric(out[col], errors="coerce")
        crit_frame[col] = numeric.where(numeric.isin([0, 1, 2]))
    out[CRITERIA] = crit_frame
    return out


def compute_derived_scores_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    crit = out[CRITERIA]
    out["criteria_scored"] = crit.notna().sum(axis=1)
    out["total_score"] = crit.sum(axis=1, min_count=1)
    # Reporting sub-scores (see core.validation.technical_subscore /
    # governance_subscore) — do not feed into total_score, which stays the
    # unweighted sum of all 10 criteria exactly as before.
    out["technical_subscore"] = out[TECHNICAL_SUBSCORE_CRITERIA].sum(axis=1, min_count=1)
    out["governance_subscore"] = out[GOVERNANCE_SUBSCORE_CRITERIA].sum(axis=1, min_count=1)
    out["max_possible_score"] = out["criteria_scored"] * 2
    out["normalized_score"] = np.where(
        out["criteria_scored"] > 0,
        out["total_score"] / out["max_possible_score"] * 20,
        np.nan,
    )
    out["score_complete"] = out["criteria_scored"] == len(CRITERIA)
    out["readiness_class"] = out["total_score"].apply(weighted_readiness_class)
    return out


def _count_missing_country(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    col = COUNTRY_COLUMN if COUNTRY_COLUMN in df.columns else "country"
    if col not in df.columns:
        return len(df)
    empty = df[col].isna() | (df[col].astype(str).str.strip() == "")
    no_fallback = empty.copy()
    if no_fallback.any() and "system_name" in df.columns:
        resolved = df.loc[no_fallback, "system_name"].map(resolve_iso3_list)
        no_fallback.loc[resolved[resolved.map(len) > 0].index] = False
    return int(no_fallback.sum())


def prepare_systems_dataframe(
    df: pd.DataFrame, auto_score_if_empty: bool = True
) -> tuple[pd.DataFrame, DataQualityReport]:
    report = DataQualityReport()
    if df is None or df.empty:
        report.warnings.append("Dataset is empty.")
        return pd.DataFrame(), report

    report.row_count = len(df)
    report.invalid_score_cells = count_invalid_criteria(df)

    if "system_id" not in df.columns:
        report.errors.append("Missing required column: system_id")
        return pd.DataFrame(), report

    missing_id_mask = df["system_id"].isna() | (df["system_id"].astype(str).str.strip() == "")
    report.missing_system_id_rows = int(missing_id_mask.sum())
    if report.missing_system_id_rows:
        report.warnings.append(
            f"{report.missing_system_id_rows} row(s) with missing system_id will be excluded."
        )

    work = df.loc[~missing_id_mask].copy()
    if work.empty:
        report.errors.append("No rows with valid system_id.")
        return pd.DataFrame(), report

    report.missing_country_rows = _count_missing_country(work)

    dupes = work.loc[work["system_id"].duplicated(keep=False), "system_id"].unique().tolist()
    report.duplicate_system_ids = [str(d) for d in dupes]
    if dupes:
        report.warnings.append(f"Duplicate system_id values: {report.duplicate_system_ids}")
        work = work.drop_duplicates(subset=["system_id"], keep="first")
        report.info.append("Kept first occurrence of each duplicate system_id for display.")

    validation = validate_systems_df(work, require_complete_scores=False)
    report.errors.extend([e for e in validation.errors if "Duplicate" not in e])
    report.warnings.extend(validation.warnings)
    report.invalid_criterion_entries = [e for e in validation.errors if "invalid value" in e]

    out = coerce_criteria_vectorized(validation.df)

    if auto_score_if_empty and out[CRITERIA].isna().all().all():
        out = apply_scoring_rules(out)
        out = coerce_criteria_vectorized(out)
        report.info.append("Auto-scoring applied (all criteria were empty).")

    out = compute_derived_scores_vectorized(out)

    if "readiness_class" in out.columns:
        stored = out["readiness_class"].fillna("").astype(str)
        computed = out["total_score"].apply(weighted_readiness_class)
        mismatch = (stored != "") & (stored != computed) & computed.notna()
        if mismatch.any():
            ids = out.loc[mismatch, "system_id"].astype(str).tolist()
            report.warnings.append(f"Readiness class mismatch (stored vs computed) for: {ids}")

    report.complete_score_rows = int(out["score_complete"].sum())
    report.partial_score_rows = int((~out["score_complete"] & (out["criteria_scored"] > 0)).sum())

    return out, report


def attach_map_quality(report: DataQualityReport, map_result: MapBuildResult) -> DataQualityReport:
    report.map_result = map_result
    if map_result.systems_skipped > 0:
        report.warnings.append(f"{map_result.systems_skipped} system(s) could not be mapped geographically.")
    return report
