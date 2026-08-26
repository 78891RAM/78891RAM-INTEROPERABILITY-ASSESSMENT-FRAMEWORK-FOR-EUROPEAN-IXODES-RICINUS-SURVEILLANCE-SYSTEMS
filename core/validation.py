# system-level metadata only — never biological records

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import (
    CRITERIA,
    IDENTITY_COLUMNS,
    READINESS_HIGH_MIN,
    READINESS_MEDIUM_MIN,
    SCORE_MAX,
    SCORE_MIN,
)


@dataclass
class ValidationResult:
    """Container for validation outcomes."""

    df: pd.DataFrame
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = [
            f"Rows: {len(self.df)}",
            f"Errors: {len(self.errors)}",
            f"Warnings: {len(self.warnings)}",
        ]
        if self.errors:
            lines.append("Errors:")
            lines.extend(f"  - {e}" for e in self.errors)
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {w}" for w in self.warnings)
        return "\n".join(lines)


def _is_empty_value(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _coerce_score_scalar(value) -> float | None:
    if _is_empty_value(value):
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != int(numeric) or numeric < SCORE_MIN or numeric > SCORE_MAX:
        return None
    return int(numeric)


def coerce_criteria_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Vectorized coercion of criterion columns to 0–2 or NA.

    Returns (dataframe, list of invalid-value error messages).
    """
    out = df.copy()
    invalid_msgs: list[str] = []

    for col in CRITERIA:
        if col not in out.columns:
            out[col] = np.nan
            continue

        raw = out[col]
        numeric = pd.to_numeric(raw, errors="coerce")
        valid_mask = numeric.isin([0, 1, 2])
        out[col] = numeric.where(valid_mask)

        if "system_id" in out.columns:
            bad_mask = numeric.notna() & ~valid_mask
            if bad_mask.any():
                for sid, val in zip(
                    out.loc[bad_mask, "system_id"],
                    raw.loc[bad_mask],
                ):
                    invalid_msgs.append(
                        f"{sid}: '{col}' has invalid value {val!r} (expected 0, 1, or 2)"
                    )
            empty_str_mask = raw.astype(str).str.strip().eq("") & numeric.isna()
            if empty_str_mask.any():
                pass  # treated as missing, not invalid

    return out, invalid_msgs


def validate_systems_df(df: pd.DataFrame, require_complete_scores: bool = False) -> ValidationResult:
    """
    Validate a systems dataframe.

    - Ensures required identity columns exist
    - Checks missing / duplicate system_id
    - Coerces criteria to integers in [0, 2] (vectorized)
    - Flags incomplete rows (missing any criterion score)
    """
    errors: list[str] = []
    warnings: list[str] = []
    working = df.copy()

    for col in IDENTITY_COLUMNS:
        if col not in working.columns:
            errors.append(f"Missing required column: {col}")

    if errors:
        return ValidationResult(working, errors, warnings)

    missing_id = working["system_id"].isna() | (
        working["system_id"].astype(str).str.strip() == ""
    )
    if missing_id.any():
        warnings.append(f"Missing system_id on {int(missing_id.sum())} row(s).")

    if working["system_id"].duplicated().any():
        dupes = working.loc[working["system_id"].duplicated(), "system_id"].astype(str).tolist()
        errors.append(f"Duplicate system_id values: {dupes}")

    for col in CRITERIA:
        if col not in working.columns:
            errors.append(f"Missing criterion column: {col}")
    if errors:
        return ValidationResult(working, errors, warnings)

    working, invalid_msgs = coerce_criteria_columns(working)
    errors.extend(invalid_msgs)

    incomplete_mask = working[CRITERIA].isna().any(axis=1)
    incomplete_ids = working.loc[incomplete_mask, "system_id"].astype(str).tolist()

    if incomplete_ids:
        msg = f"Incomplete score rows: {incomplete_ids}"
        if require_complete_scores:
            errors.append(msg)
        else:
            warnings.append(msg)

    return ValidationResult(working, errors, warnings)


def load_and_validate_systems(path, require_complete_scores: bool = False) -> ValidationResult:
    df = pd.read_csv(path, dtype={"system_id": str})
    return validate_systems_df(df, require_complete_scores=require_complete_scores)


def filter_complete_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if "score_complete" in df.columns:
        return df.loc[df["score_complete"]].copy()
    return df.dropna(subset=CRITERIA).copy()


def filter_scored_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if "criteria_scored" in df.columns:
        return df.loc[df["criteria_scored"] > 0].copy()
    return df.dropna(subset=CRITERIA, how="all").copy()


def total_score(row: pd.Series) -> float:
    vals = row[CRITERIA]
    if vals.notna().sum() == 0:
        return float("nan")
    return float(vals.sum(skipna=True))


# Two-part sub-score grouping (post-audit addition). All 10 criteria still sum
# into total_score unweighted and unchanged — these groups are a reporting
# split, not a re-weighting, so a system can't hide a weak technical profile
# behind strong documentation/governance scores, or vice versa: the two
# numbers are meant to be read side by side with total_score, not instead of it.
TECHNICAL_SUBSCORE_CRITERIA = [
    "api_availability",
    "schema_completeness",
    "semantic_alignment",
    "license_openness",
    "spatial_resolution",
]
GOVERNANCE_SUBSCORE_CRITERIA = [
    "documentation_quality",
    "governance_gdpr_clarity",
    "data_quality",
    "update_frequency",
    "temporal_coverage",
]


def _group_subscore(row: pd.Series, group: list[str]) -> float:
    vals = row[group]
    if vals.notna().sum() == 0:
        return float("nan")
    return float(vals.sum(skipna=True))


def technical_subscore(row: pd.Series) -> float:
    """
    Sum (0-10) of the 5 'hard' interoperability criteria: api_availability,
    schema_completeness, semantic_alignment, license_openness, spatial_resolution.

    These are the criteria that most directly determine whether a system's data
    can actually be machine-consumed and merged — see governance_subscore for
    the complementary 'soft' group.
    """
    return _group_subscore(row, TECHNICAL_SUBSCORE_CRITERIA)


def governance_subscore(row: pd.Series) -> float:
    """
    Sum (0-10) of the 5 'soft' governance/quality criteria: documentation_quality,
    governance_gdpr_clarity, data_quality, update_frequency, temporal_coverage.

    A high governance_subscore does not compensate for a low technical_subscore
    (or vice versa) — report both, since total_score alone can let one hide
    the other. See technical_subscore.
    """
    return _group_subscore(row, GOVERNANCE_SUBSCORE_CRITERIA)


def count_invalid_criteria(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    total = 0
    for col in CRITERIA:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        raw = df[col]
        bad = (raw.notna() & (raw.astype(str).str.strip() != "")) & (numeric.isna() | ~numeric.isin([0, 1, 2]))
        total += int(bad.sum())
    return total


def weighted_readiness_class(total: float) -> str:
    """
    Map total_score (0–20) to readiness class.

    High   >= 15
    Medium 10–14
    Low    < 10
    """
    if pd.isna(total):
        return ""
    if total >= READINESS_HIGH_MIN:
        return "High"
    if total >= READINESS_MEDIUM_MIN:
        return "Medium"
    return "Low"
