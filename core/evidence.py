"""Evidence merge logic — pure functions on DataFrames."""
from __future__ import annotations

import re

import pandas as pd

from config import EVIDENCE_COLUMNS, EVIDENCE_UNAVAILABLE

_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s,;)]+", re.IGNORECASE)


def _clean(val: object) -> str:
    """Normalise a cell to a display string."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return EVIDENCE_UNAVAILABLE
    s = str(val).strip()
    if s:
        return s
    return EVIDENCE_UNAVAILABLE


def merge_evidence(systems_df: pd.DataFrame, evidence_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge curated evidence CSV with system inventory.

    Curated evidence takes precedence; publisher falls back to operator_org.
    """
    if systems_df is None or systems_df.empty:
        return pd.DataFrame(columns=EVIDENCE_COLUMNS)

    base_ids = systems_df["system_id"].astype(str).tolist()
    rows = {sid: {"system_id": sid} for sid in base_ids}

    if evidence_df is not None and not evidence_df.empty:
        for _, erow in evidence_df.iterrows():
            sid = str(erow.get("system_id", "")).strip()
            if not sid:
                continue
            rows.setdefault(sid, {"system_id": sid})
            for col in EVIDENCE_COLUMNS:
                if col == "system_id":
                    continue
                if col in erow.index and pd.notna(erow[col]) and str(erow[col]).strip():
                    rows[sid][col] = str(erow[col]).strip()

    for _, srow in systems_df.iterrows():
        sid = str(srow.get("system_id", ""))
        rows.setdefault(sid, {"system_id": sid})
        if rows[sid].get("publisher", EVIDENCE_UNAVAILABLE) == EVIDENCE_UNAVAILABLE:
            pub = srow.get("operator_org") or srow.get("governance_body")
            if pd.notna(pub) and str(pub).strip():
                rows[sid]["publisher"] = str(pub).strip()
        if rows[sid].get("url", EVIDENCE_UNAVAILABLE) == EVIDENCE_UNAVAILABLE:
            website = rows[sid].get("official_website")
            if website and website != EVIDENCE_UNAVAILABLE:
                rows[sid]["url"] = website

    records = []
    for sid in base_ids:
        rec = rows.get(sid, {"system_id": sid})
        row = {"system_id": sid}
        for col in EVIDENCE_COLUMNS:
            if col == "system_id":
                continue
            row[col] = _clean(rec.get(col))
        records.append(row)

    return pd.DataFrame(records)
