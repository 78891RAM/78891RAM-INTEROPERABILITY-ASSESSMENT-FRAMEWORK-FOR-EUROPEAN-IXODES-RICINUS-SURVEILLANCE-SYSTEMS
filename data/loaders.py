"""Disk I/O — sole module that reads CSV files from disk."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import EVIDENCE_CSV, SYSTEMS_CSV


def load_systems(path: Path | None = None) -> pd.DataFrame:
    p = path or SYSTEMS_CSV
    return pd.read_csv(p, dtype={'system_id': str})


def load_evidence(path: Path | None = None) -> pd.DataFrame:
    p = path or EVIDENCE_CSV
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, dtype={'system_id': str})
