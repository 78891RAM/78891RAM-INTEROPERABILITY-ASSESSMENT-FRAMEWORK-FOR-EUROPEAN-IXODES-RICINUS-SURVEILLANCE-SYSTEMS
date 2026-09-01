"""
Builds a blind re-scoring worksheet for a test-retest reliability check on
the interoperability scorecard — see the methodology-audit conversation for
why (the dissertation's "49/50" claim has no computation or record behind
it anywhere in this repo).

Selects N systems (random, fixed seed, disclosed in the printed output so
selection is never arbitrary/cherry-picked), then strips every score and
every audit trail out of their evidence before writing the worksheet — the
rater sees only what a first-time reviewer would see: raw metadata fields.

Two note formats carry score information and both are stripped:
  - "... SCORING AUDIT: <correction detail>" (six systems have this)
  - "... FLAGGED (not rescored, ...): <detail that names specific scores>"
    which can appear WITHOUT a preceding SCORING AUDIT sentence (SYS09) —
    an earlier version of this script only split on "SCORING AUDIT:" and
    missed this case, leaking SYS09's schema_completeness/data_quality
    scores into what was meant to be a blind worksheet.

Usage:
    tick/venv/bin/python scripts/build_retest_worksheet.py [--seed N] [--n N] [--exclude SYS01,SYS02,...]
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
TICK_ROOT = SCRIPT_DIR.parent
SYSTEMS_CSV = TICK_ROOT / "data" / "systems.csv"

CRITERIA = [
    "api_availability", "schema_completeness", "semantic_alignment", "license_openness",
    "spatial_resolution", "documentation_quality", "governance_gdpr_clarity", "data_quality",
    "update_frequency", "temporal_coverage",
]
EVIDENCE_COLS = [
    "system_id", "system_name", "operator_org", "countries_covered", "data_type",
    "access_method", "update_freq_desc", "spatial_res_desc", "temporal_cov_desc",
    "license_type", "governance_body", "standards_used", "notes",
]


def _strip_score_hints(notes: str) -> str:
    """Remove both the SCORING AUDIT trail and any standalone FLAGGED note —
    either can name a specific score. Whichever marker appears first wins;
    everything from that point to the end of the string is dropped."""
    s = str(notes)
    cut_points = [m.start() for m in re.finditer(r"\s*(SCORING AUDIT|FLAGGED)\s*[:\(]", s)]
    if cut_points:
        s = s[: min(cut_points)]
    return s.strip()


def build(n: int, seed: int, exclude: list[str], out_path: Path) -> list[str]:
    df = pd.read_csv(SYSTEMS_CSV, dtype={"system_id": str})
    pool = [s for s in df["system_id"] if s not in exclude]
    if len(pool) < n:
        raise ValueError(f"Only {len(pool)} systems available after excluding {exclude}, need {n}")

    random.seed(seed)
    selected = sorted(random.sample(pool, n))
    print(f"Selected (seed={seed}, excluding {exclude or 'none'}): {selected}")

    sub = df[df["system_id"].isin(selected)].set_index("system_id").loc[selected].reset_index()

    # Answer key — kept out of the worksheet, for later comparison only.
    answer_key = sub[["system_id", "system_name"] + CRITERIA].copy()
    answer_key_path = out_path.parent / f"{out_path.stem}_answer_key.csv"
    answer_key.to_csv(answer_key_path, index=False)

    worksheet = sub[EVIDENCE_COLS].copy()
    worksheet["notes"] = worksheet["notes"].apply(_strip_score_hints)
    for c in CRITERIA:
        worksheet[c] = ""

    worksheet.to_csv(out_path, index=False)
    print(f"Worksheet written to {out_path}")
    print(f"Answer key written to {answer_key_path} (do not open until scoring is complete)")
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--exclude", type=str, default="")
    parser.add_argument("--out", type=str, default=str(TICK_ROOT.parent / "retest_worksheet.csv"))
    args = parser.parse_args()

    exclude_list = [s.strip() for s in args.exclude.split(",") if s.strip()]
    build(args.n, args.seed, exclude_list, Path(args.out))
