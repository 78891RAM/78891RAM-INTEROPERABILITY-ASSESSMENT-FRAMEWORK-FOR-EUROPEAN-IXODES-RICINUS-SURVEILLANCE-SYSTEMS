"""
Robustness check: does the readiness classification depend strongly on the
equal-weight baseline, or is it stable under other reasonable weightings?

core.validation.total_score sums all 10 criteria unweighted (weight=1 each,
0-20 max) — see that module's own comment: "these groups are a reporting
split, not a re-weighting." This script is the code that actually tests
what the dissertation's "robustness check" section describes: whether
alternative, principled weighting schemes would classify systems
differently, without asserting any of them is the "correct" weighting
(there isn't a stated criterion for choosing one).

Three alternative schemes, each with a stated rationale:
  - technical_emphasis:  TECHNICAL_SUBSCORE_CRITERIA x1.5, GOVERNANCE x0.5
                          (machine interoperability is primarily a technical-
                          access problem)
  - governance_emphasis: GOVERNANCE_SUBSCORE_CRITERIA x1.5, TECHNICAL x0.5
                          (legal/stewardship barriers often block integration
                          even when the technical capability exists)
  - access_critical:     api_availability and license_openness x2, rest x1
                          (the two criteria the existing hard-gate logic
                          already treats as the most practically blocking)

Each scheme's weighted total is rescaled back onto the same 0-20 range as
the baseline (so the existing High >=15 / Medium 10-14 / Low <10 thresholds
stay meaningfully comparable), then readiness_class and integration_class
are recomputed via the real pipeline (core.validation -> core.barriers ->
core.integration — not a reimplementation) and compared per system against
the equal-weight baseline.

Usage: tick/venv/bin/python scripts/weighting_robustness.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
TICK_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(TICK_ROOT))

from core.validation import (  # noqa: E402
    CRITERIA, TECHNICAL_SUBSCORE_CRITERIA, GOVERNANCE_SUBSCORE_CRITERIA, weighted_readiness_class,
)
from core.barriers import classify_barriers  # noqa: E402
from core.integration import classify_integration_readiness  # noqa: E402

SYSTEMS_CSV = TICK_ROOT / "data" / "systems.csv"
MAX_RAW = 2 * len(CRITERIA)  # 20, same scale the baseline uses

SCHEMES: dict[str, dict[str, float]] = {
    "equal_baseline": {c: 1.0 for c in CRITERIA},
    "technical_emphasis": {
        **{c: 1.5 for c in TECHNICAL_SUBSCORE_CRITERIA},
        **{c: 0.5 for c in GOVERNANCE_SUBSCORE_CRITERIA},
    },
    "governance_emphasis": {
        **{c: 0.5 for c in TECHNICAL_SUBSCORE_CRITERIA},
        **{c: 1.5 for c in GOVERNANCE_SUBSCORE_CRITERIA},
    },
    "access_critical": {
        **{c: 1.0 for c in CRITERIA},
        "api_availability": 2.0,
        "license_openness": 2.0,
    },
}


def _weighted_total(row: pd.Series, weights: dict[str, float]) -> float:
    """Weighted sum of the 10 criteria, rescaled onto the same 0-20 range
    the equal-weight baseline uses (sum of weights varies per scheme, since
    e.g. access_critical's weights don't sum to 10 the way equal weights do)."""
    weighted_sum = sum(float(row[c]) * weights[c] for c in CRITERIA)
    max_possible = sum(2.0 * weights[c] for c in CRITERIA)
    return (weighted_sum / max_possible) * MAX_RAW


def _classify_under_scheme(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    out["total_score"] = out.apply(lambda r: _weighted_total(r, weights), axis=1)
    out["readiness_class"] = out["total_score"].apply(weighted_readiness_class)
    barriers = classify_barriers(out)
    integration = classify_integration_readiness(out, barriers)
    return integration.set_index("system_id")


def main() -> None:
    systems = pd.read_csv(SYSTEMS_CSV, dtype={"system_id": str})

    sid_order = systems["system_id"].tolist()
    # classify_integration_readiness sorts its output by rank, which differs
    # per scheme (different weights -> different score order) — reindex to a
    # fixed system_id order so Series comparisons below align by system, not
    # by row position.
    results = {name: _classify_under_scheme(systems, w).loc[sid_order] for name, w in SCHEMES.items()}
    baseline = results["equal_baseline"]

    print(f"Systems: {len(systems)}")
    print(f"Schemes compared: {list(SCHEMES)}")
    print()

    summary_rows = []
    for scheme_name in SCHEMES:
        if scheme_name == "equal_baseline":
            continue
        scheme = results[scheme_name]
        n_readiness_changed = (scheme["readiness_class"] != baseline["readiness_class"]).sum()
        n_integration_changed = (scheme["integration_class"] != baseline["integration_class"]).sum()
        summary_rows.append({
            "scheme": scheme_name,
            "readiness_class_changed": int(n_readiness_changed),
            "integration_class_changed": int(n_integration_changed),
            "of_n_systems": len(systems),
        })
        print(f"=== {scheme_name} ===")
        print(f"  readiness_class changed for {n_readiness_changed}/{len(systems)} systems")
        print(f"  integration_class changed for {n_integration_changed}/{len(systems)} systems")
        for sid in systems["system_id"]:
            b, s = baseline.loc[sid], scheme.loc[sid]
            if b["readiness_class"] != s["readiness_class"] or b["integration_class"] != s["integration_class"]:
                name = systems.loc[systems.system_id == sid, "system_name"].iloc[0]
                print(
                    f"    {sid} ({name}): "
                    f"readiness {b['readiness_class']}->{s['readiness_class']}, "
                    f"integration {b['integration_class']}->{s['integration_class']} "
                    f"(score {b['total_score']:.1f}->{s['total_score']:.1f})"
                )
        print()

    out_dir = TICK_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "weighting_robustness.csv"
    pd.DataFrame(summary_rows).to_csv(out_path, index=False)
    print(f"Summary saved to {out_path}")

    total_changed = sum(r["readiness_class_changed"] for r in summary_rows)
    if total_changed == 0:
        print("\nNo system's readiness_class changed under any alternative weighting tested — "
              "the equal-weight baseline's classifications are stable across these three schemes.")
    else:
        print(f"\n{total_changed} readiness_class change(s) observed across the three alternative "
              "schemes tested — report these explicitly rather than the bare claim that weighting "
              "doesn't matter.")


if __name__ == "__main__":
    main()
