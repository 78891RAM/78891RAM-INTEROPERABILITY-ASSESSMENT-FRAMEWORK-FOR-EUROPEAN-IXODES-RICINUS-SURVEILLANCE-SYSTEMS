"""
Evaluates a completed blind re-scoring worksheet against the original scores
in data/systems.csv — this is the code that would produce a genuine,
defensible "N/50" test-retest agreement number, as opposed to the unbacked
"49/50" figure that appeared in the dissertation with no computation behind
it anywhere in this repository (see the methodology-audit conversation).

Input: a CSV with the same shape as the worksheet scripts/build produces —
one row per re-scored system, with system_id plus the 10 criterion columns
filled in with fresh 0/1/2 judgments. Only system_id and the 10 criterion
columns are read; any evidence columns alongside them are ignored, so the
same file the rater filled in can be passed directly.

Usage:
    tick/venv/bin/python scripts/evaluate_retest.py <path-to-filled-worksheet.csv>
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
TICK_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(TICK_ROOT))

from core.validation import total_score, weighted_readiness_class  # noqa: E402
from core.barriers import classify_barriers  # noqa: E402
from core.integration import classify_integration_readiness  # noqa: E402

SYSTEMS_CSV = TICK_ROOT / "data" / "systems.csv"

CRITERIA = [
    "api_availability", "schema_completeness", "semantic_alignment", "license_openness",
    "spatial_resolution", "documentation_quality", "governance_gdpr_clarity", "data_quality",
    "update_frequency", "temporal_coverage",
]


def _classify(systems_subset: pd.DataFrame) -> pd.DataFrame:
    """Run the same real pipeline the live app uses (core.validation ->
    core.barriers -> core.integration) on a small subset of systems, so the
    "does the verdict change" comparison below is never a re-implementation
    of the classification rules — it's the actual rules, run twice."""
    df = systems_subset.copy()
    df["total_score"] = df.apply(total_score, axis=1)
    df["readiness_class"] = df["total_score"].apply(weighted_readiness_class)
    barriers = classify_barriers(df)
    integration = classify_integration_readiness(df, barriers)
    return integration.set_index("system_id")


def evaluate(retest_path: Path) -> None:
    original = pd.read_csv(SYSTEMS_CSV, dtype={"system_id": str}).set_index("system_id")
    retest = pd.read_csv(retest_path, dtype={"system_id": str}).set_index("system_id")

    missing_systems = set(retest.index) - set(original.index)
    if missing_systems:
        raise ValueError(f"system_id(s) not found in data/systems.csv: {missing_systems}")

    missing_criteria = [c for c in CRITERIA if c not in retest.columns]
    if missing_criteria:
        raise ValueError(f"Retest file is missing criterion column(s): {missing_criteria}")

    unfilled = retest[CRITERIA].isna() | (retest[CRITERIA].astype(str).apply(lambda s: s.str.strip()) == "")
    if unfilled.any().any():
        blank_cells = [
            f"{sid}.{crit}"
            for sid in retest.index
            for crit in CRITERIA
            if unfilled.loc[sid, crit]
        ]
        raise ValueError(f"Retest file has unfilled score cell(s): {blank_cells}")

    total_checked = 0
    total_agree = 0
    disagreements = []

    for sid in retest.index:
        for crit in CRITERIA:
            orig_val = int(original.loc[sid, crit])
            new_val = int(retest.loc[sid, crit])
            total_checked += 1
            if orig_val == new_val:
                total_agree += 1
            else:
                disagreements.append({
                    "system_id": sid,
                    "system_name": original.loc[sid, "system_name"],
                    "criterion": crit,
                    "original_score": orig_val,
                    "retest_score": new_val,
                })

    print(f"Systems re-scored: {len(retest.index)} ({', '.join(retest.index)})")
    print(f"Criterion-scores compared: {total_checked}")
    print(f"Agreement: {total_agree}/{total_checked}  ({100 * total_agree / total_checked:.1f}%)")
    print()

    if disagreements:
        print("Disagreements:")
        for d in disagreements:
            print(
                f"  {d['system_id']} ({d['system_name']}) — {d['criterion']}: "
                f"original={d['original_score']}, retest={d['retest_score']}"
            )
    else:
        print("No disagreements — every criterion-score matched.")

    # --- Does any criterion disagreement actually change the verdict? ---
    # Raw criterion agreement (above) answers "is scoring reproducible";
    # this answers the more consequential question: for a viva, "does it
    # matter" — a disagreement on a criterion the hard gate doesn't touch,
    # that also doesn't move total_score across a High/Medium/Low threshold,
    # leaves the dissertation's headline classification unchanged. The hard
    # gate rule itself is deterministic code (core.integration.HARD_GATE_CRITERIA,
    # already covered by tests/test_core.py::test_hard_gate_caps_high_scoring_zero_api_system)
    # — it is not re-scored here, it is re-*applied*, identically, to both
    # the original and retest criterion sets, via the same real pipeline
    # functions the live app uses.
    print()
    print("=" * 70)
    print("Downstream verdict comparison (original vs retest criterion scores)")
    print("=" * 70)

    original_subset = original.loc[retest.index].reset_index()
    retest_full = original_subset.copy()
    for crit in CRITERIA:
        retest_full[crit] = [int(retest.loc[sid, crit]) for sid in retest_full["system_id"]]

    original_verdicts = _classify(original_subset)
    retest_verdicts = _classify(retest_full)

    verdict_cols = ["readiness_class", "barrier_level", "integration_class", "hard_gate_failed"]
    any_verdict_changed = False
    for sid in retest.index:
        o = original_verdicts.loc[sid]
        r = retest_verdicts.loc[sid]
        changed = [c for c in verdict_cols if o[c] != r[c]]
        name = original.loc[sid, "system_name"]
        if changed:
            any_verdict_changed = True
            print(f"  {sid} ({name}): VERDICT CHANGED — " + "; ".join(
                f"{c}: {o[c]} -> {r[c]}" for c in changed
            ))
        else:
            print(f"  {sid} ({name}): verdict unchanged "
                  f"(readiness={o['readiness_class']}, barrier={o['barrier_level']}, "
                  f"integration={o['integration_class']}, hard_gate_failed={o['hard_gate_failed']})")

    print()
    if any_verdict_changed:
        print("At least one system's final classification changed under the retest scores — "
              "report this explicitly, it's the strongest evidence a reviewer would ask about.")
    else:
        print("No system's final classification changed, even where individual criteria disagreed — "
              "the headline verdicts are robust to the observed scoring variation.")

    out_path = retest_path.parent / f"{retest_path.stem}_evaluation.csv"
    pd.DataFrame(disagreements).to_csv(out_path, index=False)
    print(f"\nDisagreement detail saved to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: venv/bin/python scripts/evaluate_retest.py <filled-worksheet.csv>")
        sys.exit(1)
    evaluate(Path(sys.argv[1]))
