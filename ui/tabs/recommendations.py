"""Recommendations tab layout."""

from __future__ import annotations

from dash import html

from data.pipeline import FrameworkSnapshot
from ui.styles import MUTED
from ui.tables import columns_from_ids, make_table

# One system can produce several recommendation rows (one per weak criterion —
# up to 7 for the lowest-scoring systems), so system_name repeats down the
# table. The default per-row zebra stripe cuts across those repeats with a
# pattern that has nothing to do with grouping, which is what made this table
# read as unclear (per the UI audit / user report). These two rules replace
# it with banding by system instead of by row, so each system's block of
# recommendations reads as one visual group.
_GROUP_BAND_STYLES = [
    {"if": {"filter_query": "{_system_band} = 0"}, "backgroundColor": "#ffffff"},
    {"if": {"filter_query": "{_system_band} = 1"}, "backgroundColor": "#eef2f5"},
]

# Extracted to a module constant (was inline in layout()) so the dissertation
# export script can select the same display columns/order from
# snapshot.recommendations.
RECOMMENDATIONS_TABLE_COLS = ["system_id", "system_name", "priority", "criterion", "recommendation"]


def layout(snapshot: FrameworkSnapshot) -> html.Div:
    """Build rule-based recommendations table."""
    recs = snapshot.recommendations
    if recs.empty:
        return html.P("No recommendations generated.")

    # Alternate a 0/1 band per distinct system_id, in the order systems first
    # appear — a hidden column (not in `columns`, so it never renders) used
    # only to key the group-banding style rules above.
    system_order = {sid: i % 2 for i, sid in enumerate(dict.fromkeys(recs["system_id"]))}
    banded = recs.copy()
    banded["_system_band"] = banded["system_id"].map(system_order)

    return html.Div([
        html.P(
            "Derived from scored criteria and documented metadata — not fabricated. Each "
            "shaded block below is one system — it can produce several rows, one per weak "
            "criterion.",
            style={**MUTED, "marginBottom": "16px"},
        ),
        make_table(
            data=banded.to_dict("records"),
            columns=columns_from_ids(RECOMMENDATIONS_TABLE_COLS),
            page_size=20,
            extra_style_data_conditional=_GROUP_BAND_STYLES,
        ),
    ])
