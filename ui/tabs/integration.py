"""Integration readiness tab layout."""

from __future__ import annotations

import plotly.express as px
from dash import html, dcc

from config import INTEGRATION_COLORS
from data.pipeline import FrameworkSnapshot
from ui.downloads import download_button, register_download
from ui.styles import BLOCK, CHART_MARGIN, MUTED
from ui.tables import columns_from_ids, make_table

# The canonical system-by-system table — extracted to a module constant
# (was a local var in layout()) so the dissertation export script can select
# the exact same display columns from snapshot.integration, in the same
# order, rather than dumping the full internal dataframe (which also carries
# working columns like hard_gate_failed / integration_rank / barrier_count).
INTEGRATION_TABLE_COLS = [
    "rank",
    "system_id",
    "system_name",
    "total_score",
    "readiness_class",
    "barrier_level",
    "integration_class",
    "barrier_summary",
]


def build_integration_chart(df):
    """Integration Readiness by System bar chart — extracted unchanged from
    layout() so the dissertation figure export script can call it directly
    instead of duplicating the figure code."""
    fig = px.bar(
        df,
        x="system_name",
        y="total_score",
        color="integration_class",
        color_discrete_map=INTEGRATION_COLORS,
        title="Integration Readiness by System",
        labels={"total_score": "Total Score (0–20)", "integration_class": "Classification"},
    )
    fig.update_layout(
        xaxis_tickangle=-40,
        height=420,
        margin={**CHART_MARGIN, "t": 90},
        xaxis_title="",
        yaxis_title="Total Score (0–20)",
        # Horizontal legend above the plot, not Plotly's default vertical
        # legend at the right — the default right-side legend didn't have
        # enough height for all 3 classifications and clipped "Low
        # integration ready" (12 of 14 systems) behind a scrollbar, same bug
        # as the Barriers tab's severity chart (see build_severity_by_system_chart).
        legend=dict(
            title_text="Classification",
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left", x=0,
        ),
    )
    return fig


def layout(snapshot: FrameworkSnapshot) -> html.Div:
    """Build integration decision table and bar chart."""
    df = snapshot.integration
    if df.empty:
        return html.P("No integration classification.")

    present = [c for c in INTEGRATION_TABLE_COLS if c in df.columns]

    fig = build_integration_chart(df)

    return html.Div([
        html.P(
            "This is the headline classification for the app (see Overview). "
            "High: score ≥ 15, low barriers, and passes the hard gate below · "
            "Low: score < 10 or high barriers · else Medium.",
            style={**MUTED, "marginBottom": "4px"},
        ),
        html.P(
            "Hard gate: a system scoring 0 on api_availability (no programmatic access to its "
            "data at all) cannot be \"High integration ready\" regardless of total score or "
            "barrier level — this is a named rule (core.integration.HARD_GATE_CRITERIA), not a "
            "side effect of the barrier scoring above it.",
            style={**MUTED, "marginBottom": "16px", "fontStyle": "italic"},
        ),
        make_table(
            data=df[present].to_dict("records"),
            columns=columns_from_ids(present),
            page_size=15,
        ),
        html.Div([download_button("dl-integration-chart"), dcc.Graph(figure=fig)], style=BLOCK),
    ])


def register_callbacks(app, snapshot: FrameworkSnapshot) -> None:
    """Wire the 'Download figure (PNG)' button above — same figure, filename,
    and dimensions (dynamic on system count) as export_dissertation_figures.py."""
    df = snapshot.integration
    register_download(
        app, "dl-integration-chart", "fig_integration_readiness_by_system.png",
        lambda: build_integration_chart(df) if not df.empty else None,
        width=max(1600, 100 * len(df)), height=950,
    )
