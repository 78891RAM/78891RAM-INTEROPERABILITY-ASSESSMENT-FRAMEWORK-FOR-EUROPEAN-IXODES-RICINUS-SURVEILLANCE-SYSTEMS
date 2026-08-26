"""
Barrier Assessment tab layout.

UI/UX redesign only — every figure here is built from the same
core.barriers / core.barrier_details data and functions as before. No
scoring, barrier-severity, or data logic changed in this file.
"""

from __future__ import annotations

import plotly.express as px
from dash import html, dcc

from config import BARRIER_SEVERITY_COLORS
from core.barrier_details import barrier_severity_distribution
from core.barriers import BARRIER_TYPES, barriers_summary_chart_df
from data.pipeline import FrameworkSnapshot
from ui.cards import kpi_card
from ui.downloads import download_button, register_download
from ui.styles import BLOCK, CHART_MARGIN, KPI_ROW, MUTED, THEME_GREEN
from ui.tables import columns_from_ids, make_table

# barrier_summary is deliberately excluded from the RENDERED table only — it's
# a text restatement of the five structured barrier columns to its left
# (e.g. "Technical: Medium; Legal: High"), so showing both says the same
# thing twice. The field itself is untouched in core.barriers/the snapshot;
# this list only controls what this tab displays.
BARRIER_DISPLAY_COLS = [
    "system_id",
    "system_name",
    "technical_barrier",
    "semantic_barrier",
    "legal_barrier",
    "governance_barrier",
    "accessibility_barrier",
    "barrier_level",
]

PLOTLY_CONFIG = {"responsive": True, "displaylogo": False}

# Scoped to this tab only (not added to ui/styles.py) — CSS grid with
# auto-fit/minmax reflows column count by available width on its own, no
# @media breakpoints needed: 4-ish across on desktop, 2 on tablet, 1 on a
# small screen, matching the layout the redesign asked for.
_KPI_GRID = {
    "display": "grid",
    "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))",
    "gap": "16px",
    "marginBottom": "24px",
}
_CHART_GRID = {
    "display": "grid",
    # 280px, not e.g. 400px: minmax()'s minimum must fit inside the smallest
    # viewport being supported (the spec's own 390px phone test width) or the
    # grid track itself becomes the thing causing horizontal overflow.
    "gridTemplateColumns": "repeat(auto-fit, minmax(280px, 1fr))",
    "gap": "16px",
}
_SECTION_HEADING = {"color": THEME_GREEN, "fontSize": "1.15rem", "marginBottom": "12px"}


def build_severity_by_system_chart(chart_df):
    """Barrier Severity by System grouped bar chart — extracted unchanged
    from layout() so the dissertation figure export script can call it
    directly instead of duplicating the figure code."""
    if chart_df.empty:
        return None
    fig_bar = px.bar(
        chart_df,
        x="system_name",
        y="severity_rank",
        color="barrier_type",
        barmode="group",
        title="Barrier Severity by System",
        labels={"barrier_type": "Barrier Type", "severity_rank": "Severity", "system_name": ""},
    )
    fig_bar.update_layout(
        margin=CHART_MARGIN,
        height=420,
        xaxis_tickangle=-40,
        legend_title_text="Barrier Type",
        autosize=True,
    )
    return fig_bar


def build_severity_distribution_chart(sev):
    """Barrier Severity Distribution bar chart — extracted unchanged from
    layout() so the dissertation figure export script can call it directly
    instead of duplicating the figure code."""
    if sev.empty:
        return None
    fig_sev = px.bar(
        sev,
        x="severity",
        y="count",
        color="severity",
        color_discrete_map=BARRIER_SEVERITY_COLORS,
        title="Barrier Severity Distribution",
        labels={"severity": "Severity", "count": "Systems / issues"},
    )
    fig_sev.update_layout(
        margin=dict(t=50, b=40, l=40, r=20),
        height=360,
        # Severity is already labelled on the x-axis (High/Medium/Low) —
        # a legend repeating those same three labels via colour swatch
        # only is redundant, not informative.
        showlegend=False,
        autosize=True,
    )
    return fig_sev


def layout(snapshot: FrameworkSnapshot) -> html.Div:
    """Build the Barrier Assessment tab: header, KPIs, severity charts, detail tables."""
    barriers = snapshot.barriers
    details = snapshot.barrier_details

    header = html.Div(
        style={"marginBottom": "24px"},
        children=[
            html.H2("Barrier Assessment", style={"color": THEME_GREEN, "fontSize": "1.6rem", "marginBottom": "4px"}),
            html.P(
                "Assessment of technical, legal, accessibility, semantic and governance "
                "barriers across surveillance systems.",
                style={**MUTED, "margin": 0},
            ),
        ],
    )

    if barriers.empty:
        return html.Div([
            header,
            html.P("No systems match the selected filters.", style={**MUTED, "marginBottom": "4px"}),
            html.P("Try removing one or more filters.", style=MUTED),
        ])

    present_cols = [c for c in BARRIER_DISPLAY_COLS if c in barriers.columns]

    # --- KPIs (all computed directly from existing snapshot data — nothing invented) ---
    n_systems = len(barriers)
    n_categories = len(BARRIER_TYPES)
    n_high = int((barriers["barrier_level"] == "High").sum()) if "barrier_level" in barriers.columns else 0
    n_flagged = len(details) if details is not None else 0

    kpi_row = html.Div(
        style=_KPI_GRID,
        children=[
            kpi_card("Systems Assessed", str(n_systems)),
            kpi_card("Barrier Categories", str(n_categories)),
            kpi_card("High-Severity Systems", str(n_high), tone="low" if n_high else "neutral"),
            kpi_card("Flagged Issues", str(n_flagged)),
        ],
    )

    # --- Charts — same underlying data/functions as before; fig_pie (Barrier
    # Frequency) is no longer rendered here since fig_bar already breaks
    # frequency/severity down by system and fig_sev summarises severity
    # counts — a third chart repeating that same information added clutter,
    # not clarity. barrier_frequency_df() itself is untouched in
    # core.barrier_details, just no longer called from this tab. ---
    chart_df = barriers_summary_chart_df(barriers)
    fig_bar = build_severity_by_system_chart(chart_df)

    sev = barrier_severity_distribution(details)
    fig_sev = build_severity_distribution_chart(sev)

    chart_cells = [
        html.Div([download_button(dl_id), dcc.Graph(figure=fig, config=PLOTLY_CONFIG, style={"width": "100%"})])
        for fig, dl_id in ((fig_bar, "dl-barrier-severity-system"), (fig_sev, "dl-barrier-severity-dist"))
        if fig is not None
    ]

    severity_section = html.Div(
        style=BLOCK,
        children=[
            html.H3("Barrier Severity & Comparison", style=_SECTION_HEADING),
            html.Div(style=_CHART_GRID, children=chart_cells) if chart_cells else html.P("No chart data available.", style=MUTED),
        ],
    )

    summary_section = html.Div(
        style=BLOCK,
        children=[
            html.H3("Detailed Barrier Comparison", style={**_SECTION_HEADING, "marginBottom": "4px"}),
            html.P(
                "One row per system — severity across five barrier dimensions, rolled up "
                "into an overall level. Colour is a supporting cue; the High/Medium/Low "
                "text label is what carries the meaning.",
                style={**MUTED, "marginBottom": "12px"},
            ),
            make_table(
                data=barriers[present_cols].to_dict("records"),
                columns=columns_from_ids(present_cols),
                page_size=15,
                sort_action="native",
            ),
        ],
    )

    sections = [header, kpi_row, severity_section, summary_section]

    if details is not None and not details.empty:
        detail_cols = list(details.columns)
        sections.append(
            html.Div(
                style=BLOCK,
                children=[
                    html.H3("Barrier Detail — Reason & Recommendation", style={**_SECTION_HEADING, "marginBottom": "4px"}),
                    html.P(
                        f"Optional supporting detail — one row per system per barrier issue "
                        f"({len(details)} rows). Only Medium/High severities are listed.",
                        style={**MUTED, "marginBottom": "12px"},
                    ),
                    make_table(
                        data=details.to_dict("records"),
                        columns=columns_from_ids(detail_cols),
                        page_size=15,
                    ),
                ],
            )
        )

    return html.Div(sections)


def register_callbacks(app, snapshot: FrameworkSnapshot) -> None:
    """Wire the two 'Download figure (PNG)' buttons above — same figures,
    filenames, and dimensions (dynamic on system count) as
    export_dissertation_figures.py."""
    barriers = snapshot.barriers
    details = snapshot.barrier_details
    n = len(barriers)
    register_download(
        app, "dl-barrier-severity-system", "fig_barriers_severity_by_system.png",
        lambda: build_severity_by_system_chart(barriers_summary_chart_df(barriers)),
        width=max(1600, 100 * n), height=950,
    )
    register_download(
        app, "dl-barrier-severity-dist", "fig_barriers_severity_distribution.png",
        lambda: build_severity_distribution_chart(barrier_severity_distribution(details)),
        width=1100, height=800,
    )
