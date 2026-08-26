"""Overview tab layout."""

from __future__ import annotations

from dash import html, dcc

from core.kpi import compute_overview_kpis
from data.pipeline import FrameworkSnapshot
from ui.cards import kpi_card
from ui.downloads import download_button, register_download
from ui.figures import readiness_pie, score_histogram
from ui.styles import (
    CHART_CELL, CHART_ROW, KPI_ROW, MUTED, THEME_GREEN, CONTENT_CARD, 
    HEADING_2, BODY_TEXT, TEXT_MUTED
)


def layout(snapshot: FrameworkSnapshot) -> html.Div:
    """Build overview KPI cards and summary charts."""
    kpis = compute_overview_kpis(snapshot.systems, snapshot.integration)
    if not snapshot.ok:
        return html.Div(html.P("No systems loaded.", style=MUTED))

    # Headline: barrier- and hard-gate-adjusted integration readiness (see
    # core.integration.classify_integration_readiness) — this is the primary
    # verdict. Raw total_score is shown after it as a supporting figure, not
    # the final answer, per the scorecard-validation audit.
    cards = html.Div(
        className="kpi-row",
        style=KPI_ROW,
        children=[
            kpi_card("Total Systems", str(kpis.total_systems)),
            kpi_card("High Integration Ready", str(kpis.high_integration_ready), tone="high"),
            kpi_card("Medium Integration Ready", str(kpis.medium_integration_ready), tone="medium"),
            kpi_card("Low Integration Ready", str(kpis.low_integration_ready), tone="low"),
            kpi_card("Average Score", f"{kpis.average_score}/20"),
            kpi_card("Score Range", f"{kpis.lowest_score}–{kpis.highest_score}/20"),
        ],
    )
    supporting_caption = html.P(
        f"Supporting figure — score-only readiness before barrier/hard-gate adjustment: "
        f"{kpis.high_readiness} High · {kpis.medium_readiness} Medium · {kpis.low_readiness} Low. "
        "See the Integration tab for barrier detail and the hard-gate rule.",
        style={**MUTED, "marginTop": "-8px", "marginBottom": "16px"},
    )
    charts = html.Div(
        className="chart-row",
        style=CHART_ROW,
        children=[
            html.Div(
                [download_button("dl-readiness-pie"), dcc.Graph(figure=readiness_pie(snapshot.systems))],
                style=CHART_CELL,
            ),
            html.Div(
                [download_button("dl-score-histogram"), dcc.Graph(figure=score_histogram(snapshot.systems))],
                style=CHART_CELL,
            ),
        ],
    )

    # Canonical system-by-system table lives on Integration (total_score,
    # readiness_class, barrier_level and integration_class together) — Overview
    # doesn't repeat it, per the UI-duplication audit.
    integration_note = html.Div(
        style={
            **CONTENT_CARD,
            "borderLeft": f"4px solid {THEME_GREEN}",
            "background": "#f8fdf9",
        },
        children=[
            html.H3(
                "Complete Integration Assessment",
                style={**HEADING_2, "fontSize": "1.1rem", "marginBottom": "8px", "marginTop": "0"},
            ),
            html.P(
                "For the full system-by-system table including scores, readiness classification, "
                "barrier levels, and the barrier-adjusted integration verdict, see the Integration tab.",
                style={**BODY_TEXT, "color": TEXT_MUTED, "margin": "0"},
            ),
        ],
    )
    
    return html.Div(
        style={"padding": "8px 0"},
        children=[cards, supporting_caption, charts, integration_note]
    )


def register_callbacks(app, snapshot: FrameworkSnapshot) -> None:
    """Wire the two 'Download figure (PNG)' buttons above — same figures,
    filenames, and dimensions as export_dissertation_figures.py."""
    register_download(
        app, "dl-readiness-pie", "fig_overview_readiness_distribution.png",
        lambda: readiness_pie(snapshot.systems), width=1000, height=800,
    )
    register_download(
        app, "dl-score-histogram", "fig_overview_score_distribution.png",
        lambda: score_histogram(snapshot.systems), width=1300, height=800,
    )
