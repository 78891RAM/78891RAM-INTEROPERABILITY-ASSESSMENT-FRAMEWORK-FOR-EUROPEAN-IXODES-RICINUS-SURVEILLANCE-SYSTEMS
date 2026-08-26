"""Interoperability scores tab layout."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
from dash import html, dcc

from config import CRITERIA, CRITERIA_LABELS
from data.pipeline import FrameworkSnapshot
from ui.downloads import download_button, register_download
from ui.figures import criteria_heatmap, ranking_bar
from ui.styles import BLOCK, CHART_MARGIN, MUTED
from ui.tables import columns_from_ids, make_table

# Canonical systems table lives on Integration (total_score + readiness_class +
# barrier_level + integration_class in one place) — this tab's own table is
# trimmed to only what's specific to it: the technical/governance sub-scores.
SCORES_TABLE_COLS = ["system_id", "system_name", "technical_subscore", "governance_subscore", "total_score"]


def _avg_score_caption(avg: pd.DataFrame) -> str:
    """Derived from the live per-criterion averages so this can't drift out of
    sync with the scorecard the way a hardcoded criterion name could — see the
    UI audit finding that flagged the previous hardcoded version."""
    lowest = avg.sort_values("avg_score").iloc[0]
    return (
        f"{lowest['label']} is the weakest dimension across systems "
        f"(average {lowest['avg_score']:.2f}/2), indicating a primary interoperability barrier."
    )


def _criteria_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Per-criterion mean score, used by both the chart and its caption."""
    avg = df[CRITERIA].mean(numeric_only=True).reset_index()
    avg.columns = ["criterion", "avg_score"]
    avg["label"] = avg["criterion"].map(CRITERIA_LABELS)
    return avg


def build_avg_criteria_chart(avg: pd.DataFrame):
    """Average Score per Criterion bar chart — extracted unchanged from
    layout() so it can be reused by the dissertation figure export script
    without duplicating the figure-building code."""
    fig_avg = px.bar(
        avg,
        x="label",
        y="avg_score",
        title="Average Score per Criterion",
        color="avg_score",
        color_continuous_scale="Blues",
        labels={"label": "Criterion", "avg_score": "Average Score"},
    )
    fig_avg.update_layout(
        yaxis=dict(range=[0, 2.2], title="Average Score"),
        xaxis=dict(title="Criterion"),
        xaxis_tickangle=-40,
        height=360,
        margin=CHART_MARGIN,
        coloraxis_showscale=False,
    )
    return fig_avg


def layout(snapshot: FrameworkSnapshot) -> html.Div:
    """Build score ranking, heatmap, and average criteria chart."""
    df = snapshot.systems
    if df.empty:
        return html.P("No scored systems.")

    avg = _criteria_averages(df)
    fig_avg = build_avg_criteria_chart(avg)

    return html.Div([
        html.Div([download_button("dl-ranking-bar"), dcc.Graph(figure=ranking_bar(df))], style=BLOCK),
        html.Div([download_button("dl-criteria-heatmap"), dcc.Graph(figure=criteria_heatmap(df))], style=BLOCK),
        html.Div([
            download_button("dl-avg-criteria"),
            dcc.Graph(figure=fig_avg),
            html.P(_avg_score_caption(avg), style={**MUTED, "marginTop": "8px"}),
        ], style=BLOCK),
        html.P(
            "Technical and Governance are reporting sub-scores (0-10 each, summing to Total "
            "Score) — see Methodology. A strong Governance sub-score does not offset a weak "
            "Technical one, or vice versa. For readiness class, barrier level, and the full "
            "system-by-system table, see Integration.",
            style={**MUTED, "marginBottom": "8px"},
        ),
        make_table(
            data=df[SCORES_TABLE_COLS].to_dict("records"),
            columns=columns_from_ids(SCORES_TABLE_COLS),
            page_size=15,
        ),
    ])


def register_callbacks(app, snapshot: FrameworkSnapshot) -> None:
    """Wire the three 'Download figure (PNG)' buttons above — same figures,
    filenames, and dimensions (dynamic on system count) as
    export_dissertation_figures.py."""
    df = snapshot.systems
    n = len(df)
    register_download(
        app, "dl-ranking-bar", "fig_scores_system_ranking.png",
        lambda: ranking_bar(df), width=1500, height=max(900, 45 * n + 250),
    )
    register_download(
        app, "dl-criteria-heatmap", "fig_scores_criteria_heatmap.png",
        lambda: criteria_heatmap(df), width=1500, height=max(850, 40 * n + 250),
    )
    register_download(
        app, "dl-avg-criteria", "fig_scores_avg_per_criterion.png",
        lambda: build_avg_criteria_chart(_criteria_averages(df)), width=1400, height=800,
    )
