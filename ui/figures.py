"""Shared Plotly figure builders for Dash tabs."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from config import BARRIER_SEVERITY_COLORS, CRITERIA, CRITERIA_LABELS, READINESS_COLORS
from ui.styles import CHART_MARGIN


def readiness_pie(systems_df: pd.DataFrame) -> go.Figure:
    if systems_df.empty:
        return go.Figure()

    counts = systems_df["readiness_class"].value_counts().reset_index()
    counts.columns = ["readiness_class", "count"]

    fig = px.pie(
        counts,
        names="readiness_class",
        values="count",
        color="readiness_class",
        color_discrete_map=READINESS_COLORS,
        hole=0.35,
        title="Readiness Distribution",
    )
    fig.update_layout(margin=dict(t=50, b=30, l=20, r=20), height=360)
    return fig


def score_histogram(systems_df: pd.DataFrame) -> go.Figure:
    if systems_df.empty:
        return go.Figure()

    fig = px.histogram(
        systems_df,
        x="total_score",
        nbins=10,
        title="Score Distribution (0–20)",
    )
    fig.update_layout(
        margin=dict(t=50, b=30, l=20, r=20),
        height=360,
        xaxis_title="Total Score (0–20)",
        yaxis_title="Systems",
    )
    return fig


def criteria_heatmap(systems_df: pd.DataFrame) -> go.Figure:
    if systems_df.empty:
        return go.Figure()

    src = systems_df.set_index("system_id")[CRITERIA].copy()
    src.columns = [CRITERIA_LABELS[c] for c in CRITERIA]

    fig = px.imshow(
        src,
        aspect="auto",
        zmin=0,
        zmax=2,
        color_continuous_scale="YlGn",
    )
    fig.update_layout(
        margin=CHART_MARGIN,
        height=420,
        xaxis_title="Criterion",
        yaxis_title="System",
    )
    fig.update_xaxes(tickangle=-40)
    return fig


def ranking_bar(systems_df: pd.DataFrame) -> go.Figure:
    if systems_df.empty:
        return go.Figure()

    ranked = systems_df.sort_values("total_score", ascending=True)

    fig = px.bar(
        ranked,
        x="total_score",
        y="system_name",
        orientation="h",
        color="readiness_class",
        color_discrete_map=READINESS_COLORS,
        title="System Ranking",
        labels={"readiness_class": "Readiness"},
    )
    fig.update_layout(
        margin=CHART_MARGIN,
        height=480,
        xaxis_title="Total Score (0–20)",
        yaxis_title="",
        legend_title_text="Readiness",
    )
    return fig
