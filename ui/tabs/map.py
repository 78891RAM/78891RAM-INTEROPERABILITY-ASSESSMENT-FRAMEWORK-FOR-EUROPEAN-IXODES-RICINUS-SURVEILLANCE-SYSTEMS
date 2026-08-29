"""Geographical map tab layout."""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html

from core.geo import (
    COUNTRY_CENTROIDS,
    COUNTRY_NAMES,
    country_label_from_iso3_all,
    short_country_label_from_iso3_all,
)
from core.validation import weighted_readiness_class
from data.pipeline import FrameworkSnapshot
from ui.downloads import download_button, register_download
from ui.maps.interoperability_map import build_interoperability_map
from ui.styles import BLOCK, MUTED, THEME_BLUE
from ui.tables import make_table

MAP_TITLE = "European Surveillance Systems — Interoperability Map (2026)"

MAP_READINESS_COLORS = {
    "High": "#27ae60",
    "Medium": "#f39c12",
    "Low": "#e74c3c",
    "Unknown": "#95a5a6",
}

READINESS_LEGEND_ORDER = ["High", "Medium", "Low"]


def _normalise_readiness(series: pd.Series) -> pd.Series:
    out = series.fillna("Unknown").astype(str).replace("", "Unknown")
    return out.where(out.isin(READINESS_LEGEND_ORDER + ["Unknown"]), "Unknown")


def _split_country_and_marker_rows(scatter: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Single-country systems fill their own country polygon (one clean shape, one
    label). EU-wide / multi-country systems can't be pinned to one polygon, so
    they stay as markers.
    """
    df = scatter.copy()
    df["readiness_class"] = _normalise_readiness(df["readiness_class"])
    df["total_score"] = pd.to_numeric(df["total_score"], errors="coerce").fillna(0)
    iso3_all = df["iso3_all"].fillna("").astype(str)
    has_single_polygon = (
        (iso3_all != "")
        & ~iso3_all.str.contains(",")
        & (iso3_all != "EUR")
        & df["iso3"].isin(COUNTRY_CENTROIDS)
    )
    return df.loc[has_single_polygon].copy(), df.loc[~has_single_polygon].copy()


def _build_country_fill(country_rows: pd.DataFrame) -> tuple[list[go.Choropleth], go.Scattergeo]:
    """One flat-coloured polygon per country, plus a country-name label on top of it."""
    grouped = (
        country_rows.groupby("iso3")
        .agg(
            total_score=("total_score", "mean"),
            system_name=("system_name", lambda s: "; ".join(s)),
            system_count=("system_id", "count"),
        )
        .reset_index()
    )
    grouped["readiness_class"] = grouped["total_score"].apply(weighted_readiness_class)
    grouped["country_name"] = grouped["iso3"].map(COUNTRY_NAMES)
    grouped["lat"] = grouped["iso3"].map(lambda c: COUNTRY_CENTROIDS[c][0])
    grouped["lon"] = grouped["iso3"].map(lambda c: COUNTRY_CENTROIDS[c][1])
    grouped["hover"] = grouped.apply(
        lambda r: (
            f"<b>{r['country_name']}</b><br>"
            f"{int(r['system_count'])} system(s): {r['system_name']}<br>"
            f"Avg. score: {r['total_score']:.1f}/20 · Readiness: {r['readiness_class']}"
        ),
        axis=1,
    )

    choropleths: list[go.Choropleth] = []
    for bucket in READINESS_LEGEND_ORDER:
        subset = grouped.loc[grouped["readiness_class"] == bucket]
        if subset.empty:
            continue
        color = MAP_READINESS_COLORS[bucket]
        choropleths.append(
            go.Choropleth(
                locations=subset["iso3"],
                locationmode="ISO-3",
                z=[1] * len(subset),
                zmin=0,
                zmax=1,
                colorscale=[[0, color], [1, color]],
                showscale=False,
                marker_line_color="#ffffff",
                marker_line_width=1.3,
                name=bucket,
                legendgroup=bucket,
                showlegend=True,
                text=subset["hover"],
                hoverinfo="text",
            )
        )

    label_trace = go.Scattergeo(
        lat=grouped["lat"],
        lon=grouped["lon"],
        mode="text",
        text=grouped["country_name"],
        # Dark, not white: several fills here are small enough that the label spills onto
        # the near-white basemap, where white text disappears entirely.
        textfont=dict(size=12, color="#1a1a1a", family="system-ui, sans-serif"),
        hoverinfo="skip",
        showlegend=False,
    )
    return choropleths, label_trace


def _spread_marker_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Fan out exact-duplicate centroids (e.g. the three EU-wide systems) for display only."""
    out = df.copy()
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")

    groups: dict[tuple[float, float], list[int]] = {}
    for idx, row in out.iterrows():
        if pd.isna(row["lat"]) or pd.isna(row["lon"]):
            continue
        key = (round(float(row["lat"]), 2), round(float(row["lon"]), 2))
        groups.setdefault(key, []).append(idx)

    for (_base_lat, _base_lon), indices in groups.items():
        if len(indices) < 2:
            continue
        ordered = sorted(indices, key=lambda ix: str(out.at[ix, "system_id"]))
        base_lat = float(out.at[ordered[0], "lat"])
        base_lon = float(out.at[ordered[0], "lon"])
        radius = 1.6 if len(ordered) == 2 else 2.0
        for i, idx in enumerate(ordered):
            angle = (2 * math.pi * i / len(ordered)) - (math.pi / 2)
            lon_scale = 1.35  # degrees longitude appear narrower at mid-latitudes
            out.at[idx, "lat"] = base_lat + radius * math.cos(angle)
            out.at[idx, "lon"] = base_lon + radius * math.sin(angle) * lon_scale

    return out


# Display-only placement for EU-wide systems, over open water north of the Netherlands —
# COUNTRY_CENTROIDS["EUR"] sits at the geometric mean of all member states, which happens to
# land on Germany's Baltic coastline and reads as "this EU-wide system belongs to Germany".
_EUR_DISPLAY_LOCATION = (58.5, 3.0)


def _build_marker_traces(marker_rows: pd.DataFrame, legend_added: set[str]) -> list[go.Scattergeo]:
    """EU-wide / multi-country systems — no single polygon to fill, so shown as markers."""
    df = marker_rows.copy()
    df["country_label"] = df["iso3_all"].map(country_label_from_iso3_all)
    # "Europe (EU-wide)" is a recognisable idiom and stays as-is; genuine multi-country
    # rows (e.g. "France, Denmark, Netherlands") get the compact ISO-2 form on the map
    # itself so the text footprint doesn't swallow neighbouring small-country labels —
    # the full names are still in the hover text below.
    is_multi = df["iso3_all"].str.contains(",")
    df["map_label"] = df["country_label"].where(~is_multi, df["iso3_all"].map(short_country_label_from_iso3_all))
    is_eur = df["iso3_all"] == "EUR"
    df.loc[is_eur, "lat"] = _EUR_DISPLAY_LOCATION[0]
    df.loc[is_eur, "lon"] = _EUR_DISPLAY_LOCATION[1]
    df = _spread_marker_rows(df)
    df["hover"] = df.apply(
        lambda r: (
            f"<b>{r['system_name']}</b><br>"
            f"Region: {r['country_label']}<br>"
            f"Score: {r['total_score']:.0f}/20 · Readiness: {r['readiness_class']}<br>"
            f"Barriers: {r.get('barrier_level', '—')} · Integration: {r.get('integration_class', '—')}"
        ),
        axis=1,
    )

    positions = ["bottom center", "top center", "middle right", "middle left"]
    traces: list[go.Scattergeo] = []
    for bucket in READINESS_LEGEND_ORDER:
        subset = df.loc[df["readiness_class"] == bucket]
        if subset.empty:
            continue
        color = MAP_READINESS_COLORS[bucket]
        show_in_legend = bucket not in legend_added
        legend_added.add(bucket)
        sizes = 15 + (pd.to_numeric(subset["total_score"], errors="coerce").fillna(0) / 20) * 13
        traces.append(
            go.Scattergeo(
                lat=subset["lat"],
                lon=subset["lon"],
                mode="markers+text",
                text=subset["map_label"],
                textposition=[positions[i % len(positions)] for i in range(len(subset))],
                textfont=dict(size=10, color="#2c3e50"),
                marker=dict(size=sizes, color=color, line=dict(width=1.4, color="#ffffff")),
                hovertext=subset["hover"],
                hoverinfo="text",
                name=bucket,
                legendgroup=bucket,
                showlegend=show_in_legend,
            )
        )
    return traces


def _build_map_figure(scatter: pd.DataFrame) -> go.Figure:
    """Europe map — single-country systems fill their own polygon; EU-wide / multi-country systems are markers."""
    country_rows, marker_rows = _split_country_and_marker_rows(scatter)

    legend_added: set[str] = set()
    choropleths, label_trace = _build_country_fill(country_rows) if not country_rows.empty else ([], None)
    legend_added.update(t.name for t in choropleths)
    marker_traces = _build_marker_traces(marker_rows, legend_added) if not marker_rows.empty else []

    fig = go.Figure(data=[*choropleths, *marker_traces] + ([label_trace] if label_trace is not None else []))

    fig.update_geos(
        scope="europe",
        projection_type="natural earth",
        showcountries=True,
        showland=True,
        showocean=True,
        landcolor="#f4f6f7",
        oceancolor="#e8f4f8",
        countrycolor="#bdc3c7",
        coastlinecolor="#95a5a6",
        lataxis_range=[34, 62],
        lonaxis_range=[-12, 32],
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        title=dict(
            text=MAP_TITLE,
            font=dict(size=16, color=THEME_BLUE),
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
        ),
        margin=dict(l=4, r=4, t=56, b=4),
        height=560,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="system-ui, sans-serif", size=13, color="#2c3e50"),
        legend=dict(
            title=dict(text="Readiness", font=dict(size=12, color="#2c3e50")),
            orientation="v",
            yanchor="bottom",
            y=0.04,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(255, 255, 255, 0.94)",
            bordercolor="#bdc3c7",
            borderwidth=1,
            font=dict(size=11),
            traceorder="normal",
            itemsizing="constant",
        ),
        showlegend=True,
    )
    return fig


def layout(snapshot: FrameworkSnapshot) -> html.Div:
    """Build Europe choropleth + marker map from precomputed map scatter data."""
    scatter = snapshot.map_scatter
    if scatter.empty or not snapshot.map_result.has_points:
        return html.P("No mappable systems. Check countries_covered in systems.csv.")

    source_block = html.Div(
        style={**MUTED, "marginTop": "10px", "lineHeight": "1.55", "maxWidth": "900px"},
        children=[
            html.P(
                "Colour = interoperability readiness. Single-country systems fill that "
                "country's shape (labelled directly on the map); EU-wide and multi-country "
                "systems, which don't map to one polygon, are shown as sized dots instead — "
                "size = total score (0–20). Hover any country or dot for full system detail.",
                style={"margin": "0 0 8px"},
            ),
            html.P(
                "Source: Author-compiled system metadata, 2026. "
                "This map locates assessed surveillance programmes — it does not show "
                "Ixodes ricinus occurrence or vector distribution. For species distribution maps, "
                "see ECDC / VectorNet tick maps (regional administrative units).",
                style={"margin": 0, "fontSize": "0.85rem", "color": "#7f8c8d"},
            ),
        ],
    )
    skip_table = (
        make_table(
            data=snapshot.map_result.skipped_details,
            columns=[{"name": "System", "id": "system_id"}, {"name": "Reason", "id": "reason"}],
            page_size=10,
        )
        if snapshot.map_result.skipped_details
        else html.Div()
    )

    return html.Div([
        html.Div(
            style=BLOCK,
            children=[
                download_button("dl-map"),
                build_interoperability_map(scatter),
                source_block,
            ],
        ),
        skip_table,
    ])


def register_callbacks(app, snapshot: FrameworkSnapshot) -> None:
    """Wire the 'Download figure (PNG)' button above — same figure, filename,
    and dimensions as export_dissertation_figures.py."""
    register_download(
        app, "dl-map", "fig_map_interoperability_europe.png",
        lambda: _build_map_figure(snapshot.map_scatter), width=1500, height=1000,
    )
