"""dash-leaflet ecological suitability map (items 3, 5-8, 12).

Three separate geographic concepts, never merged into one legend (item 2):
  - the continuous suitability/environmental raster (build_raster_overlay)
  - observed occurrence records (build_occurrence_layer) — observations, not predictions
  - study-country boundaries (build_country_boundary_layer)

Raster overlays are pre-generated, NaN-masked-to-real-borders PNGs (see
scripts/generate_raster_overlays.py) — this module only ever reads their
small metadata sidecar and points an ImageOverlay at the already-rendered
image; it never opens a GeoTIFF itself (item 18).
"""

from __future__ import annotations

import pandas as pd
import dash_leaflet as dl
from dash import html

from ui.maps.map_layers import (
    load_overlay_meta,
    load_study_country_boundaries,
    overlay_image_url,
    SUITABILITY_VMIN,
    SUITABILITY_VMAX,
)

# One radio list, not two (item 3's "MODEL OUTPUT" single-option radio and
# "ENVIRONMENTAL CONTEXT" radio overlapped on the same "Predicted
# environmental suitability" entry) — a single selector choosing which one
# continuous raster is visible covers both without a redundant control that
# picks between an option and itself.
RASTER_LAYER_OPTIONS = [
    {"label": "Predicted environmental suitability", "value": "suitability"},
    {"label": "Temperature seasonality (bio04)", "value": "bio04"},
    {"label": "Annual precipitation (bio12)", "value": "bio12"},
    {"label": "Precipitation seasonality (bio15)", "value": "bio15"},
    {"label": "NDVI", "value": "ndvi"},
]
DEFAULT_RASTER_LAYER = "suitability"

COUNTRY_VIEW_OPTIONS = [
    {"label": "All study countries", "value": "all"},
    {"label": "Austria", "value": "Austria"},
    {"label": "Croatia", "value": "Croatia"},
    {"label": "Estonia", "value": "Estonia"},
    {"label": "Ireland", "value": "Ireland"},
]
_COUNTRY_VIEWS = {
    "all": {"center": [50.5, 10.0], "zoom": 4},
    "Austria": {"center": [47.5, 14.4], "zoom": 7},
    "Croatia": {"center": [45.5, 16.3], "zoom": 7},
    "Estonia": {"center": [58.7, 25.5], "zoom": 7},
    "Ireland": {"center": [53.3, -8.0], "zoom": 7},
}


def get_country_view(country: str) -> dict:
    """Center/zoom preset for the COUNTRY VIEW dropdown (item 3/20-H)."""
    return _COUNTRY_VIEWS.get(country, _COUNTRY_VIEWS["all"])


def build_raster_overlay(layer_name: str) -> dl.ImageOverlay | None:
    """The one currently-selected continuous raster, pre-rendered and
    NaN-masked outside the study countries — never zero (items 5, 6)."""
    meta = load_overlay_meta(layer_name)
    if meta is None:
        return None
    return dl.ImageOverlay(
        id="suitability-raster-overlay",
        url=overlay_image_url(layer_name),
        bounds=meta["bounds"],
        opacity=0.92,
    )


def build_raster_legend(layer_name: str) -> html.Div:
    """Simple HTML gradient legend — dash-leaflet has no built-in colourbar
    widget. Suitability is always titled/scaled exactly per item 7; other
    layers show their own real data range with the same cividis/viridis
    treatment (see scripts/generate_raster_overlays.py)."""
    meta = load_overlay_meta(layer_name)
    label = next((o["label"] for o in RASTER_LAYER_OPTIONS if o["value"] == layer_name), layer_name)
    if meta is None:
        return html.Div(f"{label}: overlay not available", style={"fontSize": "0.8rem", "color": "#888"})

    cmap = meta.get("cmap", "cividis")
    gradient = (
        "linear-gradient(to right, #00204d, #31446b, #666970, #958f78, #cab969, #ffea46)"
        if cmap == "cividis"
        else "linear-gradient(to right, #440154, #3b528b, #21908c, #5dc963, #fde725)"
    )
    vmin, vmax = meta["vmin"], meta["vmax"]
    title = "Predicted environmental suitability" if layer_name == "suitability" else label
    return html.Div(
        style={"fontSize": "0.8rem", "color": "#2c3e50"},
        children=[
            html.Div(title, style={"fontWeight": 600, "marginBottom": "4px"}),
            html.Div(style={
                "height": "10px", "borderRadius": "2px", "background": gradient,
                "border": "1px solid #999",
            }),
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "marginTop": "2px"},
                children=[
                    html.Span("0 = lower" if layer_name == "suitability" else f"{vmin:.3g}"),
                    html.Span("1 = higher" if layer_name == "suitability" else f"{vmax:.3g}"),
                ],
            ),
        ],
    )


def build_occurrence_layer(occurrence_points: pd.DataFrame, show_background: bool) -> list:
    """Observed presence/background records (item 8) — 'observations, not
    predictions' (item 2B). Presence: white fill, black border, moderate
    size, high opacity — stays visible over every raster including light
    cividis colours where green-on-green would have disappeared. Background:
    small grey, off by default.
    """
    layers = []
    if show_background:
        background = occurrence_points[occurrence_points["type"] == "background"]
        layers.append(
            dl.LayerGroup(
                id="occurrence-background-layer",
                children=[
                    dl.CircleMarker(
                        center=[float(row["lat"]), float(row["lon"])],
                        radius=3,
                        color="#4a4a4a",
                        weight=1,
                        fillColor="#9e9e9e",
                        fillOpacity=0.5,
                        children=[dl.Tooltip(f"{row['system']} — background")],
                    )
                    for _, row in background.iterrows()
                ],
            )
        )
    presence = occurrence_points[occurrence_points["type"] == "presence"]
    layers.append(
        dl.LayerGroup(
            id="occurrence-presence-layer",
            children=[
                dl.CircleMarker(
                    center=[float(row["lat"]), float(row["lon"])],
                    radius=5,
                    color="#000000",
                    weight=1.4,
                    fillColor="#ffffff",
                    fillOpacity=0.95,
                    children=[
                        dl.Tooltip(
                            f"System / country: {row['system']}\n"
                            f"Latitude: {row['lat']:.3f}\n"
                            f"Longitude: {row['lon']:.3f}\n"
                            f"Record type: observed presence"
                        )
                    ],
                )
                for _, row in presence.iterrows()
            ],
        )
    )
    return layers


def build_country_boundary_layer() -> dl.GeoJSON:
    """Study-country outlines only (no fill) — a separate, optional
    BOUNDARIES checkbox layer (item 3), distinct from the raster/occurrence
    layers so it can't be mistaken for either."""
    boundaries = load_study_country_boundaries()
    return dl.GeoJSON(
        id="suitability-country-boundary-layer",
        data=boundaries,
        style={"color": "#2c3e50", "weight": 2, "fillOpacity": 0, "dashArray": "4"},
    )


def build_suitability_map(
    occurrence_points: pd.DataFrame,
    raster_layer: str = DEFAULT_RASTER_LAYER,
    show_occurrence: bool = True,
    show_background: bool = False,
    show_boundaries: bool = True,
    country_view: str = "all",
) -> dl.Map:
    """Assembles the full Leaflet map for the initial render; the
    radio/checkbox/dropdown controls then update it via callbacks that only
    swap layer visibility/selection and the map's center/zoom — never
    re-read or re-mask raster data (item 18)."""
    view = get_country_view(country_view)
    children = [
        dl.TileLayer(
            # Standard OSM tiles — no API key required (CARTO's basemaps.cartocdn.com
            # now requires one for its light_all style; this doesn't).
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        ),
    ]
    overlay = build_raster_overlay(raster_layer)
    if overlay is not None:
        children.append(overlay)
    if show_boundaries:
        children.append(build_country_boundary_layer())
    if show_occurrence:
        children.extend(build_occurrence_layer(occurrence_points, show_background))

    return dl.Map(
        id="suitability-leaflet-map",
        center=view["center"],
        zoom=view["zoom"],
        style={"height": "640px", "width": "100%", "borderRadius": "4px"},
        children=children,
    )
