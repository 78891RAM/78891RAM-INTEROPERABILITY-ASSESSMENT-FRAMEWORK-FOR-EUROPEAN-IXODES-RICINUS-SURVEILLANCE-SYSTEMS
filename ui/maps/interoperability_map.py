"""dash-leaflet interoperability readiness map (item 14).

Represents the 14-system interoperability *assessment*, not ecological
suitability — kept on a separate tab, separate legend, separate colour
scale from the suitability map (item 2). Data comes from
snapshot.map_scatter / core.geo / core.integration exactly as the previous
Plotly version used — no scores or classifications are recalculated here,
only rendered differently.
"""

from __future__ import annotations

import pandas as pd
import dash_leaflet as dl
from dash import html
from dash_extensions.javascript import assign

from core.validation import weighted_readiness_class
from ui.maps.map_layers import INTEROP_COLORS, load_europe_boundaries

# Per-feature styling has to be a client-side JS function in dash-leaflet
# (Python style callbacks aren't sent to the browser per-render) — colour
# and readiness class are pre-computed in Python and baked into each
# feature's properties below, so this JS only ever reads them back.
_geojson_style = assign(
    """
    function(feature, context) {
        return {
            fillColor: feature.properties.fillColor,
            color: "#555555",
            weight: 1,
            fillOpacity: feature.properties.assessed ? 0.75 : 0.35,
        };
    }
    """
)

_geojson_hover_style = assign(
    "function(feature, context) { return {weight: 2.5, color: '#222222', fillOpacity: 0.9}; }"
)


def _country_level_summary(map_scatter: pd.DataFrame) -> pd.DataFrame:
    """One row per ISO3, single-country systems only — same rule the
    previous Plotly map used (ui/tabs/map.py's _split_country_and_marker_rows):
    EU-wide / multi-country systems don't map to one polygon, so they're
    excluded here and shown as markers instead (see build_interoperability_map).
    """
    df = map_scatter.copy()
    df["total_score"] = pd.to_numeric(df["total_score"], errors="coerce").fillna(0)
    iso3_all = df["iso3_all"].fillna("").astype(str)
    single_country = (iso3_all != "") & ~iso3_all.str.contains(",") & (iso3_all != "EUR")
    single = df.loc[single_country]
    if single.empty:
        return pd.DataFrame(columns=["iso3", "total_score", "system_name", "system_count", "readiness_class"])
    grouped = (
        single.groupby("iso3")
        .agg(
            total_score=("total_score", "mean"),
            system_name=("system_name", lambda s: "; ".join(s)),
            system_count=("system_id", "count"),
        )
        .reset_index()
    )
    grouped["readiness_class"] = grouped["total_score"].apply(weighted_readiness_class)
    return grouped


def _multi_country_markers(map_scatter: pd.DataFrame) -> list:
    """EU-wide / multi-country systems as circle markers — can't be pinned
    to one polygon (same systems the Plotly map showed as dots)."""
    df = map_scatter.copy()
    df["total_score"] = pd.to_numeric(df["total_score"], errors="coerce").fillna(0)
    iso3_all = df["iso3_all"].fillna("").astype(str)
    is_multi = (iso3_all == "EUR") | iso3_all.str.contains(",")
    markers = []
    for _, row in df.loc[is_multi].iterrows():
        readiness = row.get("readiness_class") or "Not assessed"
        color = INTEROP_COLORS.get(readiness, INTEROP_COLORS["Not assessed"])
        lat, lon = row.get("lat"), row.get("lon")
        if pd.isna(lat) or pd.isna(lon):
            continue
        # Radius scaled by total_score (0-20), same convention the previous
        # Plotly map used — matches the tab's own caption ("size = total score").
        radius = 7 + (row["total_score"] / 20) * 8
        markers.append(
            dl.CircleMarker(
                center=[float(lat), float(lon)],
                radius=radius,
                color="#333333",
                weight=1.5,
                fillColor=color,
                fillOpacity=0.85,
                children=[
                    dl.Tooltip(f"{row['system_name']} — {readiness} ({row['total_score']:.0f}/20)"),
                ],
            )
        )
    return markers


def build_interoperability_map(map_scatter: pd.DataFrame) -> html.Div:
    """Full interoperability tab map: neutral basemap, ~50-country GeoJSON
    layer (10 assessed + coloured, ~40 explicitly 'Not assessed' — never
    implied as Low), multi-country systems as markers, plus a text legend
    (dl.GeoJSON has no built-in colour-legend widget)."""
    boundaries = load_europe_boundaries()
    country_summary = _country_level_summary(map_scatter).set_index("iso3")

    features = []
    for feat in boundaries["features"]:
        iso3 = feat["properties"].get("iso3")
        assessed = bool(feat["properties"].get("assessed"))
        new_feat = {"type": "Feature", "geometry": feat["geometry"], "properties": dict(feat["properties"])}
        if assessed and iso3 in country_summary.index:
            row = country_summary.loc[iso3]
            new_feat["properties"]["fillColor"] = INTEROP_COLORS[row["readiness_class"]]
            new_feat["properties"]["readiness_class"] = row["readiness_class"]
            new_feat["properties"]["system_name"] = row["system_name"]
            new_feat["properties"]["total_score"] = round(float(row["total_score"]), 1)
            new_feat["properties"]["assessed"] = True
        else:
            new_feat["properties"]["fillColor"] = INTEROP_COLORS["Not assessed"]
            new_feat["properties"]["readiness_class"] = "Not assessed"
            new_feat["properties"]["assessed"] = False
        features.append(new_feat)

    geojson_layer = dl.GeoJSON(
        data={"type": "FeatureCollection", "features": features},
        id="interop-country-layer",
        style=_geojson_style,
        hoverStyle=_geojson_hover_style,
        zoomToBoundsOnClick=False,
    )

    leaflet_map = dl.Map(
        id="interop-leaflet-map",
        center=[50.5, 10.0],
        zoom=4,
        style={"height": "640px", "width": "100%", "borderRadius": "4px"},
        children=[
            dl.TileLayer(
                # Standard OSM tiles — no API key required (CARTO's basemaps.cartocdn.com
                # now requires one for its light_all style; this doesn't).
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            ),
            geojson_layer,
            dl.LayerGroup(_multi_country_markers(map_scatter)),
        ],
    )

    legend = html.Div(
        style={
            "display": "flex", "flexWrap": "wrap", "gap": "16px",
            "fontSize": "0.85rem", "marginTop": "10px", "color": "#2c3e50",
        },
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "6px"},
                children=[
                    html.Span(style={
                        "display": "inline-block", "width": "14px", "height": "14px",
                        "background": color, "border": "1px solid #555", "borderRadius": "2px",
                    }),
                    html.Span(label),
                ],
            )
            for label, color in INTEROP_COLORS.items()
        ],
    )

    return html.Div([leaflet_map, legend])
