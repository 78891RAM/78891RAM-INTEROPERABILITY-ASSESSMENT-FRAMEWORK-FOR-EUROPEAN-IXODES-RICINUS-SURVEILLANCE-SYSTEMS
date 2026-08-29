"""
Enhanced Ecological Suitability tab — integrates notebook ML outputs.

Renders suitability predictions, occurrence points, model comparison tables,
transfer matrices, exploratory held-out transfer results (Estonia/Ireland —
not "external validation"; the held-out samples are very small), and
feature importance from the enhanced notebook pipeline.
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from ui.cards import kpi_card
from ui.downloads import download_button, register_download
from ui.maps.map_layers import (
    MODEL_DEVELOPMENT_COUNTRIES,
    EXPLORATORY_TRANSFER_COUNTRIES,
    SUITABILITY_INTERPRETATION_NOTE,
)
from ui.maps.suitability_map import (
    build_suitability_map,
    build_raster_legend,
    RASTER_LAYER_OPTIONS,
    COUNTRY_VIEW_OPTIONS,
    DEFAULT_RASTER_LAYER,
    get_country_view,
)
from ui.styles import (
    BLOCK, CHART_MARGIN, KPI_ROW, MUTED, THEME_GREEN, CONTENT_CARD,
    SECTION_CARD, HEADING_2, HEADING_3, BODY_TEXT, TEXT_MUTED, TEXT_DARK, BORDER_LIGHT
)
from config import READINESS_COLORS
from ui.tables import make_table

# Path to notebook outputs. PROJECT_ROOT is tick/ itself — this used to
# climb 4 parents up to the Dessertation folder (i.e. one level *above*
# tick/, outside this git repo entirely), which only worked because the
# app happened to be run from inside that larger folder structure. Fixed
# so tick/ is fully self-contained: deployable (Render, GitHub) without
# anything living outside this repo. See tick/outputs/ — only the ~2.6MB
# subset of the full outputs/ folder that this tab actually reads.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Color schemes matching the existing dashboard
SUITABILITY_COLORS = "RdYlGn"  # Red-Yellow-Green for suitability
PRESENCE_COLORS = {"presence": "#27ae60", "background": "#9e9e9e"}  # More distinct background color

# Caching configuration
CACHE_DIR = PROJECT_ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_EXPIRY_HOURS = 24  # Cache expires after 24 hours

def _get_cache_path(filename: str) -> Path:
    """Get cache file path."""
    return CACHE_DIR / f"{filename}.cache.pkl"

def _is_cache_valid(cache_path: Path, source_path: Path) -> bool:
    """Check if cache is valid (exists, newer than source, not expired)."""
    if not cache_path.exists() or not source_path.exists():
        return False
    
    # Check if cache is newer than source file
    cache_mtime = cache_path.stat().st_mtime
    source_mtime = source_path.stat().st_mtime
    
    if cache_mtime < source_mtime:
        return False
    
    # Check if cache hasn't expired
    cache_age_hours = (time.time() - cache_mtime) / 3600
    return cache_age_hours < CACHE_EXPIRY_HOURS

def _load_from_cache(cache_path: Path):
    """Load data from cache file."""
    try:
        import pickle
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Warning: Could not load cache {cache_path}: {e}")
        return None

def _save_to_cache(data, cache_path: Path):
    """Save data to cache file."""
    try:
        import pickle
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Warning: Could not save cache {cache_path}: {e}")

def _load_csv_safe(path: Path) -> pd.DataFrame | None:
    """Load CSV file safely with caching, return None if not found."""
    try:
        if not path.exists():
            return None
            
        # Check cache first
        cache_path = _get_cache_path(f"csv_{path.name}")
        if _is_cache_valid(cache_path, path):
            cached_data = _load_from_cache(cache_path)
            if cached_data is not None:
                print(f"✓ Loaded {path.name} from cache")
                return cached_data
        
        # Load from source and cache
        print(f"Loading {path.name}...")
        data = pd.read_csv(path)
        _save_to_cache(data, cache_path)
        return data
        
    except Exception as e:
        print(f"Warning: Could not load {path}: {e}")
    return None

def _load_geojson_safe(path: Path) -> dict | None:
    """Load GeoJSON file safely with caching, return None if not found."""
    try:
        if not path.exists():
            return None
            
        # Check cache first
        cache_path = _get_cache_path(f"geojson_{path.name}")
        if _is_cache_valid(cache_path, path):
            cached_data = _load_from_cache(cache_path)
            if cached_data is not None:
                print(f"✓ Loaded {path.name} from cache")
                return cached_data
        
        # Load from source and cache
        print(f"Loading {path.name}...")
        with open(path, 'r') as f:
            data = json.load(f)
        _save_to_cache(data, cache_path)
        return data
        
    except Exception as e:
        print(f"Warning: Could not load {path}: {e}")
    return None

class NotebookData:
    """Container for all notebook ML outputs with performance optimizations."""
    
    def __init__(self):
        print("Loading notebook data...")
        
        # Load all available data
        self.suitability_grid = _load_csv_safe(OUTPUTS_DIR / "suitability_grid.csv")
        self.occurrence_geojson = _load_geojson_safe(OUTPUTS_DIR / "occurrence_layer.geojson")
        self.model_results = _load_csv_safe(OUTPUTS_DIR / "model_results.csv")
        self.transfer_matrix = _load_csv_safe(OUTPUTS_DIR / "transfer_matrix.csv")
        self.external_validation = _load_csv_safe(OUTPUTS_DIR / "external_validation.csv")
        self.feature_importance = _load_csv_safe(OUTPUTS_DIR / "feature_importance.csv")
        
        # Performance optimization: Convert to appropriate dtypes
        if self.suitability_grid is not None:
            self.suitability_grid = self.suitability_grid.astype({
                'lat': 'float32',
                'lon': 'float32', 
                'probability': 'float32'
            })
        
        # Extract occurrence points from GeoJSON
        self.occurrence_points = None
        if self.occurrence_geojson and 'features' in self.occurrence_geojson:
            occurrence_data = []
            for feature in self.occurrence_geojson['features']:
                coords = feature['geometry']['coordinates']
                props = feature['properties']
                occurrence_data.append({
                    'lon': float(coords[0]),
                    'lat': float(coords[1]),
                    'type': props.get('type', 'unknown'),
                    'system': props.get('system', 'unknown'),
                    'presence': int(props.get('presence', 0))
                })
            self.occurrence_points = pd.DataFrame(occurrence_data)
            
        print(f"Data loaded: suitability={len(self.suitability_grid) if self.suitability_grid is not None else 0:,} points, occurrence={len(self.occurrence_points) if self.occurrence_points is not None else 0:,} points")
    
    @property
    def available(self) -> bool:
        """Check if core data is available."""
        return self.suitability_grid is not None and self.occurrence_points is not None

# Global data cache
_NOTEBOOK_DATA: NotebookData | None = None
_FIGURE_CACHE: dict[str, go.Figure] = {}

def _get_notebook_data() -> NotebookData:
    """Get cached notebook data."""
    global _NOTEBOOK_DATA
    if _NOTEBOOK_DATA is None:
        print("Initializing notebook data cache...")
        _NOTEBOOK_DATA = NotebookData()
        print("✓ Notebook data cache ready")
    return _NOTEBOOK_DATA

def _get_cache_key(active_layers: list[str]) -> str:
    """Generate cache key for map figure based on active layers."""
    return "_".join(sorted(active_layers))

def _get_cached_figure(cache_key: str) -> go.Figure | None:
    """Get cached figure if available."""
    return _FIGURE_CACHE.get(cache_key)

def _cache_figure(cache_key: str, figure: go.Figure):
    """Cache figure for reuse."""
    _FIGURE_CACHE[cache_key] = figure
    print(f"✓ Cached map figure for layers: {cache_key}")

def _clear_figure_cache():
    """Clear figure cache (useful for development)."""
    global _FIGURE_CACHE
    _FIGURE_CACHE = {}
    print("✓ Figure cache cleared")

def _get_cache_status() -> str:
    """Get cache status for display."""
    try:
        if not CACHE_DIR.exists():
            return "No cache"
        
        cache_files = list(CACHE_DIR.glob("*.cache.pkl"))
        if not cache_files:
            return "Empty"
        
        # Count cached items
        n_cached = len(cache_files)
        total_size_mb = sum(f.stat().st_size for f in cache_files) / (1024 * 1024)
        
        return f"{n_cached} items ({total_size_mb:.1f}MB)"
        
    except Exception:
        return "Unknown"

def clear_all_caches():
    """Clear all caches - useful for development/debugging."""
    global _NOTEBOOK_DATA, _FIGURE_CACHE, _IMAGE_CACHE
    
    # Clear memory caches
    _NOTEBOOK_DATA = None
    _FIGURE_CACHE = {}
    _IMAGE_CACHE = {}
    
    # Clear disk cache
    try:
        import shutil
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            CACHE_DIR.mkdir()
        print("✓ All caches cleared")
    except Exception as e:
        print(f"Warning: Could not clear disk cache: {e}")

# Layer options for the enhanced map
ENHANCED_LAYER_OPTIONS = [
    {"label": "Ecological Suitability", "value": "suitability"},
    {"label": "Occurrence Records", "value": "occurrence"},
    {"label": "Temperature Seasonality", "value": "bio04"},
    {"label": "Annual Precipitation", "value": "bio12"}, 
    {"label": "NDVI", "value": "ndvi"},
]
DEFAULT_ENHANCED_LAYERS = ["suitability", "occurrence"]

def layout(snapshot=None) -> html.Div:
    """Render the enhanced ecological suitability section."""
    data = _get_notebook_data()
    
    if not data.available:
        return _missing_data_block()
    
    # KPI cards showing key metrics
    n_suitability = len(data.suitability_grid) if data.suitability_grid is not None else 0
    n_occurrence = len(data.occurrence_points) if data.occurrence_points is not None else 0
    n_presence = len(data.occurrence_points[data.occurrence_points['type'] == 'presence']) if data.occurrence_points is not None else 0
    
    # Cache status for performance info  
    cache_status = _get_cache_status()
    
    # Get suitability level distribution from the ecological model data
    suitability_levels = None
    if data.suitability_grid is not None:
        try:
            # Classify suitability into High/Medium/Low based on probability thresholds
            suitability_df = data.suitability_grid.copy()
            suitability_df['suitability_class'] = pd.cut(
                suitability_df['probability'], 
                bins=[0, 0.33, 0.67, 1.0], 
                labels=['Low', 'Medium', 'High'],
                include_lowest=True
            )
            suitability_levels = suitability_df['suitability_class'].value_counts().to_dict()
        except:
            suitability_levels = None
    
    # Performance caption for large datasets  
    perf_caption = None
    if n_suitability > 50000:
        perf_caption = html.P([
            "Performance Note: ",
            f"Large dataset ({n_suitability:,} suitability points). Map rendering optimized to ~10k points for smooth interaction."
        ], style={
            "fontSize": "0.75rem",
            "color": "#6c757d",
            "margin": "8px 0 0 0",
            "fontStyle": "italic",
            "textAlign": "center"
        })
    
    kpi_row = html.Div(
        style=KPI_ROW,
        children=[
            kpi_card("Suitability Predictions", f"{n_suitability:,}"),
            kpi_card("Occurrence Points", f"{n_occurrence:,}"),
            kpi_card("Presence Locations", f"{n_presence:,}"),
        ],
    )

    # Study roles panel (items 4, 9) — methodological role, not just flags.
    # Estonia/Ireland are always "Exploratory held-out transfer", never
    # "validation" — the held-out samples are very small (7 and 19 unique
    # locations; see outputs/external_validation.csv) — see
    # ui/maps/map_layers.py's STUDY_COUNTRY_ROLE docstring for why.
    study_roles_panel = html.Div(
        style={
            "border": f"1px solid {BORDER_LIGHT}", "borderRadius": "4px",
            "padding": "10px 12px", "marginBottom": "12px", "background": "#fafbfa",
        },
        children=[
            html.Div("Study systems", style={"fontWeight": 600, "fontSize": "0.8rem", "marginBottom": "6px", "color": THEME_GREEN}),
            html.Div(
                style={"display": "flex", "gap": "24px", "flexWrap": "wrap"},
                children=[
                    html.Div([
                        html.Div("Model development", style={"fontSize": "0.72rem", "fontWeight": 600, "color": "#555"}),
                        html.Div(", ".join(MODEL_DEVELOPMENT_COUNTRIES), style={"fontSize": "0.78rem"}),
                    ]),
                    html.Div([
                        html.Div("Exploratory held-out transfer", style={"fontSize": "0.72rem", "fontWeight": 600, "color": "#555"}),
                        html.Div(", ".join(EXPLORATORY_TRANSFER_COUNTRIES), style={"fontSize": "0.78rem"}),
                    ]),
                ],
            ),
        ],
    )

    # Structured control panel (item 3) — replaces the flat layer checkbox.
    # One radio selects which single continuous raster is shown (merges the
    # spec's "MODEL OUTPUT" and "ENVIRONMENTAL CONTEXT" groups, which both
    # listed "Predicted environmental suitability" as an option — one
    # control choosing among all 5 layers covers both without asking the
    # same question twice).
    control_panel = html.Div(
        style={
            "width": "220px", "padding": "10px 12px", "height": "fit-content",
            "border": f"1px solid {BORDER_LIGHT}", "borderRadius": "4px", "background": "#fff",
        },
        children=[
            html.Div("Map layers", style={"fontWeight": 600, "fontSize": "0.85rem", "marginBottom": "10px", "color": THEME_GREEN}),

            html.Label("Environmental context", style={"fontWeight": 600, "fontSize": "0.72rem", "display": "block", "marginBottom": "4px"}),
            dcc.RadioItems(
                id="suitability-raster-radio",
                options=RASTER_LAYER_OPTIONS,
                value=DEFAULT_RASTER_LAYER,
                labelStyle={"display": "block", "fontSize": "0.75rem", "margin": "2px 0"},
            ),
            html.P(
                "Only one continuous layer is shown at a time.",
                style={"fontSize": "0.68rem", "color": TEXT_MUTED, "margin": "4px 0 12px", "fontStyle": "italic"},
            ),

            html.Label("Observed data", style={"fontWeight": 600, "fontSize": "0.72rem", "display": "block", "marginBottom": "4px"}),
            dcc.Checklist(
                id="suitability-observed-checklist",
                options=[
                    {"label": "Ixodes ricinus occurrence records", "value": "occurrence"},
                    {"label": "Background points", "value": "background"},
                ],
                value=["occurrence"],
                labelStyle={"display": "block", "fontSize": "0.75rem", "margin": "2px 0"},
            ),
            html.Div(style={"height": "10px"}),

            html.Label("Boundaries", style={"fontWeight": 600, "fontSize": "0.72rem", "display": "block", "marginBottom": "4px"}),
            dcc.Checklist(
                id="suitability-boundaries-checklist",
                options=[{"label": "Study-country boundaries", "value": "boundaries"}],
                value=["boundaries"],
                labelStyle={"display": "block", "fontSize": "0.75rem", "margin": "2px 0"},
            ),
            html.Div(style={"height": "10px"}),

            html.Label("Country view", style={"fontWeight": 600, "fontSize": "0.72rem", "display": "block", "marginBottom": "4px"}),
            dcc.Dropdown(
                id="suitability-country-dropdown",
                options=COUNTRY_VIEW_OPTIONS,
                value="all",
                clearable=False,
                style={"fontSize": "0.78rem"},
            ),
        ],
    )

    # Main map section with legend on left, map centered, controls on right
    map_section = html.Div(
        style=BLOCK,
        children=[
            html.Div([
                html.H4("Ecological Suitability Map", style={
                    "color": THEME_GREEN, "marginBottom": "4px", "fontSize": "1.1rem", "fontWeight": "600",
                }),
            ]),
            study_roles_panel,
            html.Div(
                style={"display": "flex", "gap": "12px", "alignItems": "flex-start", "flexWrap": "wrap"},
                children=[
                    html.Div(
                        style={"flex": "1", "minWidth": "600px"},
                        children=[
                            download_button("dl-suitability-map"),
                            html.Div(
                                id="suitability-leaflet-container",
                                children=build_suitability_map(data.occurrence_points),
                            ),
                        ],
                    ),
                    html.Div(
                        style={"width": "220px", "display": "flex", "flexDirection": "column", "gap": "12px"},
                        children=[
                            control_panel,
                            html.Div(
                                id="suitability-raster-legend",
                                style={"border": f"1px solid {BORDER_LIGHT}", "borderRadius": "4px", "padding": "10px 12px"},
                                children=build_raster_legend(DEFAULT_RASTER_LAYER),
                            ),
                        ],
                    ),
                ],
            ),
            # Interpretation note (item 11) — concise, directly under the map.
            html.P(
                SUITABILITY_INTERPRETATION_NOTE,
                style={**MUTED, "marginTop": "10px", "fontStyle": "italic", "maxWidth": "820px"},
            ),
        ],
    )
    # Model comparison table
    model_table_section = html.Div(
        style=BLOCK,
        children=[
            html.H4("Model Performance Comparison", style={"color": THEME_GREEN, "marginBottom": "12px"}),
            _create_model_comparison_table(data),
        ],
    )

    # Transfer matrix section
    transfer_section = html.Div(
        style=BLOCK, 
        children=[
            html.H4("Cross-System Transfer Matrix", style={"color": THEME_GREEN, "marginBottom": "12px"}),
            html.P("Diagonal shows self-prediction (≈1.0 indicates overfitting). Off-diagonal values indicate cross-system transferability.", 
                   style={**MUTED, "marginBottom": "12px"}),
            _create_transfer_matrix_display(data),
        ],
    )

    # External validation section
    external_section = html.Div(
        style=BLOCK,
        children=[
            html.H4("Exploratory Held-Out Transfer Results", style={"color": THEME_GREEN, "marginBottom": "12px"}),
            html.P(
                "Estonia (7 locations) and Ireland (19 locations) — an exploratory test of the "
                "model's transfer to held-out systems, not an independent validation; results "
                "are indicative only given the very small sample sizes.",
                style={**MUTED, "marginBottom": "12px"}),
            _create_external_validation_table(data),
        ],
    )

    # Feature importance and figures section
    figures_section = html.Div(
        style=BLOCK,
        children=[
            html.H4("Driver Importance and Partial Dependence", style={"color": THEME_GREEN, "marginBottom": "12px"}),
            _create_figures_display(),
        ],
    )

    return html.Div([
        html.P(
            [
                "Ecological suitability model — integrates harmonized occurrence data from multiple surveillance systems ",
                "with environmental predictors using spatially-leakage-safe machine learning. Model predictions and validation ",
                "results are derived from the analysis pipeline."
            ],
            style={**MUTED, "marginBottom": "16px"},
        ),
        kpi_row,
        map_section,
        model_table_section,
        transfer_section,
        external_section,
        figures_section,
    ])

def register_callbacks(app) -> None:
    """Wire the Leaflet control panel to the map + legend.

    Deliberately coarse-grained (item 18): rebuilds the map's children /
    legend content via the same build_suitability_map()/build_raster_legend()
    helpers the initial layout used, rather than trying to patch individual
    Leaflet layers in place — it never re-reads or re-masks raster data (the
    overlay PNGs are already pre-generated on disk; this only picks which
    one's URL to point at and which pre-built marker layers to include), so
    the coarseness costs nothing at request time.
    """

    @app.callback(
        Output("suitability-leaflet-container", "children"),
        Output("suitability-raster-legend", "children"),
        Input("suitability-raster-radio", "value"),
        Input("suitability-observed-checklist", "value"),
        Input("suitability-boundaries-checklist", "value"),
        Input("suitability-country-dropdown", "value"),
    )
    def _update_suitability_map(raster_layer, observed_values, boundaries_values, country_view):
        data = _get_notebook_data()
        if not data.available:
            return html.Div("Notebook outputs not available."), html.Div()

        observed_values = observed_values or []
        leaflet_map = build_suitability_map(
            data.occurrence_points,
            raster_layer=raster_layer or DEFAULT_RASTER_LAYER,
            show_occurrence="occurrence" in observed_values,
            show_background="background" in observed_values,
            show_boundaries="boundaries" in (boundaries_values or []),
            country_view=country_view or "all",
        )
        legend = build_raster_legend(raster_layer or DEFAULT_RASTER_LAYER)
        return leaflet_map, legend

    # Download button always exports the default layers (suitability +
    # occurrence) at full grid resolution — matching
    # export_dissertation_figures.py's fig_suitability_map_full_resolution —
    # regardless of which layers are currently toggled on screen, so the
    # downloaded file is reproducible independent of interactive state.
    register_download(
        app, "dl-suitability-map", "fig_suitability_map_full_resolution.png",
        lambda: (
            _build_enhanced_map_figure(
                _get_notebook_data(), DEFAULT_ENHANCED_LAYERS,
                max_suitability_points=None, restrict_to_study_countries=True,
            )
            if _get_notebook_data().available else None
        ),
        width=1500, height=960,
    )

# Real country boundaries (Natural Earth 10m admin-0, simplified to 0.01°)
# for the four countries the suitability model was actually trained
# (Austria, Croatia) and externally validated (Estonia, Ireland) on — same
# four countries as country_readiness/country_centers below. Loaded once
# and cached; used by default (see restrict_to_study_countries above) on
# both the live map and the dissertation export, so the figure never
# implies the model was assessed beyond the area it has evidence for.
#
# A first version of this filter used rectangular lat/lon bounding boxes
# per country instead of real borders. Checked directly against Natural
# Earth's actual polygons (not assumed): of 1,470 points the bounding boxes
# let through, only 849 (58%) were actually inside these four countries —
# Austria's elongated shape and Croatia's crescent shape pulled in Bosnia,
# Germany, Slovenia, Hungary, Italy and more. Real polygons fix that.
_STUDY_COUNTRY_BOUNDARIES_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "study_country_boundaries.geojson"
_STUDY_COUNTRY_GEOMETRIES: list | None = None


def _get_study_country_geometries() -> list:
    """Lazily load+cache the four study countries' boundary polygons as
    shapely geometries (shapely only — no geopandas at dashboard runtime;
    that's a one-time preprocessing dependency, not a shipped one)."""
    global _STUDY_COUNTRY_GEOMETRIES
    if _STUDY_COUNTRY_GEOMETRIES is None:
        from shapely.geometry import shape

        with open(_STUDY_COUNTRY_BOUNDARIES_PATH) as f:
            geojson = json.load(f)
        _STUDY_COUNTRY_GEOMETRIES = [shape(feat["geometry"]) for feat in geojson["features"]]
    return _STUDY_COUNTRY_GEOMETRIES


def _mask_within_study_countries(lat, lon):
    """True/False numpy mask — which (lat, lon) points fall inside any of
    the four study countries' real borders (shapely.vectorized.contains,
    GEOS-backed, fast enough for the full 86k-point grid)."""
    from shapely import vectorized

    mask = None
    for geom in _get_study_country_geometries():
        hit = vectorized.contains(geom, lon, lat)
        mask = hit if mask is None else (mask | hit)
    return mask


def _build_enhanced_map_figure(
    data: NotebookData,
    active_layers: list[str],
    max_suitability_points: int | None = 10_000,
    restrict_to_study_countries: bool = True,
) -> go.Figure:
    """Build the enhanced suitability map with multiple layers.

    max_suitability_points caps how many suitability-grid rows are plotted,
    for interactive-rendering performance (default 10,000, unchanged from
    before). Pass None to plot the full grid — used by the dissertation
    figure export script, which renders once, offline, so the interactive
    performance cap doesn't apply.

    restrict_to_study_countries (default True): suitability_grid.csv's raw
    model output covers a much larger box than Europe (see the
    lataxis_range/lonaxis_range note on update_geos below), and even within
    Europe the model has only ever been trained or validated on these four
    countries — everywhere else is extrapolation the model has no evidence
    for. Default on, for both the live map and the dissertation export, so
    the figure never implies more geographic coverage than the model
    actually has; pass False to see the full raw grid (e.g. for debugging
    the underlying data).
    """
    traces = []
    
    # Add country readiness overlay (before other layers)
    country_readiness = {
        "Austria": {"color": READINESS_COLORS["High"], "readiness": "High", "iso": "AUT"},
        "Croatia": {"color": READINESS_COLORS["High"], "readiness": "High", "iso": "HRV"}, 
        "Estonia": {"color": READINESS_COLORS["Medium"], "readiness": "Medium", "iso": "EST"},
        "Ireland": {"color": READINESS_COLORS["Low"], "readiness": "Medium", "iso": "IRL"},  # Using Low color for Medium to differentiate
    }
    
    # Add country highlight traces
    for country, info in country_readiness.items():
        traces.append(
            go.Choropleth(
                locations=[info["iso"]],
                z=[1],
                colorscale=[[0, info["color"]], [1, info["color"]]],
                showscale=False,
                hovertemplate=f"<b>{country}</b><br>Readiness: {info['readiness']}<extra></extra>",
                name=f"{country} ({info['readiness']})",
                showlegend=False,
                marker_line_color="#333333",
                marker_line_width=1.5,
            )
        )
    
    # Add country name labels for clarity
    country_centers = {
        "Austria": {"lat": 47.5, "lon": 14.5},
        "Croatia": {"lat": 45.8, "lon": 16.0},
        "Estonia": {"lat": 59.0, "lon": 26.0}, 
        "Ireland": {"lat": 53.0, "lon": -8.0},
    }
    
    for country, coords in country_centers.items():
        readiness_info = country_readiness[country]
        traces.append(
            go.Scattergeo(
                lat=[coords["lat"]],
                lon=[coords["lon"]],
                mode="text",
                text=[country],
                textfont=dict(size=10, color="#2c3e50", family="Arial Black"),
                showlegend=False,
                hoverinfo="skip",
                name=f"{country} Label",
            )
        )
    
    # 1. Suitability heatmap (if requested) - optimized for performance
    if "suitability" in active_layers and data.suitability_grid is not None:
        suitability_df = data.suitability_grid.copy()

        if restrict_to_study_countries:
            n_before = len(suitability_df)
            mask = _mask_within_study_countries(
                suitability_df["lat"].to_numpy(), suitability_df["lon"].to_numpy()
            )
            suitability_df = suitability_df.loc[mask]
            print(f"Restricted suitability points to real study-country borders: {n_before:,} -> {len(suitability_df):,}")

        # Performance optimization: Aggressive reduction for smooth rendering
        if max_suitability_points and len(suitability_df) > max_suitability_points:
            # Take every nth row to stay under the cap for much faster rendering
            step = len(suitability_df) // max_suitability_points + 1
            suitability_df = suitability_df.iloc[::step]
            print(f"Reduced suitability points from {len(data.suitability_grid):,} to {len(suitability_df):,} for performance")
        
        traces.append(
            go.Scattergeo(
                lat=suitability_df["lat"],
                lon=suitability_df["lon"],
                mode="markers",
                marker=dict(
                    size=2,  # Smaller markers for better performance
                    color=suitability_df["probability"],
                    colorscale=SUITABILITY_COLORS,
                    cmin=0,
                    cmax=1,
                    showscale=True,
                    colorbar=dict(
                        title="Predicted Suitability",
                        x=1.02,  # Position outside map on right
                        len=0.6,
                        y=0.7,
                        thickness=15,
                        titleside="right",
                    ),
                    opacity=0.7,  # Slightly higher opacity since markers are smaller
                    line=dict(width=0),  # Remove marker borders for performance
                ),
                name="Predicted Suitability",
                hovertemplate="<b>Suitability:</b> %{marker.color:.3f}<extra></extra>",  # Faster than apply()
                showlegend=True,
            )
        )
    
    # 2. Environmental variable heatmaps (bio04, bio12, ndvi)
    env_vars = {
        "bio04": {"name": "Temperature Seasonality", "colorscale": "Oranges"},
        "bio12": {"name": "Annual Precipitation", "colorscale": "Blues"}, 
        "ndvi": {"name": "NDVI", "colorscale": "Greens"},
    }
    
    for var_key, var_info in env_vars.items():
        if var_key in active_layers and data.occurrence_points is not None:
            # For environmental variables, we'd need the raster data
            # For now, show a placeholder or skip if data not available
            pass
    
    # 3. Occurrence points (if requested) - with performance limits
    if "occurrence" in active_layers and data.occurrence_points is not None:
        occ_df = data.occurrence_points.copy()

        if restrict_to_study_countries:
            # Checked directly against real borders: ~25-30% of points
            # labelled "Austria"/"Croatia" sit just outside that country's
            # actual polygon (a few km, e.g. across into Germany/Italy/
            # Bosnia) — plausible citizen-science GPS noise near a border,
            # but visually it read as points scattered into neighbouring
            # countries. Same real-polygon filter as the suitability grid,
            # applied here for the same reason.
            n_before = len(occ_df)
            mask = _mask_within_study_countries(occ_df["lat"].to_numpy(), occ_df["lon"].to_numpy())
            occ_df = occ_df.loc[mask]
            print(f"Restricted occurrence points to real study-country borders: {n_before:,} -> {len(occ_df):,}")

        # Limit occurrence points for better performance
        if len(occ_df) > 3000:
            occ_df = occ_df.sample(n=3000, random_state=42)
            print(f"Sampled {len(occ_df):,} occurrence points for performance")
        
        # Plot background points first - optimized
        background = occ_df[occ_df["type"] == "background"]
        if len(background) > 0:
            traces.append(
                go.Scattergeo(
                    lat=background["lat"],
                    lon=background["lon"],
                    mode="markers",
                    marker=dict(
                        size=6,  # Smaller for better performance
                        color=PRESENCE_COLORS["background"],
                        opacity=0.6,
                        line=dict(width=0.5, color="white"),  # Thinner border
                    ),
                    name="Background Points",
                    customdata=background["system"],
                    hovertemplate="<b>System:</b> %{customdata}<br><b>Type:</b> Background<extra></extra>",
                    showlegend=True,
                )
            )
        
        # Plot presence points on top - optimized
        presence = occ_df[occ_df["type"] == "presence"]
        if len(presence) > 0:
            traces.append(
                go.Scattergeo(
                    lat=presence["lat"],
                    lon=presence["lon"],
                    mode="markers",
                    marker=dict(
                        size=10,  # Slightly smaller
                        color=PRESENCE_COLORS["presence"],
                        opacity=0.8,
                        line=dict(width=0.5, color="white"),  # Thinner border
                    ),
                    name="Presence Points",
                    customdata=presence["system"],
                    hovertemplate="<b>System:</b> %{customdata}<br><b>Type:</b> Presence<extra></extra>",
                    showlegend=True,
                )
            )

    # Configure map layout
    fig = go.Figure(data=traces)
    fig.update_geos(
        scope="europe",
        projection_type="natural earth",
        showcountries=True,
        showland=True,
        showocean=True,
        landcolor="#f9f9f9",  # Lighter land for better contrast
        oceancolor="#e8f4f8",
        countrycolor="#666666",  # Darker country borders for clarity
        # Without an explicit range, Plotly auto-zooms geo scope="europe" to
        # fit every plotted point — and suitability_grid.csv's raw model
        # output covers lat 20-82°/lon -32-70° (the full CHELSA raster
        # extent: North Africa to deep Russia/Central Asia), not just
        # Europe. That's why the suitability layer appeared to blanket the
        # whole map instead of looking like a Europe map. Same crop the Map
        # tab already uses (ui/tabs/map.py), for the same reason.
        lataxis_range=[34, 62],
        lonaxis_range=[-12, 32],
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        title="",  # Remove duplicate title since we have section heading
        margin={**CHART_MARGIN, "r": 120},  # Right margin for external colorbar
        height=None,  # Let it use container height
        font=dict(family="system-ui, sans-serif", size=13, color="#2c3e50"),
        showlegend=False,  # Hide map legend since we have it in left panel
        # Performance optimizations
        dragmode="pan",  # Disable zoom box for faster interaction
        hovermode="closest",  # More efficient hover detection
    )
    
    # Update traces for better performance
    fig.update_traces(
        # Disable marker selection for performance
        selected=dict(marker=dict(opacity=1)),
        unselected=dict(marker=dict(opacity=1)),
    )
    
    return fig

def _create_model_comparison_table(data: NotebookData) -> html.Div:
    """Create model comparison table using shared table factory."""
    if data.model_results is None:
        return _placeholder_message("Model results not available")
    
    # Format the model results for display
    display_data = data.model_results.copy()
    
    # Round numeric columns
    numeric_cols = display_data.select_dtypes(include=['float64', 'float32']).columns
    for col in numeric_cols:
        display_data[col] = display_data[col].round(3)
    
    return make_table(
        data=display_data.to_dict('records'),
        extra_style_cell_conditional=[
            {"if": {"column_id": "model"}, "fontWeight": "600"},
        ],
        wrapper_style={
            "overflowX": "visible",  # Make table not scrollable
            "width": "100%",
        }
    )

def _create_transfer_matrix_display(data: NotebookData) -> html.Div:
    """Create transfer matrix heatmap."""
    if data.transfer_matrix is None:
        return _placeholder_message("Transfer matrix not available")
    
    # Create a simple table view of the transfer matrix
    return make_table(
        data=data.transfer_matrix.to_dict('records'),
        extra_style_cell_conditional=[
            {"if": {"column_id": col}, "textAlign": "center"} 
            for col in data.transfer_matrix.columns if col != data.transfer_matrix.columns[0]
        ],
        wrapper_style={
            "overflowX": "visible",  # Make table not scrollable
            "width": "100%",
        }
    )

def _create_external_validation_table(data: NotebookData) -> html.Div:
    """Create external validation results table."""
    if data.external_validation is None:
        return _placeholder_message("External validation results not available")
    
    # Format for display
    display_data = data.external_validation.copy()
    
    # Round numeric columns
    numeric_cols = display_data.select_dtypes(include=['float64', 'float32']).columns
    for col in numeric_cols:
        display_data[col] = display_data[col].round(3)
    
    return make_table(
        data=display_data.to_dict('records'),
        extra_style_cell_conditional=[
            {"if": {"column_id": "test_system"}, "fontWeight": "600"},
        ],
        wrapper_style={
            "overflowX": "visible",  # Make table not scrollable
            "width": "100%",
        }
    )

# Image cache for base64 encoded figures
_IMAGE_CACHE: dict[str, str] = {}

def _get_cached_image(figure_path: Path) -> str | None:
    """Get cached base64 image if file hasn't changed."""
    cache_key = str(figure_path)
    
    if cache_key in _IMAGE_CACHE:
        # Check if file was modified since caching
        cache_path = _get_cache_path(f"img_{figure_path.name}")
        if _is_cache_valid(cache_path, figure_path):
            return _IMAGE_CACHE[cache_key]
    
    return None

def _cache_image(figure_path: Path, encoded_image: str):
    """Cache base64 encoded image."""
    cache_key = str(figure_path)
    _IMAGE_CACHE[cache_key] = encoded_image
    
    # Also save to disk cache
    cache_path = _get_cache_path(f"img_{figure_path.name}")
    _save_to_cache(encoded_image, cache_path)

def _create_figures_display() -> html.Div:
    """Display feature importance and partial dependence figures with caching."""
    figures_html = []
    
    # List of figures to display
    figure_files = [
        ("feature_importance.png", "Feature Importance"),
        ("partial_dependence_curves.png", "Partial Dependence Curves"),
        ("observed_vs_suitability.png", "Observations vs Suitability"),
        ("spatial_validation_blocks.png", "Spatial Validation Blocks"),
    ]
    
    for filename, title in figure_files:
        figure_path = FIGURES_DIR / filename
        if figure_path.exists():
            try:
                # Check cache first
                encoded = _get_cached_image(figure_path)
                
                if encoded is None:
                    # Load and encode image
                    print(f"Encoding image: {filename}")
                    with open(figure_path, 'rb') as f:
                        encoded = base64.b64encode(f.read()).decode()
                    _cache_image(figure_path, encoded)
                else:
                    print(f"✓ Using cached image: {filename}")
                
                figures_html.append(
                    html.Div([
                        html.H5(title, style={"marginBottom": "8px", "color": "#34495e"}),
                        html.Img(
                            src=f"data:image/png;base64,{encoded}",
                            style={"maxWidth": "100%", "height": "auto", "marginBottom": "16px"}
                        ),
                    ])
                )
            except Exception as e:
                figures_html.append(
                    html.P(f"Could not load {title}: {e}", style=MUTED)
                )
        else:
            figures_html.append(
                html.P(f"{title}: Figure not available", style=MUTED)
            )
    
    return html.Div(figures_html)

def _placeholder_message(message: str) -> html.Div:
    """Create a professional placeholder message for missing data with thematic styling."""
    return html.Div(
        style={
            **SECTION_CARD,
            "textAlign": "center",
            "background": "linear-gradient(135deg, #f8fdf9 0%, #f4f9f5 100%)",
            "borderLeft": f"4px solid {THEME_GREEN}",
        },
        children=[
            html.Div(
                style={
                    "width": "56px",
                    "height": "56px",
                    "background": f"rgba(45, 90, 61, 0.1)",
                    "borderRadius": "50%",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "margin": "0 auto 16px",
                },
                children=[
                    html.Img(
                        src="/assets/img/ixodes_icon.svg",
                        style={
                            "width": "28px",
                            "height": "28px",
                            "opacity": "0.6",
                            "filter": "grayscale(0.3)",
                        },
                        alt="Data placeholder"
                    ),
                ],
            ),
            html.P(
                message,
                style={
                    **BODY_TEXT,
                    "color": TEXT_MUTED,
                    "margin": "0",
                    "fontSize": "0.9rem",
                    "fontStyle": "italic",
                },
            ),
        ],
    )

def _missing_data_block() -> html.Div:
    """Display when notebook outputs are not available."""
    return html.Div([
        html.H3("Ecological Suitability", style={"color": THEME_GREEN}),
        html.P("Enhanced notebook outputs not available.", style=MUTED),
        html.P("Required files:", style={**MUTED, "marginTop": "12px"}),
        html.Ul([
            html.Li("suitability_grid.csv — suitability predictions for mapping"),
            html.Li("occurrence_layer.geojson — presence and background points"),
            html.Li("model_results.csv — model performance comparison"),
            html.Li("transfer_matrix.csv — cross-system transfer results"),
            html.Li("external_validation.csv — Estonia and Ireland validation"),
            html.Li("feature_importance.csv — driver importance rankings"),
        ], style=MUTED),
        html.P(
            "These files should be generated by running the enhanced notebook pipeline (cells N13, N16, N17).",
            style={**MUTED, "marginTop": "12px"},
        ),
    ])