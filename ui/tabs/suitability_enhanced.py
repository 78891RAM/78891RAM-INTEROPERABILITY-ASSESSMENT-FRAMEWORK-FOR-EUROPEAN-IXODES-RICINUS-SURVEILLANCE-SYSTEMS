"""
Enhanced Ecological Suitability tab — integrates notebook ML outputs.

Renders suitability predictions, occurrence points, model comparison tables,
transfer matrices, external validation results, and feature importance from
the enhanced notebook pipeline. Uses the existing styling and color maps.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from ui.cards import kpi_card
from ui.styles import BLOCK, CHART_MARGIN, KPI_ROW, MUTED, THEME_BLUE
from ui.tables import make_table

# Path to notebook outputs
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Color schemes matching the existing dashboard
SUITABILITY_COLORS = "RdYlGn"  # Red-Yellow-Green for suitability
PRESENCE_COLORS = {"presence": "#27ae60", "background": "#7f8c8d"}

def _load_csv_safe(path: Path) -> pd.DataFrame | None:
    """Load CSV file safely, return None if not found."""
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception as e:
        print(f"Warning: Could not load {path}: {e}")
    return None

def _load_geojson_safe(path: Path) -> dict | None:
    """Load GeoJSON file safely, return None if not found."""
    try:
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {path}: {e}")
    return None

class NotebookData:
    """Container for all notebook ML outputs."""
    
    def __init__(self):
        # Load all available data
        self.suitability_grid = _load_csv_safe(OUTPUTS_DIR / "suitability_grid.csv")
        self.occurrence_geojson = _load_geojson_safe(OUTPUTS_DIR / "occurrence_layer.geojson")
        self.model_results = _load_csv_safe(OUTPUTS_DIR / "model_results.csv")
        self.transfer_matrix = _load_csv_safe(OUTPUTS_DIR / "transfer_matrix.csv")
        if self.transfer_matrix is not None:
            # transfer_matrix.csv's row-label column has no header in the
            # source CSV, so pandas names it "Unnamed: 0" — rename it before
            # display, same fix already applied in export_dissertation_figures.py.
            self.transfer_matrix = self.transfer_matrix.rename(
                columns={self.transfer_matrix.columns[0]: "System"}
            )
        self.external_validation = _load_csv_safe(OUTPUTS_DIR / "external_validation.csv")
        self.feature_importance = _load_csv_safe(OUTPUTS_DIR / "feature_importance.csv")
        
        # Extract occurrence points from GeoJSON
        self.occurrence_points = None
        if self.occurrence_geojson and 'features' in self.occurrence_geojson:
            occurrence_data = []
            for feature in self.occurrence_geojson['features']:
                coords = feature['geometry']['coordinates']
                props = feature['properties']
                occurrence_data.append({
                    'lon': coords[0],
                    'lat': coords[1],
                    'type': props.get('type', 'unknown'),
                    'system': props.get('system', 'unknown'),
                    'presence': props.get('presence', 0)
                })
            self.occurrence_points = pd.DataFrame(occurrence_data)
    
    @property
    def available(self) -> bool:
        """Check if core data is available."""
        return self.suitability_grid is not None and self.occurrence_points is not None

# Global data cache
_NOTEBOOK_DATA: NotebookData | None = None

def _get_notebook_data() -> NotebookData:
    """Get cached notebook data."""
    global _NOTEBOOK_DATA
    if _NOTEBOOK_DATA is None:
        _NOTEBOOK_DATA = NotebookData()
    return _NOTEBOOK_DATA

# Layer options for the enhanced map
ENHANCED_LAYER_OPTIONS = [
    {"label": "Predicted suitability heatmap", "value": "suitability"},
    {"label": "Occurrence points (presence vs background)", "value": "occurrence"},
    {"label": "Temperature seasonality (bio04)", "value": "bio04"},
    {"label": "Annual precipitation (bio12)", "value": "bio12"}, 
    {"label": "NDVI", "value": "ndvi"},
]
DEFAULT_ENHANCED_LAYERS = ["suitability", "occurrence"]

def layout(_snapshot=None) -> html.Div:
    """Render the enhanced ecological suitability section."""
    data = _get_notebook_data()
    
    if not data.available:
        return _missing_data_block()
    
    # KPI cards showing key metrics
    n_suitability = len(data.suitability_grid) if data.suitability_grid is not None else 0
    n_occurrence = len(data.occurrence_points) if data.occurrence_points is not None else 0
    n_presence = len(data.occurrence_points[data.occurrence_points['type'] == 'presence']) if data.occurrence_points is not None else 0
    
    kpi_row = html.Div(
        style=KPI_ROW,
        children=[
            kpi_card("Suitability Predictions", f"{n_suitability:,}"),
            kpi_card("Occurrence Points", f"{n_occurrence:,}"),
            kpi_card("Presence Locations", f"{n_presence:,}"),
        ],
    )

    # Map controls
    map_controls = html.Div(
        style={"marginBottom": "16px"},
        children=[
            html.Label("Map Layers", style={"fontWeight": "600", "fontSize": "0.85rem", "marginBottom": "8px", "display": "block"}),
            dcc.Checklist(
                id="enhanced-suitability-layers",
                options=ENHANCED_LAYER_OPTIONS,
                value=DEFAULT_ENHANCED_LAYERS,
                labelStyle={"display": "block", "fontSize": "0.88rem", "margin": "4px 0"},
            ),
        ]
    )

    # Main map
    map_section = html.Div(
        style=BLOCK,
        children=[
            html.H4("Interactive Suitability Map", style={"color": THEME_BLUE, "marginBottom": "12px"}),
            map_controls,
            dcc.Graph(
                id="enhanced-suitability-map",
                figure=_build_enhanced_map_figure(data, DEFAULT_ENHANCED_LAYERS),
                config={"scrollZoom": True},
            ),
        ],
    )

    # Model comparison table
    model_table_section = html.Div(
        style=BLOCK,
        children=[
            html.H4("Model Performance Comparison", style={"color": THEME_BLUE, "marginBottom": "12px"}),
            _create_model_comparison_table(data),
        ],
    )

    # Transfer matrix section
    transfer_section = html.Div(
        style=BLOCK, 
        children=[
            html.H4("Cross-System Transfer Matrix", style={"color": THEME_BLUE, "marginBottom": "12px"}),
            html.P("Diagonal shows self-prediction (≈1.0 indicates overfitting). Off-diagonal values indicate cross-system transferability.", 
                   style={**MUTED, "marginBottom": "12px"}),
            _create_transfer_matrix_display(data),
        ],
    )

    # External validation section
    external_section = html.Div(
        style=BLOCK,
        children=[
            html.H4("External Validation Results", style={"color": THEME_BLUE, "marginBottom": "12px"}),
            html.P("Estonia (7 locations) and Ireland (19 locations) — results are indicative only due to very small sample sizes.", 
                   style={**MUTED, "marginBottom": "12px"}),
            _create_external_validation_table(data),
        ],
    )

    # Feature importance and figures section
    figures_section = html.Div(
        style=BLOCK,
        children=[
            html.H4("Driver Importance and Partial Dependence", style={"color": THEME_BLUE, "marginBottom": "12px"}),
            _create_figures_display(),
        ],
    )

    return html.Div([
        html.P(
            [
                "Enhanced ecological suitability model — integrates harmonized occurrence data from multiple surveillance systems ",
                "with environmental predictors using spatially-leakage-safe machine learning. Model predictions and validation ",
                "results are derived from the extended notebook pipeline."
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
    """Wire the enhanced controls to the map figure."""

    @app.callback(
        Output("enhanced-suitability-map", "figure"),
        Input("enhanced-suitability-layers", "value"),
    )
    def _update_enhanced_map(active_layers):
        data = _get_notebook_data()
        if not data.available:
            return go.Figure()
        return _build_enhanced_map_figure(data, active_layers or [])

def _build_enhanced_map_figure(data: NotebookData, active_layers: list[str]) -> go.Figure:
    """Build the enhanced suitability map with multiple layers."""
    traces = []
    
    # 1. Suitability heatmap (if requested)
    if "suitability" in active_layers and data.suitability_grid is not None:
        suitability_df = data.suitability_grid.copy()
        
        traces.append(
            go.Scattergeo(
                lat=suitability_df["lat"],
                lon=suitability_df["lon"],
                mode="markers",
                marker=dict(
                    size=3,
                    color=suitability_df["probability"],
                    colorscale=SUITABILITY_COLORS,
                    cmin=0,
                    cmax=1,
                    showscale=True,
                    colorbar=dict(title="Predicted Suitability", x=1.02, len=0.5, y=0.75),
                    opacity=0.6,
                ),
                name="Predicted Suitability",
                hovertext=suitability_df.apply(
                    lambda r: f"Suitability: {r['probability']:.3f}",
                    axis=1,
                ),
                hoverinfo="text",
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
    
    # 3. Occurrence points (if requested)
    if "occurrence" in active_layers and data.occurrence_points is not None:
        occ_df = data.occurrence_points.copy()
        
        # Plot background points first
        background = occ_df[occ_df["type"] == "background"]
        if len(background) > 0:
            traces.append(
                go.Scattergeo(
                    lat=background["lat"],
                    lon=background["lon"],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=PRESENCE_COLORS["background"],
                        opacity=0.6,
                        line=dict(width=1, color="white"),
                    ),
                    name="Background Points",
                    hovertext=background.apply(
                        lambda r: f"System: {r['system']}<br>Type: Background",
                        axis=1,
                    ),
                    hoverinfo="text",
                    showlegend=True,
                )
            )
        
        # Plot presence points on top
        presence = occ_df[occ_df["type"] == "presence"]
        if len(presence) > 0:
            traces.append(
                go.Scattergeo(
                    lat=presence["lat"],
                    lon=presence["lon"],
                    mode="markers",
                    marker=dict(
                        size=12,
                        color=PRESENCE_COLORS["presence"],
                        opacity=0.8,
                        line=dict(width=1, color="white"),
                    ),
                    name="Presence Points",
                    hovertext=presence.apply(
                        lambda r: f"System: {r['system']}<br>Type: Presence",
                        axis=1,
                    ),
                    hoverinfo="text", 
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
        landcolor="#f4f6f7",
        oceancolor="#e8f4f8",
        countrycolor="#bdc3c7",
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        title="Ecological Suitability — Predictions and Observations",
        margin={**CHART_MARGIN, "r": 120},
        height=600,
        font=dict(family="system-ui, sans-serif", size=13, color="#2c3e50"),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.85)",
        ),
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
        ]
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
        ]
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
        ]
    )

def _create_figures_display() -> html.Div:
    """Display feature importance and partial dependence figures."""
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
                # Encode image as base64
                with open(figure_path, 'rb') as f:
                    encoded = base64.b64encode(f.read()).decode()
                
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
    """Create a placeholder message for missing data."""
    return html.Div(
        html.P(message, style={**MUTED, "textAlign": "center", "padding": "20px"}),
        style={"border": "1px solid #ddd", "borderRadius": "4px", "margin": "10px 0"}
    )

def _missing_data_block() -> html.Div:
    """Display when notebook outputs are not available."""
    return html.Div([
        html.H3("Ecological Suitability", style={"color": THEME_BLUE}),
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