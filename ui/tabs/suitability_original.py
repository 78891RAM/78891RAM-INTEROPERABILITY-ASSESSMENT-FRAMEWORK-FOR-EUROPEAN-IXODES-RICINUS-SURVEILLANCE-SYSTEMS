"""
Ecological Suitability tab — field-validated tick presence model.

Renders model_layer.py's output: predicted suitability at the 34 surveyed
field sites only (never extrapolated to a continental surface), plus
display-only occurrence/environment/density backdrops. The model is loaded
from a saved file, never retrained at app startup — see model_layer.py.
"""

from __future__ import annotations

import base64

import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

import model_layer
from model_layer import ModelLayerData
from ui.cards import kpi_card
from ui.styles import BLOCK, CHART_MARGIN, KPI_ROW, MUTED, THEME_BLUE

_CACHE: ModelLayerData | None = None

LAYER_OPTIONS = [
    {"label": "Tick presence / absence (field sites)", "value": "presence_absence"},
    {"label": "Environmental drivers", "value": "environment"},
    {"label": "ML model output (suitability)", "value": "ml_output"},
    {"label": "Occurrence density (backdrop)", "value": "density_backdrop"},
]
DEFAULT_LAYERS = ["presence_absence", "ml_output"]

# One line per layer, shown next to the checklist so a reader knows what each
# toggle actually draws before turning it on.
LAYER_DESCRIPTIONS = {
    "presence_absence": (
        "Every raw field sample (green = ticks found, red = none), 579 present / "
        "2,133 absent. Sites are tightly clustered so same-site points overlap — "
        "green draws on top at ~60% opacity so overlapping colours blend instead "
        "of one hiding the other."
    ),
    "environment": (
        "Background survey points coloured by the selected weather/vegetation "
        "variable, for context only — not used to train the model."
    ),
    "ml_output": (
        "The model's predicted probability of tick presence at each of the 34 "
        "field sites — not extrapolated beyond those sites."
    ),
    "density_backdrop": (
        "Faint grid of how many occurrence records exist per map cell, across "
        "all sources — a backdrop, not a prediction."
    ),
}

ENV_VARIABLE_OPTIONS = [
    {"label": "Temperature", "value": "temp_C"},
    {"label": "NDVI", "value": "ndvi"},
    {"label": "Rainfall", "value": "rain_mm"},
]
DEFAULT_ENV_VARIABLE = "temp_C"
_ENV_LABELS = {opt["value"]: opt["label"] for opt in ENV_VARIABLE_OPTIONS}

PATHOGEN_NOTE = (
    "Pathogen reports: no data available for these field sites — documented coverage gap. "
    "No points are shown or invented for this layer."
)


def _get_data() -> ModelLayerData:
    global _CACHE
    if _CACHE is None:
        _CACHE = model_layer.load_model_layer_data()
    return _CACHE


def layout(_snapshot=None) -> html.Div:
    """Render the field tick-model section, or a setup message if inputs are missing."""
    data = _get_data()

    if not data.available:
        return _missing_data_block(data)

    kpi_row = html.Div(
        style=KPI_ROW,
        children=[
            kpi_card("Site-grouped AUC", _fmt_auc(data)),
            kpi_card("Field sites", str(data.n_sites)),
            kpi_card("Field points", str(data.n_points)),
        ],
    )

    warning_block = (
        html.P(f"⚠ {data.warning}", style={**MUTED, "color": "#b9770e", "marginBottom": "12px"})
        if data.warning
        else html.Div()
    )

    controls = html.Div(
        style={"display": "flex", "flexWrap": "wrap", "gap": "24px", "marginBottom": "16px"},
        children=[
            html.Div([
                html.Label("Map layers", style={"fontWeight": "600", "fontSize": "0.85rem"}),
                dcc.Checklist(
                    id="field-model-layers",
                    options=LAYER_OPTIONS,
                    value=DEFAULT_LAYERS,
                    labelStyle={"display": "block", "fontSize": "0.88rem", "margin": "4px 0"},
                ),
                html.Ul(
                    [
                        html.Li(f"{opt['label']} — {LAYER_DESCRIPTIONS[opt['value']]}")
                        for opt in LAYER_OPTIONS
                    ],
                    style={**MUTED, "fontSize": "0.78rem", "maxWidth": "480px", "marginTop": "6px", "paddingLeft": "18px"},
                ),
            ]),
            html.Div([
                html.Label("Environmental variable", style={"fontWeight": "600", "fontSize": "0.85rem"}),
                dcc.Dropdown(
                    id="field-model-env-var",
                    options=ENV_VARIABLE_OPTIONS,
                    value=DEFAULT_ENV_VARIABLE,
                    clearable=False,
                    disabled=data.environment is None,
                    style={"width": "220px"},
                ),
            ]),
        ],
    )

    downloads = html.Div([
        html.H4("Download Predictions", style={"color": THEME_BLUE, "marginBottom": "8px"}),
        _csv_download(data.field, "field_model_predictions.csv", "Download field predictions (CSV)"),
    ])

    return html.Div([
        html.P(
            [
                "Field-validated tick presence model — predicted suitability is shown only at the "
                "34 surveyed field sites, not extrapolated to a continental surface. This is a "
                "micro-habitat proof-of-concept validated on unseen sites, independent from the "
                "interoperability scoring elsewhere in this app. ",
                html.Span(f"Model source: {_fmt_source(data.model_source)}.", style={"fontStyle": "italic"}),
            ],
            style={**MUTED, "marginBottom": "16px"},
        ),
        warning_block,
        kpi_row,
        controls,
        html.Div(
            style=BLOCK,
            children=[
                dcc.Graph(
                    id="field-model-map",
                    figure=_build_field_map_figure(data, DEFAULT_LAYERS, DEFAULT_ENV_VARIABLE),
                    config={"scrollZoom": True},
                ),
                html.P(PATHOGEN_NOTE, style={**MUTED, "marginTop": "8px", "fontStyle": "italic"}),
            ],
        ),
        html.Div(style=BLOCK, children=[downloads]),
    ])


def register_callbacks(app) -> None:
    """Wire the layer/variable controls to the map figure."""

    @app.callback(
        Output("field-model-map", "figure"),
        Input("field-model-layers", "value"),
        Input("field-model-env-var", "value"),
    )
    def _update_field_model_map(active_layers, env_variable):
        data = _get_data()
        if not data.available:
            return go.Figure()
        return _build_field_map_figure(data, active_layers or [], env_variable or DEFAULT_ENV_VARIABLE)


def _build_field_map_figure(data: ModelLayerData, active_layers: list[str], env_variable: str) -> go.Figure:
    # Draw order matters: quiet backdrops first, model output next, field
    # presence/absence last so it always renders on top and stays visible.
    traces: list[go.Scattergeo] = []

    if "density_backdrop" in active_layers and data.cells is not None and not data.cells.empty:
        cells = data.cells.dropna(subset=["cell_lat", "cell_lon"])
        if not cells.empty:
            traces.append(
                go.Scattergeo(
                    lat=cells["cell_lat"],
                    lon=cells["cell_lon"],
                    mode="markers",
                    marker=dict(size=4, color="#bdc3c7", opacity=0.35, line=dict(width=0)),
                    name="Occurrence density",
                    hovertext=cells.apply(
                        lambda r: (
                            f"{_fmt_int(r.get('n_records'))} records · "
                            f"{_fmt_int(r.get('n_sources'))} sources"
                        ),
                        axis=1,
                    ),
                    hoverinfo="text",
                    showlegend=True,
                )
            )

    show_env_colorbar = "ml_output" not in active_layers
    if "environment" in active_layers and data.environment is not None and not data.environment.empty:
        env = data.environment.dropna(subset=["lat", "lon"]).copy()
        col = env_variable if env_variable in env.columns else None
        if col:
            # Real exports mix numeric values with a "NO DATA AVAILABLE" string
            # sentinel in the same column — coerce so both true NaN and that
            # sentinel are treated as missing, not plotted or shown as "NaN".
            env[col] = pd.to_numeric(env[col], errors="coerce")
            env = env.dropna(subset=[col])
        if not env.empty:
            traces.append(
                go.Scattergeo(
                    lat=env["lat"],
                    lon=env["lon"],
                    mode="markers",
                    marker=dict(
                        size=5.5,
                        color=env[col] if col else "#7f8c8d",
                        # Single hue, light -> dark for all three variables (never a
                        # rainbow) — the variable name is what changes, not the ramp,
                        # so the colour language stays consistent as you switch it.
                        colorscale="Oranges",
                        # Only one colourbar on screen at a time — when the ML output
                        # layer is also on, its P(present) scale takes priority and
                        # this layer's scale is suppressed rather than overlapping it.
                        showscale=bool(col) and show_env_colorbar,
                        colorbar=dict(title=_ENV_LABELS.get(env_variable, env_variable), x=1.02, len=0.5, y=0.78)
                        if col and show_env_colorbar
                        else None,
                        opacity=0.5,
                    ),
                    name="Environmental drivers",
                    hoverinfo="text",
                    hovertext=env.apply(
                        lambda r: _fmt_hover(_ENV_LABELS.get(env_variable, env_variable), r.get(col) if col else None),
                        axis=1,
                    ),
                    showlegend=True,
                )
            )

    if "ml_output" in active_layers:
        field = data.field.dropna(subset=["lat", "lon", "pred_prob"])
        if not field.empty:
            traces.append(
                go.Scattergeo(
                    lat=field["lat"],
                    lon=field["lon"],
                    mode="markers",
                    marker=dict(
                        size=13,
                        color=field["pred_prob"],
                        # Single hue (blue, this app's brand colour), light -> dark —
                        # matches the "one hue per magnitude" rule instead of a
                        # multi-hue scale like Viridis.
                        colorscale="Blues",
                        cmin=0,
                        cmax=1,
                        showscale=True,
                        colorbar=dict(title="P(present)", x=1.02, len=0.5, y=0.22),
                        line=dict(width=1, color="#ffffff"),
                    ),
                    name="ML model output",
                    hovertext=field.apply(
                        lambda r: f"{r.get('site', 'n/a')}<br>P(present) = {r['pred_prob']:.3f}",
                        axis=1,
                    ),
                    hoverinfo="text",
                    showlegend=True,
                )
            )

    if "presence_absence" in active_layers:
        field = data.field.dropna(subset=["lat", "lon", "present"])
        # 34 tightly-clustered UK sites, ~80 raw rows each sharing the exact same
        # lat/lon, and present is the minority class (579 vs 2,133) — so plotting
        # both classes as raw points means they stack directly on top of each
        # other. Draw absent FIRST, present SECOND so green sits on top; sized up
        # with a white outline so it reads clearly; both at ~60% opacity so a
        # green dot over a red one blends rather than fully hiding it.
        absent = field.loc[field["present"] == 0]
        present = field.loc[field["present"] == 1]
        print(f"[field-model map] absent points plotted: {len(absent)}")
        print(f"[field-model map] present points plotted: {len(present)}")

        if not absent.empty:
            traces.append(
                go.Scattergeo(
                    lat=absent["lat"],
                    lon=absent["lon"],
                    mode="markers",
                    marker=dict(size=9, opacity=0.6, color="#c0392b", line=dict(width=1, color="#ffffff")),
                    name="Absence (field)",
                    hovertext=absent.apply(
                        lambda r: f"{r.get('site', 'n/a')}<br>Ticks: {_fmt_int(r.get('count_ticks'))}",
                        axis=1,
                    ),
                    hoverinfo="text",
                    showlegend=True,
                )
            )
        if not present.empty:
            traces.append(
                go.Scattergeo(
                    lat=present["lat"],
                    lon=present["lon"],
                    mode="markers",
                    marker=dict(size=11, opacity=0.6, color="#27ae60", line=dict(width=1, color="#ffffff")),
                    name="Presence (field)",
                    hovertext=present.apply(
                        lambda r: f"{r.get('site', 'n/a')}<br>Ticks: {_fmt_int(r.get('count_ticks'))}",
                        axis=1,
                    ),
                    hoverinfo="text",
                    showlegend=True,
                )
            )

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
        title="Field Tick Model — Suitability at Surveyed Sites",
        margin={**CHART_MARGIN, "r": 90},
        height=560,
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


def _fmt_int(value) -> str:
    return "n/a" if pd.isna(value) else str(int(value))


def _fmt_hover(label: str, value) -> str:
    return f"{label}: n/a" if pd.isna(value) else f"{label}: {float(value):.2f}"


def _missing_data_block(data: ModelLayerData) -> html.Div:
    return html.Div([
        html.H3("Ecological Suitability", style={"color": THEME_BLUE}),
        html.P(data.error or "Field model data not available.", style=MUTED),
        html.P("Place these files in data/field_model/:", style={**MUTED, "marginTop": "12px"}),
        html.Ul([
            html.Li("field_clean.csv — required (site, lat, lon, present, count_ticks, and field conditions)"),
            html.Li("model.pkl — required to skip retraining (a saved sklearn Pipeline)"),
            html.Li("occurrence_layer.csv — optional, display-only occurrence backdrop"),
            html.Li("environment_layer.csv — optional, display-only environmental overlay"),
            html.Li("dashboard_cells.csv — optional, display-only occurrence density grid"),
        ], style=MUTED),
        html.P(
            "If model.pkl is missing but field_clean.csv is present, a fallback Random Forest "
            "is trained once (GroupKFold by site) and saved to data/field_model/model.pkl.",
            style={**MUTED, "marginTop": "12px"},
        ),
    ])


def _fmt_auc(data: ModelLayerData) -> str:
    if data.auc is None:
        return "—"
    suffix = " (in-sample)" if data.auc_is_insample else ""
    return f"{data.auc:.3f}{suffix}"


def _fmt_source(source: str) -> str:
    return {
        "loaded": "Loaded (saved model)",
        "trained_fallback": "Trained (fallback)",
        "unavailable": "Unavailable",
    }.get(source, source)



def _csv_download(df: pd.DataFrame, filename: str, label: str) -> html.A:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return html.A(
        label,
        href=f"data:text/csv;base64,{b64}",
        download=filename,
        style={"display": "block", "margin": "8px 0", "color": "#2980b9"},
    )
