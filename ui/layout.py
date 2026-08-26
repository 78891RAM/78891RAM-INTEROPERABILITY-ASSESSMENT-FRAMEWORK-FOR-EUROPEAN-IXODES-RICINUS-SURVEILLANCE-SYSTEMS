"""Main Dash application layout."""

from __future__ import annotations

from dash import html, dcc

from data.pipeline import FrameworkSnapshot
from ui.styles import (
    FIXED_HEADER, HEADER_SPACER_HEIGHT, TAB_PANEL, THEME_GREEN, 
    NEUTRAL_BG, BORDER_LIGHT, HEADING_1, BODY_TEXT, TEXT_MUTED
)
from ui.tabs import barriers, evidence, export, integration, map as map_tab
from ui.tabs import methodology, overview, recommendations, scores, suitability

TAB_REGISTRY: list[tuple[str, str]] = [
    ("overview", "Overview"),
    ("map", "Map"),
    ("suitability", "Ecological Suitability"),
    ("scores", "Scores"),
    ("barriers", "Barriers"),
    ("integration", "Integration"),
    ("recommendations", "Recommendations"),
    ("evidence", "Evidence"),
    ("methodology", "Methodology"),
    ("export", "Export"),
]

_TAB_LAYOUTS = {
    "overview": overview.layout,
    "map": map_tab.layout,
    "suitability": suitability.layout,
    "scores": scores.layout,
    "barriers": barriers.layout,
    "integration": integration.layout,
    "recommendations": recommendations.layout,
    "evidence": evidence.layout,
    "methodology": methodology.layout,
    "export": export.layout,
}


def _tab_panel(content) -> html.Div:
    """Centre tab content with a comfortable max width."""
    return html.Div(style=TAB_PANEL, children=content)


def render_tab_content(snapshot: FrameworkSnapshot, tab_id: str) -> html.Div:
    """Render the active tab body (used by tab-switch callback)."""
    layout_fn = _TAB_LAYOUTS.get(tab_id, overview.layout)
    return _tab_panel(layout_fn(snapshot))


def _footer(snapshot: FrameworkSnapshot) -> html.Div:
    """Enhanced footer with system count and image credits."""
    n = len(snapshot.systems) if snapshot.ok else 0
    return html.Div(
        style={
            "maxWidth": "1150px",
            "margin": "40px auto 32px",
            "padding": "20px 16px",
            "borderTop": f"1px solid {BORDER_LIGHT}",
            "background": "transparent",
        },
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "flexWrap": "wrap", "gap": "16px"},
                children=[
                    html.P(
                        f"Dashboard assessment: {n} surveillance systems evaluated",
                        style={
                            **BODY_TEXT,
                            "color": TEXT_MUTED,
                            "fontSize": "0.85rem",
                            "margin": "0",
                        },
                    ),
                    html.P(
                        "Images: Custom surveillance icons. Data visualization: Plotly/Dash.",
                        style={
                            **BODY_TEXT,
                            "color": TEXT_MUTED,
                            "fontSize": "0.8rem",
                            "margin": "0",
                            "fontStyle": "italic",
                        },
                    ),
                ],
            ),
        ],
    )


def create_layout(snapshot: FrameworkSnapshot) -> html.Div:
    """Build the full Dash layout with fixed header and scrollable tab content."""
    return html.Div(
        style={
            "fontFamily": "system-ui, -apple-system, sans-serif",
            "background": NEUTRAL_BG,
            "minHeight": "100vh",
        },
        children=[
            html.Div(
                id="app-chrome",
                style=FIXED_HEADER,
                children=[
                    # Professional header banner with background image
                    html.Div(
                        style={
                            "background": f"linear-gradient(135deg, {THEME_GREEN} 0%, #1e4a32 100%)",
                            "backgroundImage": "url(/assets/img/surveillance_header.svg), linear-gradient(135deg, #2d5a3d 0%, #1e4a32 100%)",
                            "backgroundSize": "cover, cover",
                            "backgroundPosition": "center, center",
                            "padding": "20px 0",
                            "marginBottom": "8px",
                            "position": "relative",
                        },
                        children=[
                            html.Div(
                                style={**TAB_PANEL, "padding": "0 20px"},
                                children=[
                                    html.Div(
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                            "justifyContent": "space-between",
                                        },
                                        children=[
                                            html.Div(
                                                children=[
                                                    html.H1(
                                                        "European Ixodes ricinus Surveillance: Data Interoperability Assessment",
                                                        style={
                                                            **HEADING_1,
                                                            "color": "#ffffff",
                                                            "fontSize": "1.65rem",
                                                            "marginBottom": "4px",
                                                        },
                                                    ),
                                                    html.P(
                                                        "MSc Dissertation • Advanced Computational Methods for Vector-Borne Disease Surveillance",
                                                        style={
                                                            **BODY_TEXT,
                                                            "color": "rgba(255, 255, 255, 0.9)",
                                                            "fontSize": "0.9rem",
                                                            "margin": "0",
                                                        },
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                children=[
                                                    # Ixodes ricinus professional icon
                                                    html.Div(
                                                        style={
                                                            "width": "56px",
                                                            "height": "56px",
                                                            "background": "rgba(255, 255, 255, 0.12)",
                                                            "borderRadius": "50%",
                                                            "display": "flex",
                                                            "alignItems": "center",
                                                            "justifyContent": "center",
                                                            "border": "2px solid rgba(255, 255, 255, 0.25)",
                                                            "marginBottom": "6px",
                                                            "backdropFilter": "blur(4px)",
                                                        },
                                                        children=[
                                                            html.Div(
                                                                style={
                                                                    "width": "32px",
                                                                    "height": "32px",
                                                                    "backgroundImage": "url(/assets/img/ixodes_icon.svg)",
                                                                    "backgroundSize": "contain",
                                                                    "backgroundRepeat": "no-repeat",
                                                                    "backgroundPosition": "center",
                                                                    "backgroundColor": "rgba(255,255,255,0.1)",
                                                                    "borderRadius": "50%",
                                                                    "opacity": "0.9",
                                                                },
                                                                title="Ixodes ricinus surveillance"
                                                            ),
                                                        ],
                                                    ),
                                                    html.P(
                                                        "University of East London",
                                                        style={
                                                            "color": "rgba(255, 255, 255, 0.85)",
                                                            "fontSize": "0.8rem",
                                                            "margin": "0",
                                                            "textAlign": "center",
                                                            "fontWeight": "500",
                                                            "letterSpacing": "0.02em",
                                                        },
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        style=TAB_PANEL,
                        children=[
                            dcc.Tabs(
                                id="main-tabs",
                                value="overview",
                                # Horizontal scrolling tabs for long labels
                                style={"overflowX": "auto", "overflowY": "hidden"},
                                children=[
                                    dcc.Tab(
                                        label=label,
                                        value=tab_id,
                                        style={"flex": "0 0 auto", "whiteSpace": "nowrap", "padding": "10px 16px"},
                                        selected_style={
                                            "flex": "0 0 auto",
                                            "whiteSpace": "nowrap",
                                            "padding": "10px 16px",
                                            "fontWeight": "600",
                                        },
                                    )
                                    for tab_id, label in TAB_REGISTRY
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(style={"height": HEADER_SPACER_HEIGHT}),
            html.Div(
                id="tab-content",
                style={"background": NEUTRAL_BG, "minHeight": "60vh"},
                children=render_tab_content(snapshot, "overview"),
            ),
            _footer(snapshot),
        ],
    )


def register_callbacks(app, snapshot: FrameworkSnapshot) -> None:
    """Wire tab switching — header stays fixed, only body content updates."""
    from dash import Input, Output

    @app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
    def _switch_tab(tab_id: str | None):
        return render_tab_content(snapshot, tab_id or "overview")

    # Each tab's "Download figure (PNG)" button (ui/downloads.py) — additive,
    # optional, one figure at a time. For the full dissertation figure set in
    # one command, use export_dissertation_figures.py instead.
    overview.register_callbacks(app, snapshot)
    map_tab.register_callbacks(app, snapshot)
    scores.register_callbacks(app, snapshot)
    barriers.register_callbacks(app, snapshot)
    integration.register_callbacks(app, snapshot)
    suitability.register_callbacks(app)
