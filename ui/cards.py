"""Shared KPI card component — used by Overview and Ecological Suitability."""

from __future__ import annotations

from dash import html

from config import READINESS_COLORS
from ui.styles import KPI_CARD, THEME_GREEN

# Sourced directly from config.READINESS_COLORS — the exact palette
# ui.figures.readiness_pie already uses for Overview's pie chart — so a card's
# tone and the pie chart's wedge colour are the *same* colour, not a separate
# palette invented for cards. White text for contrast against these saturated
# fills (unlike the pastel table-cell backgrounds, which use dark text).
_CARD_TONES = {
    "neutral": {"background": "#f0f4f2", "color": THEME_GREEN},
    "high": {"background": READINESS_COLORS["High"], "color": "#ffffff"},
    "medium": {"background": READINESS_COLORS["Medium"], "color": "#ffffff"},
    "low": {"background": READINESS_COLORS["Low"], "color": "#ffffff"},
}


def kpi_card(label: str, value: str, tone: str = "neutral") -> html.Div:
    """Single KPI card: a label and a big value, full-background-coloured by tone
    ('neutral' | 'high' | 'medium' | 'low') — apply a tone whenever a card
    represents a High/Medium/Low-style verdict; leave plain metrics neutral."""
    colors = _CARD_TONES.get(tone, _CARD_TONES["neutral"])
    label_color = colors["color"] if tone == "neutral" else "#ffffff"
    label_opacity = 0.85 if tone == "neutral" else 0.9
    return html.Div(
        className="kpi-card",
        style={**KPI_CARD, "background": colors["background"], "borderTop": "none"},
        children=[
            html.Div(
                label,
                style={
                    "fontSize": "0.85rem", 
                    "color": label_color, 
                    "opacity": label_opacity, 
                    "fontWeight": "600", 
                    "marginBottom": "6px",
                    "letterSpacing": "0.02em",
                    "fontFamily": "system-ui, -apple-system, sans-serif",
                },
            ),
            html.Div(
                value, 
                style={
                    "fontSize": "1.6rem", 
                    "fontWeight": "700", 
                    "color": colors["color"],
                    "lineHeight": "1.2",
                    "fontFamily": "system-ui, -apple-system, sans-serif",
                }
            ),
        ],
    )
