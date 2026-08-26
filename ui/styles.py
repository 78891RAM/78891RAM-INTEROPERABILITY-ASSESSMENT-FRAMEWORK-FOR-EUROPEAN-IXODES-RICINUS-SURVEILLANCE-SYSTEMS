"""Shared UI layout constants."""

from __future__ import annotations

# Professional surveillance theme color palette
THEME_GREEN = "#2d5a3d"  # Deep forest green (primary)
THEME_TEAL = "#4a7c59"   # Secondary green-teal
NEUTRAL_BG = "#f8f9fa"   # Light neutral background
NEUTRAL_CARD = "#ffffff" # Card background
TEXT_DARK = "#2c3e50"    # Dark slate text
TEXT_MUTED = "#5a6c7d"   # Muted text
BORDER_LIGHT = "#e8ecef" # Light borders

# Legacy color for backwards compatibility
THEME_BLUE = THEME_GREEN

TAB_PANEL = {
    "maxWidth": "1150px",
    "margin": "0 auto",
    "padding": "20px 16px",
    "@media screen and (max-width: 768px)": {
        "padding": "12px 8px",
    },
}

BLOCK = {"marginBottom": "32px"}

MUTED = {"color": "#4a5568", "fontSize": "0.9rem", "lineHeight": "1.6"}  # Darker for better contrast

# Enhanced typography hierarchy
HEADING_1 = {
    "color": TEXT_DARK,
    "fontSize": "1.75rem",
    "fontWeight": "600",
    "lineHeight": "1.3",
    "marginBottom": "8px",
    "fontFamily": "system-ui, -apple-system, sans-serif",
}

HEADING_2 = {
    "color": TEXT_DARK,
    "fontSize": "1.35rem", 
    "fontWeight": "600",
    "lineHeight": "1.4",
    "marginBottom": "16px",
    "marginTop": "24px",
    "fontFamily": "system-ui, -apple-system, sans-serif",
}

HEADING_3 = {
    "color": TEXT_DARK,
    "fontSize": "1.1rem",
    "fontWeight": "600", 
    "lineHeight": "1.5",
    "marginBottom": "12px",
    "marginTop": "20px",
    "fontFamily": "system-ui, -apple-system, sans-serif",
}

BODY_TEXT = {
    "color": TEXT_DARK,
    "fontSize": "0.95rem",
    "lineHeight": "1.7",
    "fontFamily": "system-ui, -apple-system, sans-serif",
}

KPI_ROW = {
    "display": "flex",
    "flexWrap": "wrap",
    "gap": "14px",
    "marginBottom": "24px",
}

KPI_CARD = {
    "flex": "1 1 160px",  # Allow cards to grow and shrink
    "minWidth": "140px",   # Smaller minimum for mobile
    "minHeight": "96px",
    "padding": "18px 20px",
    "background": NEUTRAL_CARD,
    "borderRadius": "12px",
    "border": f"1px solid {BORDER_LIGHT}",
    "borderTop": f"4px solid {THEME_GREEN}",
    "boxShadow": "0 2px 8px rgba(45, 90, 61, 0.06), 0 1px 3px rgba(45, 90, 61, 0.04)",
    "display": "flex",
    "flexDirection": "column",
    "justifyContent": "center",
    "transition": "box-shadow 0.2s ease-in-out",
}

CHART_ROW = {
    "display": "flex",
    "flexWrap": "wrap",
    "gap": "20px",
    "marginBottom": "32px",
}

CHART_CELL = {
    "flex": "1", 
    "minWidth": "280px",  # Smaller for mobile
    "background": NEUTRAL_CARD,
    "borderRadius": "12px",
    "border": f"1px solid {BORDER_LIGHT}",
    "padding": "20px",
    "boxShadow": "0 2px 8px rgba(45, 90, 61, 0.06), 0 1px 3px rgba(45, 90, 61, 0.04)",
}

CHART_MARGIN = {"t": 50, "b": 90, "l": 140, "r": 30}

FIXED_HEADER = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "right": 0,
    "zIndex": 1000,
    "background": NEUTRAL_CARD,
    "borderBottom": f"1px solid {BORDER_LIGHT}",
    "boxShadow": "0 2px 12px rgba(45, 90, 61, 0.08)",
    "paddingBottom": "8px",
}

HEADER_SPACER_HEIGHT = "200px"  # Increased for larger header

# New card-based content styles
CONTENT_CARD = {
    "background": NEUTRAL_CARD,
    "borderRadius": "12px",
    "border": f"1px solid {BORDER_LIGHT}",
    "padding": "24px",
    "marginBottom": "24px",
    "boxShadow": "0 2px 8px rgba(45, 90, 61, 0.06), 0 1px 3px rgba(45, 90, 61, 0.04)",
}

SECTION_CARD = {
    "background": NEUTRAL_CARD,
    "borderRadius": "12px",
    "border": f"1px solid {BORDER_LIGHT}",
    "padding": "20px",
    "marginBottom": "20px",
    "boxShadow": "0 1px 6px rgba(45, 90, 61, 0.05)",
}
