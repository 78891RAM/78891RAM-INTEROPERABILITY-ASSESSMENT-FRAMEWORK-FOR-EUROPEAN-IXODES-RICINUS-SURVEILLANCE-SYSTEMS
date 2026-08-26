"""Optional "Download figure (PNG)" buttons that let a dissertation figure be
saved directly from the running dashboard — one figure at a time, using the
exact same figure-builder functions and Kaleido export as
export_dissertation_figures.py at the repo root. For generating the full
figure set in one go, use that script; these buttons are for grabbing a
single figure ad hoc while browsing the app.

Requires the kaleido package (see export_dissertation_figures.py's module
docstring for the install command). If kaleido isn't installed, clicking a
button surfaces Dash's normal callback-error handling — nothing else in the
tab is affected.
"""

from __future__ import annotations

from typing import Callable

import plotly.graph_objects as go
from dash import Input, Output, dcc, html
from dash.exceptions import PreventUpdate

from ui.styles import THEME_GREEN

BUTTON_STYLE = {
    "background": "#ffffff",
    "color": THEME_GREEN,
    "border": f"1px solid {THEME_GREEN}",
    "borderRadius": "6px",
    "padding": "6px 14px",
    "fontSize": "0.8rem",
    "fontWeight": "600",
    "cursor": "pointer",
}


def download_button(button_id: str, label: str = "⬇ Download figure (PNG)") -> html.Div:
    """One button + its matching dcc.Download — place directly above (or
    below) the dcc.Graph it downloads."""
    return html.Div(
        [
            html.Button(label, id=button_id, n_clicks=0, style=BUTTON_STYLE),
            dcc.Download(id=f"{button_id}-dl"),
        ],
        style={"textAlign": "right", "marginBottom": "6px"},
    )


def register_download(
    app,
    button_id: str,
    filename: str,
    figure_fn: Callable[[], go.Figure | None],
    width: int = 1600,
    height: int = 1000,
    scale: int = 2,
) -> None:
    """Wire a download_button() to export figure_fn() as a PNG via Kaleido,
    at the same width/height/scale convention export_dissertation_figures.py
    uses for print resolution. figure_fn is called fresh on each click (not
    memoised) so it always reflects the current snapshot; return None from
    it (e.g. no data for the current filters) to no-op the download."""

    @app.callback(
        Output(f"{button_id}-dl", "data"),
        Input(button_id, "n_clicks"),
        prevent_initial_call=True,
    )
    def _download(_n_clicks):
        fig = figure_fn()
        if fig is None:
            raise PreventUpdate
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
        return dcc.send_bytes(lambda buf: buf.write(img_bytes), filename)
