"""Dash application entry point."""

from __future__ import annotations

import warnings

from dash import Dash

from config import DASH_HOST, DASH_PORT
from data.pipeline import FrameworkSnapshot, build_framework
from ui.layout import create_layout, register_callbacks


warnings.filterwarnings(
    "ignore",
    message=r"ChainedAssignmentError",
    category=FutureWarning,
)


def create_app(snapshot: FrameworkSnapshot) -> Dash:
    """Wire immutable snapshot into Dash layout."""
    app = Dash(__name__, suppress_callback_exceptions=True, 
               meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
    app.title = "I. ricinus Surveillance Interoperability Assessment"
    app.layout = create_layout(snapshot)
    register_callbacks(app, snapshot)
    return app


# Built at import time so gunicorn can find `server`
snapshot = build_framework()
if not snapshot.ok:
    print("WARNING: No systems loaded — check data/systems.csv")

app = create_app(snapshot)
server = app.server          # <-- gunicorn targets this


def main() -> None:
    """Run the Dash dev server locally."""
    app.run(host=DASH_HOST, port=DASH_PORT, debug=False)


if __name__ == "__main__":
    main()
