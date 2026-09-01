"""Shared constants and cached loaders for both dash-leaflet maps.

Boundary GeoJSON and raster-overlay metadata are loaded once and cached at
module level (item 18: no per-callback disk/geometry work) — callbacks
below only ever switch visibility/selection, never re-read or re-process
geographic data.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # tick/
DATA_DIR = PROJECT_ROOT / "data"
OVERLAYS_DIR = PROJECT_ROOT / "assets" / "overlays"

# ---------------------------------------------------------------------------
# Study-system roles (item 4). Estonia/Ireland are always "exploratory
# held-out transfer" here — never "validation" — because the held-out
# samples are very small (7 and 19 unique locations respectively; see
# outputs/external_validation.csv). This wording applies only to the
# ecological-suitability model's country roles; it is unrelated to, and
# must not be confused with, the model's own spatial-block cross-validation
# (a genuine statistical validation procedure, done during model training —
# see leakage_comparison.png).
# ---------------------------------------------------------------------------
MODEL_DEVELOPMENT_COUNTRIES = ["Austria", "Croatia"]
EXPLORATORY_TRANSFER_COUNTRIES = ["Estonia", "Ireland"]
STUDY_COUNTRY_ROLE = {
    "Austria": "Model development",
    "Croatia": "Model development",
    "Estonia": "Exploratory held-out transfer",
    "Ireland": "Exploratory held-out transfer",
}

SUITABILITY_INTERPRETATION_NOTE = (
    "Suitability represents proof-of-concept environmental model output and "
    "should not be interpreted as a validated Europe-wide distribution or "
    "disease-risk map. Occurrence points represent observed surveillance "
    "records."
)

# ---------------------------------------------------------------------------
# Colour-blind-friendly categorical palette (Okabe-Ito based) for the
# interoperability readiness map — item 14. Deliberately not red/green:
# blue vs orange vs vermillion stays distinguishable under deuteranopia,
# protanopia, and tritanopia alike, unlike the traffic-light red/amber/green
# the Plotly map previously used.
# ---------------------------------------------------------------------------
INTEROP_COLORS = {
    "High": "#0072B2",     # blue
    "Medium": "#E69F00",   # orange
    "Low": "#D55E00",      # vermillion
    "Not assessed": "#B0B0B0",  # neutral grey — never implies "Low"
}

# ---------------------------------------------------------------------------
# Suitability colour scale (item 7): fixed 0-1, sequential, colour-vision-
# friendly. Matches scripts/generate_raster_overlays.py, which pre-renders
# the actual overlay images at this same scale — this constant is for any
# UI-side legend/colourbar rendering, not for recolouring the image itself.
# ---------------------------------------------------------------------------
SUITABILITY_CMAP = "cividis"
SUITABILITY_VMIN = 0.0
SUITABILITY_VMAX = 1.0

_europe_boundaries_cache: dict | None = None
_study_boundaries_cache: dict | None = None
_overlay_meta_cache: dict[str, dict] = {}


def load_europe_boundaries() -> dict:
    """All ~50 European countries (item 14's basemap for the interoperability
    map), each tagged properties.assessed=True for the 10 countries that
    have an interoperability system, False otherwise — the 'Not assessed'
    category is a real, positively-styled state, not just absent colour."""
    global _europe_boundaries_cache
    if _europe_boundaries_cache is None:
        with open(DATA_DIR / "europe_country_boundaries.geojson") as f:
            _europe_boundaries_cache = json.load(f)
    return _europe_boundaries_cache


def load_study_country_boundaries() -> dict:
    """The 4 ecological-suitability study countries (Austria, Croatia,
    Estonia, Ireland) — same file the suitability Plotly map already used
    for its real-polygon restriction (ui/tabs/suitability.py)."""
    global _study_boundaries_cache
    if _study_boundaries_cache is None:
        with open(DATA_DIR / "study_country_boundaries.geojson") as f:
            _study_boundaries_cache = json.load(f)
    return _study_boundaries_cache


def load_overlay_meta(layer_name: str) -> dict | None:
    """Bounds/vmin/vmax/cmap sidecar for a pre-generated raster overlay PNG
    (assets/overlays/<layer_name>.png + .json) — see
    scripts/generate_raster_overlays.py. Returns None if that layer hasn't
    been generated (e.g. rerun the script after a raster changes)."""
    if layer_name in _overlay_meta_cache:
        return _overlay_meta_cache[layer_name]
    meta_path = OVERLAYS_DIR / f"{layer_name}.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    _overlay_meta_cache[layer_name] = meta
    return meta


def overlay_image_url(layer_name: str) -> str:
    """Dash serves assets/ automatically at /assets/... — the overlay PNGs
    from scripts/generate_raster_overlays.py need no extra routing."""
    return f"/assets/overlays/{layer_name}.png"
