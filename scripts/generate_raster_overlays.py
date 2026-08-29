"""
One-time (offline) raster pre-processing for the Ecological Suitability
Leaflet map. Run this locally whenever the source rasters change — NOT part
of the deployed app's runtime (item 18: don't process large GeoTIFFs on
every callback / at every startup).

For the final suitability raster and each of the four environmental-context
rasters (bio04, bio12, bio15, NDVI), this:
  1. Reads only the windowed region covering the 4 study countries' real
     polygon boundaries (rasterio windowed read — never loads the full
     global raster, which is ~900M pixels).
  2. Masks to those real polygons (geopandas/rasterio.mask) AND to the
     raster's own nodata — both become transparent, not zero.
  3. Renders a small PNG at a fixed colour scale (suitability: 0-1,
     cividis; environmental layers: their own data range, viridis) with a
     transparent alpha channel outside the valid/study-country area.
  4. Writes the PNG plus its geographic bounds (for dl.ImageOverlay) to
     tick/assets/overlays/.

Usage: tick/venv/bin/python scripts/generate_raster_overlays.py
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import rasterio
import rasterio.mask
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
TICK_ROOT = SCRIPT_DIR.parent
DESSERTATION_ROOT = TICK_ROOT.parent

BOUNDARIES_PATH = TICK_ROOT / "data" / "study_country_boundaries.geojson"
OUT_DIR = TICK_ROOT / "assets" / "overlays"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Suitability: fixed 0-1 domain, sequential colour-vision-friendly scale
# (item 7). Environmental layers: rendered on their own actual data range
# (there's no natural shared 0-1 domain for e.g. mm of precipitation),
# still viridis (item 7's alternative), still masked identically.
LAYERS = {
    "suitability": {
        "path": DESSERTATION_ROOT / "rasters" / "final_suitability.tif",
        "cmap": "cividis",
        "vmin": 0.0,
        "vmax": 1.0,
    },
    "bio04": {
        "path": DESSERTATION_ROOT / "Raw_data" / "rasters" / "CHELSA_bio04_1981-2010_V.2.1.tif",
        "cmap": "viridis",
        "vmin": None,  # computed from the masked data itself
        "vmax": None,
    },
    "bio12": {
        "path": DESSERTATION_ROOT / "Raw_data" / "rasters" / "CHELSA_bio12_1981-2010_V.2.1.tif",
        "cmap": "viridis",
        "vmin": None,
        "vmax": None,
    },
    "bio15": {
        "path": DESSERTATION_ROOT / "Raw_data" / "rasters" / "CHELSA_bio15_1981-2010_V.2.1.tif",
        "cmap": "viridis",
        "vmin": None,
        "vmax": None,
    },
    "ndvi": {
        "path": DESSERTATION_ROOT / "Raw_data" / "rasters" / "avg_ndvi.tif",
        "cmap": "viridis",
        "vmin": None,
        "vmax": None,
    },
}


def _load_study_country_geometries() -> list:
    with open(BOUNDARIES_PATH) as f:
        geojson = json.load(f)
    gdf = gpd.GeoDataFrame.from_features(geojson["features"], crs="EPSG:4326")
    return list(gdf.geometry)


def _render_layer(name: str, cfg: dict, geometries: list) -> None:
    path = cfg["path"]
    if not path.exists():
        print(f"  SKIPPED {name} — source raster not found: {path}")
        return

    with rasterio.open(path) as src:
        # mask()'s own fill value must match the source dtype (these CHELSA
        # rasters are uint16 — NaN can't be used directly here). Use the
        # source's own nodata (or 0 as a fallback) for the fill, then cast
        # to float32 and convert that sentinel to NaN ourselves — same
        # masked-outside-study-area result, dtype-safe.
        fill_sentinel = src.nodata if src.nodata is not None else 0
        # Windowed + masked read: only decodes the pixels covering the 4
        # country polygons, never the full (potentially global) raster.
        out_image, out_transform = rasterio.mask.mask(
            src, geometries, crop=True, nodata=fill_sentinel, filled=True
        )
        data = out_image[0].astype("float32")
        data[data == fill_sentinel] = np.nan

        height, width = data.shape
        west, north = out_transform * (0, 0)
        east, south = out_transform * (width, height)

    vmin = cfg["vmin"] if cfg["vmin"] is not None else float(np.nanmin(data))
    vmax = cfg["vmax"] if cfg["vmax"] is not None else float(np.nanmax(data))

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = mpl.colormaps[cfg["cmap"]]
    rgba = cmap(norm(np.where(np.isnan(data), vmin, data)))  # placeholder value, alpha fixed below
    rgba[..., 3] = np.where(np.isnan(data), 0.0, 0.92)  # transparent outside valid/study area

    img = Image.fromarray((rgba * 255).astype("uint8"), mode="RGBA")
    png_path = OUT_DIR / f"{name}.png"
    img.save(png_path)

    meta = {
        "bounds": [[south, west], [north, east]],  # dl.ImageOverlay format: [[south,west],[north,east]]
        "vmin": vmin,
        "vmax": vmax,
        "cmap": cfg["cmap"],
        "width": width,
        "height": height,
    }
    meta_path = OUT_DIR / f"{name}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  wrote {png_path.name} ({width}x{height}px, range [{vmin:.3g}, {vmax:.3g}]) + {meta_path.name}")


def main() -> None:
    geometries = _load_study_country_geometries()
    print(f"Study-country geometries loaded: {len(geometries)}")
    for name, cfg in LAYERS.items():
        print(f"{name}:")
        _render_layer(name, cfg, geometries)
    print(f"\nDone. Overlays in {OUT_DIR}")


if __name__ == "__main__":
    main()
