"""
Screenshot the two dash-leaflet maps (Map tab, Ecological Suitability tab)
as dissertation figures.

These maps render entirely client-side (Leaflet.js: OSM basemap tiles,
GeoJSON country polygons styled by JS, the pre-generated raster overlay PNG)
— there is no Python figure object to hand to Kaleido the way
export_dissertation_figures.py does for the Plotly-based figures. Capturing
them means actually running the app, loading each tab in a real (headless)
browser, waiting for tiles/layers to finish drawing, and screenshotting the
map element — a different pipeline from the rest of the export scripts, not
a replacement for them.

Requires the `playwright` dev dependency (not part of the deployed app):
    tick/venv/bin/python -m pip install playwright
    tick/venv/bin/python -m playwright install chromium

Usage: tick/venv/bin/python scripts/export_leaflet_figures.py
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

SCRIPT_DIR = Path(__file__).resolve().parent
TICK_ROOT = SCRIPT_DIR.parent
DESSERTATION_ROOT = TICK_ROOT.parent
OUT_DIR = DESSERTATION_ROOT / "dissertation_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HOST = "127.0.0.1"
PORT = 8050  # config.DASH_HOST/DASH_PORT — not env-configurable, so this must match.
BASE_URL = f"http://{HOST}:{PORT}"

CAPTURES = [
    # (tab link text, map container CSS selector, output filename, extra wait for tiles/overlay)
    ("Map", ".leaflet-container", "fig_map_interoperability_leaflet.png", 2500),
    ("Ecological Suitability", ".leaflet-container", "fig_suitability_map_leaflet.png", 3000),
]


def _server_is_up() -> bool:
    try:
        with socket.create_connection((HOST, PORT), timeout=1.0):
            return True
    except OSError:
        return False


def _wait_for_server(timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _server_is_up():
            return
        time.sleep(0.5)
    raise RuntimeError(f"Dash server did not come up on {BASE_URL} within {timeout}s")


def main() -> None:
    proc = None
    if _server_is_up():
        print(f"Reusing already-running server on {BASE_URL}.")
    else:
        print(f"Starting app.py on {BASE_URL} ...")
        proc = subprocess.Popen(
            [sys.executable, "app.py"],
            cwd=TICK_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _wait_for_server()
        print("Server up.")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.goto(BASE_URL, wait_until="networkidle", timeout=30_000)
            page.wait_for_timeout(1500)

            for tab_label, selector, filename, extra_wait_ms in CAPTURES:
                print(f"{tab_label}: switching tab ...")
                page.locator(f"text={tab_label}").first.click()
                # Some tabs (Suitability) show a first render, then a Dash
                # callback swaps the leaflet container for the real one with
                # data-driven layers — networkidle waits out that callback's
                # request before the fixed extra wait for tiles to paint.
                page.wait_for_load_state("networkidle", timeout=15_000)
                page.wait_for_timeout(extra_wait_ms)

                out_path = OUT_DIR / filename
                # Retry once: the element found by the locator can still get
                # detached-and-replaced between locating it and the shot
                # itself if a callback re-renders right around this point.
                for attempt in range(2):
                    try:
                        el = page.locator(selector).first
                        el.wait_for(state="visible", timeout=15_000)
                        el.screenshot(path=str(out_path))
                        break
                    except Exception:
                        if attempt == 1:
                            raise
                        page.wait_for_timeout(1500)
                print(f"  wrote {filename} ({out_path.stat().st_size / 1024:.0f} KB)")

            browser.close()
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            print("Server stopped.")

    print(f"\nDone. Leaflet figures in {OUT_DIR}")


if __name__ == "__main__":
    main()
