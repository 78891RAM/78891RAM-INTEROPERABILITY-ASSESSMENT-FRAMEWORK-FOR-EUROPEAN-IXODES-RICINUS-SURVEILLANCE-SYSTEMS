"""dash-leaflet-based geographic views: interoperability_map.py (14-system
readiness map) and suitability_map.py (ecological suitability + occurrence
records). map_layers.py holds shared constants/loaders used by both.

Kept separate from the Plotly charts elsewhere in ui/ (bar charts, heatmaps,
ROC-style comparisons) — those stay Plotly; only the two GIS-style maps
moved to dash-leaflet. See docs/architecture.md / CONTEXT.md for the
rationale (dash-leaflet + GeoPandas + Rasterio for real country polygons
and NaN-masked raster overlays, which Plotly's choropleth/scattergeo
can't do).
"""
