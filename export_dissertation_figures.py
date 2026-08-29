"""
Batch-exports the dashboard's real Plotly figures, plus the notebook's
pre-rendered model-diagnostic PNGs and three key result tables, as
publication-quality files for the MSc dissertation.

This script builds the exact same FrameworkSnapshot and calls the exact same
figure-builder functions the running Dash app uses (ui.figures, and the
builder functions in ui/tabs/{scores,barriers,integration,map,suitability}.py)
— it does not recompute, re-derive, or invent any figure, score, or model
result. The only differences from the on-screen versions are export-only
presentation settings (explicit width/height/scale/margins for print
resolution) applied to a *copy* of each figure; nothing about the dashboard's
live behaviour changes. See the "EXPORT PRESENTATION" comment block below for
the exact list of what is adjusted and why.

Usage (from the tick/ directory, using the project virtualenv):

    venv/bin/python export_dissertation_figures.py

Output:

    ../dissertation_figures/*.png   (and .svg for a few line/heatmap charts)

Requires kaleido for static image export:

    venv/bin/python -m pip install kaleido==0.2.1

(Use `python -m pip`, not `pip`/`venv/bin/pip` directly — see the README note
in this repo about the venv's pip having a stale shebang.)
"""

from __future__ import annotations

import shutil
import sys
import warnings
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

# Same filter app.py applies — a pre-existing pandas Copy-on-Write warning
# from the data pipeline, unrelated to this script.
warnings.filterwarnings("ignore", message=r"ChainedAssignmentError", category=FutureWarning)

SCRIPT_DIR = Path(__file__).resolve().parent  # .../tick
sys.path.insert(0, str(SCRIPT_DIR))

from config import EVIDENCE_COLUMNS  # noqa: E402
from data.pipeline import build_framework  # noqa: E402
from core.barriers import barriers_summary_chart_df  # noqa: E402
from core.barrier_details import barrier_severity_distribution  # noqa: E402
from ui.figures import readiness_pie, score_histogram, criteria_heatmap, ranking_bar  # noqa: E402
from ui.tabs.map import _build_map_figure  # noqa: E402
from ui.tabs.scores import _criteria_averages, build_avg_criteria_chart, SCORES_TABLE_COLS  # noqa: E402
from ui.tabs.barriers import (  # noqa: E402
    build_severity_by_system_chart,
    build_severity_distribution_chart,
    BARRIER_DISPLAY_COLS,
)
from ui.tabs.integration import build_integration_chart, INTEGRATION_TABLE_COLS  # noqa: E402
from ui.tabs.recommendations import RECOMMENDATIONS_TABLE_COLS  # noqa: E402
from ui.tabs.suitability import (  # noqa: E402
    _get_notebook_data,
    _build_enhanced_map_figure,
    _mask_within_study_countries,
    DEFAULT_ENHANCED_LAYERS,
    FIGURES_DIR as DASHBOARD_STATIC_FIGURES_DIR,
)
from ui.styles import THEME_GREEN  # noqa: E402

# Same 4 colours core.geo/ui.tabs.suitability use for these countries
# elsewhere (READINESS_COLORS-adjacent, but this is a category-by-country
# map, not a readiness map, so it gets its own small, consistent palette).
STUDY_COUNTRY_COLORS = {
    "Austria": "#2d5a3d",
    "Croatia": "#2980b9",  # distinct from Austria's green — their points touch at the shared border
    "Estonia": "#f39c12",
    "Ireland": "#c0392b",
}

DISSERTATION_ROOT = SCRIPT_DIR.parent  # .../Dessertation
OUTPUT_DIR = DISSERTATION_ROOT / "dissertation_figures"

PNG_SCALE = 2  # write_image scale factor: uniformly upsamples the whole
# rendered figure (text, lines, markers together) with no change to relative
# proportions — this is how the 2000-3000px target width is reached without
# altering the chart's visual design.


def export_figure(fig: go.Figure, name: str, width: int, height: int, svg: bool = False) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    png_path = OUTPUT_DIR / f"{name}.png"
    fig.write_image(str(png_path), format="png", width=width, height=height, scale=PNG_SCALE)
    print(f"  wrote {png_path.name}  ({png_path.stat().st_size / 1024:.0f} KB, "
          f"{width * PNG_SCALE}x{height * PNG_SCALE}px)")
    if svg:
        svg_path = OUTPUT_DIR / f"{name}.svg"
        fig.write_image(str(svg_path), format="svg", width=width, height=height)
        print(f"  wrote {svg_path.name}")


_HEADER_CHAR_PX = 9.2  # avg. character width at the header's bold 14px font
_BODY_CHAR_PX = 7.2    # avg. character width at the cells' regular 13px font
_LINE_PX = 20        # line height at that font size, for wrapped rows
_CELL_VPAD = 16      # vertical padding above+below text in a cell
_HEADER_HEIGHT = 40
_MAX_COL_PX = 460    # free-text columns (URLs, recommendation sentences)
# are capped here and left to wrap onto multiple lines, rather than
# sizing the whole column to its single longest un-wrapped value —
# uncapped, one long URL or sentence made the *entire* table image
# thousands of pixels wide (e.g. the Evidence table hit 14,600px).
_MIN_COL_PX = 90


def _wrap_cell_text(text: str, max_chars: int) -> str:
    """Hard-wrap text to at most max_chars per line, inserting explicit
    <br> tags for every line break.

    Plotly Table cells wrap at whitespace on their own, like a browser —
    but that was observed (checked directly against Kaleido's rendered
    output, not assumed) to stop working for the rest of a cell once one
    explicit <br> already appears earlier in it. A cell mixing ordinary
    prose with one long trailing URL — common in the Evidence table's
    "reference" column — hit exactly that: the URL got a manual break, but
    the prose before it then rendered as one long unwrapped line that
    overflowed the column. Wrapping every cell fully ourselves, rather than
    mixing one manual break with Plotly's implicit wrap, avoids that.

    A word (e.g. a URL) longer than max_chars on its own is still broken —
    preferably right after a '/' or '-' near the limit, or with a hard cut
    if it has neither — so no single line can ever overflow the column.
    """
    lines: list[str] = []
    current = ""
    for word in text.split(" "):
        candidate = f"{current} {word}".strip() if current else word
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = ""
        if len(word) <= max_chars:
            current = word
            continue
        remaining = word
        while len(remaining) > max_chars:
            window = remaining[: max_chars + 1]
            break_at = max(window.rfind("/"), window.rfind("-"))
            if break_at < max_chars * 0.4:  # too early a break reads worse than a clean cut
                break_at = max_chars
            else:
                break_at += 1  # keep the separator on the line it ends
            lines.append(remaining[:break_at])
            remaining = remaining[break_at:]
        current = remaining
    if current:
        lines.append(current)
    return "<br>".join(lines) if lines else text


def _dataframe_to_table_figure(df: pd.DataFrame, title: str) -> tuple[go.Figure, int, int]:
    """Render an existing dashboard DataTable's data as a Plotly table image,
    styled to match ui/tables.py's actual TABLE_HEADER/TABLE_CELL colours
    (THEME_GREEN header, white cells) — used for every dashboard table that
    has no dcc.Graph of its own (model performance, transfer matrix,
    external validation, sub-scores, barrier comparison/detail, integration
    summary, recommendations, evidence).

    Column widths are sized from each column's own header/content length,
    capped at _MAX_COL_PX so a long free-text field wraps onto multiple
    lines instead of stretching the whole table absurdly wide. cells.height
    is left unset so Plotly auto-computes each row's height from its
    wrapped content (verified directly against Kaleido's actual output,
    not assumed); the returned height is a matching estimate, generous by
    design, for the caller's write_image() call — a little blank space at
    the bottom is fine, a clipped row is not.

    Returns (figure, width, height).
    """
    display = df.copy()
    for col in display.select_dtypes(include=["float", "float32", "float64"]).columns:
        display[col] = display[col].round(3)

    col_px: list[float] = []
    wrap_chars: list[int] = []
    for col in display.columns:
        # Headers never wrap, so their width is a hard floor regardless of
        # the cap below — only body content is allowed to wrap/cap.
        header_px = len(str(col)) * _HEADER_CHAR_PX + 30
        body_len = display[col].astype(str).map(len).max() if len(display) else 0
        body_px = min(body_len * _BODY_CHAR_PX + 30, _MAX_COL_PX)
        px = max(_MIN_COL_PX, header_px, body_px)
        col_px.append(px)
        wrap_chars.append(max(10, int((px - 24) / _BODY_CHAR_PX)))
    total_width = int(sum(col_px) + 40)

    # Wrap every cell ourselves (see _wrap_cell_text) rather than relying on
    # Plotly's own wrap, so line count is exactly known — both the rendered
    # table and this height estimate read off the same <br>-delimited text.
    # fillna("") first so a missing value renders as a blank cell, not the
    # literal text "nan" that plain str() would produce.
    for i, col in enumerate(display.columns):
        display[col] = display[col].fillna("").astype(str).map(lambda v, wc=wrap_chars[i]: _wrap_cell_text(v, wc))

    total_height = _HEADER_HEIGHT
    for _, row in display.iterrows():
        max_lines = max(str(row[col]).count("<br>") + 1 for col in display.columns)
        total_height += max_lines * _LINE_PX + _CELL_VPAD
    # Line counts are exact (we inserted every <br> ourselves), so this only
    # needs a small flat buffer for the title bar/margins, not a percentage
    # safety margin — the first version of this formula used a generous 15%
    # multiplier (left over from when Plotly's own wrap was still involved)
    # and produced a noticeably blank bottom third on longer tables.
    total_height = int(total_height + 80)

    fig = go.Figure(
        data=[
            go.Table(
                columnwidth=col_px,
                header=dict(
                    values=[f"<b>{c}</b>" for c in display.columns],
                    fill_color=THEME_GREEN,
                    font=dict(color="white", size=14),
                    align="left",
                    height=_HEADER_HEIGHT,
                ),
                cells=dict(
                    values=[display[c] for c in display.columns],
                    fill_color=[["white", "#fbfcfc"] * len(display)],
                    font=dict(color="#2c3e50", size=13),
                    align="left",
                ),
            )
        ]
    )
    fig.update_layout(
        title=dict(text=title, x=0.01, font=dict(size=16, color=THEME_GREEN)),
        margin=dict(t=50, b=10, l=10, r=10),
        paper_bgcolor="white",
    )
    return fig, total_width, total_height


def _build_spatial_validation_blocks_map(occurrence_points: pd.DataFrame) -> go.Figure:
    """Real map replacement for outputs/figures/spatial_validation_blocks.png
    (the matplotlib original has no basemap, coastlines, or country borders —
    just a bare lat/lon scatter). Same per-country colouring the source image
    used, same real-polygon restriction to the 4 study countries as the
    suitability map, same basemap style as the rest of this script's maps.
    """
    df = occurrence_points.copy()
    mask = _mask_within_study_countries(df["lat"].to_numpy(), df["lon"].to_numpy())
    df = df.loc[mask]

    traces = []
    for country in ["Austria", "Croatia", "Estonia", "Ireland"]:
        subset = df[df["system"] == country]
        if subset.empty:
            continue
        traces.append(
            go.Scattergeo(
                lat=subset["lat"], lon=subset["lon"],
                mode="markers",
                marker=dict(size=6, color=STUDY_COUNTRY_COLORS[country], opacity=0.75, line=dict(width=0.5, color="white")),
                name=f"{country} ({len(subset)} pts)",
            )
        )

    fig = go.Figure(data=traces)
    fig.update_geos(
        scope="europe",
        projection_type="natural earth",
        showcountries=True,
        showland=True,
        showocean=True,
        landcolor="#f9f9f9",
        oceancolor="#e8f4f8",
        countrycolor="#666666",
        lataxis_range=[34, 62],
        lonaxis_range=[-12, 32],
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        title=dict(text="Spatial Cross-Validation Blocks by Country", x=0.5, xanchor="center", font=dict(size=16, color=THEME_GREEN)),
        margin=dict(t=50, b=10, l=10, r=10),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#bdc3c7", borderwidth=1),
        paper_bgcolor="white",
        font=dict(family="system-ui, sans-serif", size=13, color="#2c3e50"),
    )
    return fig


def _build_presence_absence_map(occurrence_points: pd.DataFrame) -> go.Figure:
    """Real map replacement for outputs/figures/observed_presence_map.png and
    presence_background.png — both had correct data and legends but no
    basemap/coastlines/country borders (bare lat/lon scatter). Shows presence
    AND background (absence) points together across all 4 study countries,
    on the same basemap style as the other maps in this script.
    """
    df = occurrence_points.copy()
    mask = _mask_within_study_countries(df["lat"].to_numpy(), df["lon"].to_numpy())
    df = df.loc[mask]

    background = df[df["type"] == "background"]
    presence = df[df["type"] == "presence"]

    traces = [
        go.Scattergeo(
            lat=background["lat"], lon=background["lon"],
            mode="markers",
            marker=dict(size=5, color="#9e9e9e", opacity=0.6, line=dict(width=0.5, color="white")),
            name=f"Absence ({len(background)})",
        ),
        go.Scattergeo(
            lat=presence["lat"], lon=presence["lon"],
            mode="markers",
            marker=dict(size=6, color="#27ae60", opacity=0.8, line=dict(width=0.5, color="white")),
            name=f"Presence ({len(presence)})",
        ),
    ]

    fig = go.Figure(data=traces)
    fig.update_geos(
        scope="europe",
        projection_type="natural earth",
        showcountries=True,
        showland=True,
        showocean=True,
        landcolor="#f9f9f9",
        oceancolor="#e8f4f8",
        countrycolor="#666666",
        lataxis_range=[34, 62],
        lonaxis_range=[-12, 32],
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        title=dict(
            text="Observed Presence and Background Locations (training: Austria, Croatia — external: Estonia, Ireland)",
            x=0.5, xanchor="center", font=dict(size=15, color=THEME_GREEN),
        ),
        # r=260, not the ~10px the other maps on this page use: a geo subplot
        # doesn't auto-reserve canvas space for an outside legend the way a
        # cartesian plot does, and this legend's longest label ("Background /
        # absence (357 pts)") was clipping past the image edge without an
        # explicit, generous right margin reserved for it.
        margin=dict(t=60, b=10, l=10, r=260),
        legend=dict(x=1.0, xanchor="left", y=0.95, bgcolor="rgba(255,255,255,0.9)", bordercolor="#bdc3c7", borderwidth=1),
        paper_bgcolor="white",
        font=dict(family="system-ui, sans-serif", size=13, color="#2c3e50"),
    )
    return fig


def main() -> None:
    print(f"Output directory: {OUTPUT_DIR}")
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir()

    # ---- 1. Interoperability scorecard snapshot (same object app.py builds) ----
    snapshot = build_framework()
    if not snapshot.ok:
        print("ERROR: snapshot has no systems — check tick/data/systems.csv")
        sys.exit(1)
    systems = snapshot.systems
    n_systems = len(systems)
    print(f"Loaded snapshot: {n_systems} systems")

    # --- Overview tab ---
    print("Overview tab:")
    export_figure(readiness_pie(systems), "fig_overview_readiness_distribution", width=1000, height=800)
    export_figure(score_histogram(systems), "fig_overview_score_distribution", width=1300, height=800)

    # --- Map tab ---
    print("Map tab:")
    export_figure(_build_map_figure(snapshot.map_scatter), "fig_map_interoperability_europe", width=1500, height=1000)

    # --- Scores tab ---
    print("Scores tab:")
    bar_height = max(900, 45 * n_systems + 250)
    export_figure(ranking_bar(systems), "fig_scores_system_ranking", width=1500, height=bar_height)
    heat_height = max(850, 40 * n_systems + 250)
    export_figure(criteria_heatmap(systems), "fig_scores_criteria_heatmap", width=1500, height=heat_height, svg=True)
    avg = _criteria_averages(systems)
    export_figure(build_avg_criteria_chart(avg), "fig_scores_avg_per_criterion", width=1400, height=800)
    # Sub-score table (system_id, system_name, technical_subscore,
    # governance_subscore, total_score) — the exact columns the Scores tab
    # itself displays (SCORES_TABLE_COLS), not the full 31-column systems df.
    scores_cols = [c for c in SCORES_TABLE_COLS if c in systems.columns]
    fig, width, height = _dataframe_to_table_figure(systems[scores_cols], "Technical / Governance Sub-scores")
    export_figure(fig, "table_scores_subscores", width=width, height=height)

    # --- Barriers tab ---
    print("Barriers tab:")
    barriers = snapshot.barriers
    details = snapshot.barrier_details
    chart_df = barriers_summary_chart_df(barriers)
    fig_bar = build_severity_by_system_chart(chart_df)
    if fig_bar is not None:
        export_figure(fig_bar, "fig_barriers_severity_by_system", width=max(1600, 100 * n_systems), height=950)
    sev = barrier_severity_distribution(details)
    fig_sev = build_severity_distribution_chart(sev)
    if fig_sev is not None:
        export_figure(fig_sev, "fig_barriers_severity_distribution", width=1100, height=800)
    # Barrier comparison table — same columns as the tab's own
    # BARRIER_DISPLAY_COLS (barrier_summary is excluded there too — a text
    # restatement of the five structured columns already shown).
    barrier_cols = [c for c in BARRIER_DISPLAY_COLS if c in barriers.columns]
    fig, width, height = _dataframe_to_table_figure(barriers[barrier_cols], "Barrier Comparison by System")
    export_figure(fig, "table_barriers_summary", width=width, height=height)
    if details is not None and not details.empty:
        # Barrier Detail — Reason & Recommendation: one row per system per
        # flagged (Medium/High-severity) barrier issue, all columns — same
        # data the tab's own detail table shows in full.
        fig, width, height = _dataframe_to_table_figure(details, "Barrier Detail — Reason & Recommendation")
        export_figure(fig, "table_barriers_detail", width=width, height=height)
    else:
        print("  SKIPPED barrier detail table — no Medium/High severity issues flagged")

    # --- Integration tab ---
    print("Integration tab:")
    integration_df = snapshot.integration
    export_figure(
        build_integration_chart(integration_df),
        "fig_integration_readiness_by_system",
        width=max(1600, 100 * n_systems),
        height=950,
    )
    # The canonical system-by-system table (rank, score, readiness class,
    # barrier level, integration class, barrier summary) — this is the
    # dashboard's headline record and the most likely dissertation appendix
    # table: it's the one place all four verdicts sit side by side per system.
    integration_cols = [c for c in INTEGRATION_TABLE_COLS if c in integration_df.columns]
    fig, width, height = _dataframe_to_table_figure(integration_df[integration_cols], "System-by-System Integration Summary")
    export_figure(fig, "table_integration_system_summary", width=width, height=height)

    # --- Recommendations tab ---
    print("Recommendations tab:")
    recs = snapshot.recommendations
    if recs is not None and not recs.empty:
        rec_cols = [c for c in RECOMMENDATIONS_TABLE_COLS if c in recs.columns]
        fig, width, height = _dataframe_to_table_figure(recs[rec_cols], "Rule-Based Recommendations")
        export_figure(fig, "table_recommendations", width=width, height=height)
    else:
        print("  SKIPPED — no recommendations generated")

    # --- Evidence tab ---
    print("Evidence tab:")
    evidence = snapshot.evidence
    if evidence is not None and not evidence.empty:
        ev_cols = [c for c in EVIDENCE_COLUMNS if c in evidence.columns]
        # Plain text, not the tab's markdown-link formatting (e.g.
        # "[https://x](https://x)") — that syntax is for the Dash DataTable's
        # markdown renderer and would show as literal brackets in a static
        # table image, so it's left as plain URL text here instead.
        fig, width, height = _dataframe_to_table_figure(evidence[ev_cols], "Evidence Traceability")
        export_figure(fig, "table_evidence", width=width, height=height)
    else:
        print("  SKIPPED — no evidence records")

    # --- Ecological Suitability tab ---
    print("Ecological Suitability tab:")
    notebook_data = _get_notebook_data()
    if notebook_data.available:
        # max_suitability_points=None: plot the full suitability grid, not the
        # ~10k-point interactive-performance subsample the live dashboard map
        # uses. This is a one-off static render, so the performance cap that
        # exists purely for browser pan/zoom smoothness doesn't apply.
        # restrict_to_study_countries=True: only colour Austria, Croatia,
        # Estonia, Ireland — the four countries the model was actually
        # trained/validated on — rather than the full raw grid, which
        # extrapolates suitability across all of Europe and beyond with no
        # visual distinction from the evidenced area. See the docstring on
        # _build_enhanced_map_figure in ui/tabs/suitability.py.
        suit_fig = _build_enhanced_map_figure(
            notebook_data, DEFAULT_ENHANCED_LAYERS,
            max_suitability_points=None, restrict_to_study_countries=True,
        )
        export_figure(suit_fig, "fig_suitability_map_full_resolution", width=1500, height=960)

        # "observed_vs_suitability" replacement: outputs/figures/observed_vs_suitability.png
        # is broken (title literally says "Suitability surface not shown -
        # requires raster processing", and has no basemap) — this is
        # genuinely the same content, done correctly: occurrence points over
        # the real suitability surface, on a proper map, restricted to the
        # study countries. Same figure object as fig_suitability_map_full_resolution
        # (that IS the corrected "observed vs suitability" map) under the
        # expected filename, so both dissertation section references resolve.
        export_figure(suit_fig, "fig_suitability_observed_vs_predicted", width=1500, height=960)

        # "spatial_validation_blocks" replacement: same broken-basemap problem
        # (no coastlines/borders, just raw scatter) — real map version below.
        blocks_fig = _build_spatial_validation_blocks_map(notebook_data.occurrence_points)
        export_figure(blocks_fig, "fig_suitability_spatial_validation_blocks", width=1500, height=960)

        # Presence + background (absence) locations together, across all 4
        # study countries, on a real basemap — replaces observed_presence_map.png
        # / presence_background.png, which had correct data but no map context.
        pa_fig = _build_presence_absence_map(notebook_data.occurrence_points)
        export_figure(pa_fig, "fig_suitability_presence_absence_map", width=1500, height=960)
    else:
        print("  SKIPPED — notebook outputs not available (see ui/tabs/suitability.py _missing_data_block).")

    # --- Notebook-generated static PNGs, copied through unchanged (matplotlib,
    # not Plotly, so Kaleido can't regenerate them). All individually checked
    # for rendering bugs (see chat write-up) — feature_importance and
    # partial_dependence_curves are the two already shown in the dashboard;
    # the rest are extra modelling-evidence figures from the full notebook
    # pipeline (Ixodes_ricinus_Model.ipynb) that aren't in the dashboard but
    # are genuinely useful dissertation evidence — cleaning_funnel had a real
    # duplicate-row bug (Estonia/Ireland listed twice) that's been fixed at
    # the source (data/report + notebook code) before this copy. ---
    print("Static notebook figures shown in the dashboard (copied, not regenerated):")
    static_figures = [
        ("feature_importance.png", "fig_suitability_feature_importance.png"),
        ("partial_dependence_curves.png", "fig_suitability_partial_dependence.png"),
        ("architecture.png", "fig_pipeline_architecture.png"),
        ("cleaning_funnel.png", "fig_suitability_cleaning_funnel.png"),
        ("leakage_comparison.png", "fig_suitability_leakage_comparison.png"),
        ("feature_set_progression.png", "fig_suitability_feature_set_progression.png"),
        ("model_comparison_heatmap.png", "fig_suitability_model_comparison_heatmap.png"),
        ("metrics_heatmap.png", "fig_suitability_metrics_heatmap.png"),
        ("external_validation_heatmap.png", "fig_suitability_external_validation_heatmap.png"),
        ("environmental_distributions.png", "fig_suitability_environmental_distributions.png"),
        ("roc_pr_curves.png", "fig_suitability_roc_pr_curves.png"),
    ]
    for src_name, dst_name in static_figures:
        src = DASHBOARD_STATIC_FIGURES_DIR / src_name
        if src.exists():
            dst = OUTPUT_DIR / dst_name
            shutil.copy2(src, dst)
            print(f"  copied {dst_name}  ({dst.stat().st_size / 1024:.0f} KB — matplotlib source, "
                  f"150 DPI; see notes on resolution in the chat write-up)")
        else:
            print(f"  MISSING: {src} — not copied. Re-run the notebook pipeline to regenerate it.")

    # --- Suitability tab result tables (rendered as Dash DataTables in the
    # dashboard, not as charts — exported here as table images that match the
    # dashboard's own header colour/typography, for anyone who wants a camera-
    # ready image rather than a native Word table). ---
    print("Suitability tab result tables:")
    if notebook_data.model_results is not None:
        fig, width, height = _dataframe_to_table_figure(notebook_data.model_results, "Model Performance Comparison")
        export_figure(fig, "table_suitability_model_performance", width=width, height=height)
    else:
        print("  SKIPPED model_results — outputs/model_results.csv not found")

    if notebook_data.transfer_matrix is not None:
        # transfer_matrix.csv's row-label column has no header in the source
        # file (pandas reads it as "Unnamed: 0"); every row is a country name
        # (Austria/Croatia/Estonia/Ireland), matching the other four column
        # headers, so "System" is the accurate label — a display-only rename,
        # the CSV and its values are untouched.
        tm = notebook_data.transfer_matrix.rename(
            columns={notebook_data.transfer_matrix.columns[0]: "System"}
        )
        fig, width, height = _dataframe_to_table_figure(tm, "Cross-System Transfer Matrix")
        export_figure(fig, "table_suitability_transfer_matrix", width=width, height=height)
    else:
        print("  SKIPPED transfer_matrix — outputs/transfer_matrix.csv not found")

    if notebook_data.external_validation is not None:
        fig, width, height = _dataframe_to_table_figure(notebook_data.external_validation, "External Validation Results")
        export_figure(fig, "table_suitability_external_validation", width=width, height=height)
    else:
        print("  SKIPPED external_validation — outputs/external_validation.csv not found")

    print(f"\nDone. {len(list(OUTPUT_DIR.glob('*')))} files written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
