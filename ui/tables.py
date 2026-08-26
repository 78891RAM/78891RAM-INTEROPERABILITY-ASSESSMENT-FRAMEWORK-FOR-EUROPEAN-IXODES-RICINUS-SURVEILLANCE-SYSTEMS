"""Shared Dash DataTable styling and factory."""

from __future__ import annotations

from dash import dash_table, html

from config import BARRIER_SEVERITY_COLORS, INTEGRATION_COLORS
from ui.styles import THEME_GREEN, TEXT_DARK, BORDER_LIGHT, NEUTRAL_CARD

TABLE_CELL = {
    "textAlign": "left",
    "padding": "12px 16px",
    "fontFamily": "system-ui, -apple-system, sans-serif",
    "fontSize": "13px",
    "whiteSpace": "normal",
    "height": "auto",
    "verticalAlign": "top",
    "color": TEXT_DARK,
    "lineHeight": "1.5",
}

TABLE_HEADER = {
    "backgroundColor": THEME_GREEN,
    "color": "#ffffff",
    "fontWeight": "600",
    "textAlign": "left",
    "padding": "14px 16px",
    "fontFamily": "system-ui, -apple-system, sans-serif",
    "fontSize": "13px",
    "letterSpacing": "0.02em",
}

TABLE_STYLE = {
    "width": "fit-content",
    "maxWidth": "100%",
    "margin": "0",
    "overflowX": "auto",
}

# TABLE_FILTER removed - no longer using in-table filters

TABLE_WRAPPER = {
    "marginBottom": "32px",
    "maxWidth": "1080px",
    "margin": "0 auto 32px",
    "overflowX": "auto",
    "background": NEUTRAL_CARD,
    "borderRadius": "12px",
    "border": f"1px solid {BORDER_LIGHT}",
    "boxShadow": "0 2px 8px rgba(45, 90, 61, 0.06), 0 1px 3px rgba(45, 90, 61, 0.04)",
}

ZEBRA_STRIPE = [
    {"if": {"row_index": "odd"}, "backgroundColor": "#fbfcfc"},
]

# Shared pastel palette for every High/Medium/Low-style verdict column in the
# app (readiness_class, integration_class, barrier_level) — previously only
# readiness_class got this treatment; extended per the UI-consistency audit so
# every tab whose headline column is a classification gets the same green/
# amber/red styling, not just the ones that show readiness_class.
_VERDICT_PASTELS = {
    "High": {"backgroundColor": "#d5f5e3", "color": "#1e5631"},
    "Medium": {"backgroundColor": "#fdebd0", "color": "#9a6700"},
    "Low": {"backgroundColor": "#fadbd8", "color": "#922b21"},
}


def _level_from_label(label: str) -> str | None:
    """'High integration ready' -> 'High', 'Medium' -> 'Medium', 'Unknown' -> None."""
    for level in ("High", "Medium", "Low"):
        if label.startswith(level):
            return level
    return None


def _verdict_cell_styles(column_id: str, labels: list[str]) -> list[dict]:
    styles = []
    for label in labels:
        pastel = _VERDICT_PASTELS.get(_level_from_label(label))
        if not pastel:
            continue
        styles.append({
            "if": {"filter_query": f'{{{column_id}}} = "{label}"', "column_id": column_id},
            **pastel,
            "fontWeight": "600",
        })
    return styles


READINESS_CELL_STYLES = _verdict_cell_styles("readiness_class", ["High", "Medium", "Low"])
# Sourced from config.py's own colour maps so the set of values stays in sync
# with core.integration / core.barriers if those classes ever change.
INTEGRATION_CELL_STYLES = _verdict_cell_styles("integration_class", list(INTEGRATION_COLORS))
BARRIER_LEVEL_CELL_STYLES = _verdict_cell_styles("barrier_level", list(BARRIER_SEVERITY_COLORS))

ID_COLS = frozenset({"system_id"})
NUMERIC_COLS = frozenset(
    {
        "total_score",
        "count",
        "severity_rank",
        "criteria_scored",
        "priority",
        "barrier_count",
        "avg_score",
        "rank",
        "technical_subscore",
        "governance_subscore",
    }
)
NAME_COLS = frozenset({"system_name"})
TEXT_COLS = frozenset(
    {
        "official_website",
        "barrier_summary",
        "publisher",
        "countries_covered",
        "reason",
        "research_publication",
        "operator_org",
        "url",
        "criterion",
        "recommendation",
        "reference",
        "notes",
        "data_source",
    }
)
MEDIUM_COLS = frozenset(
    {
        "barrier_level",
        "data_type",
        "access_method",
        "severity",
        "date_accessed",
        "doi",
        "readiness_class",
        "license_type",
        "barrier",
        "integration_class",
    }
)
BARRIER_SEVERITY_COLS = frozenset(
    {
        "technical_barrier",
        "legal_barrier",
        "organisational_barrier",
        "governance_barrier",
        "semantic_barrier",
        "accessibility_barrier",
    }
)

COLUMN_LABELS: dict[str, str] = {
    "system_id": "System",
    "system_name": "Name",
    "total_score": "Total Score",
    "technical_subscore": "Technical (0-10)",
    "governance_subscore": "Governance (0-10)",
    "readiness_class": "Readiness",
    "barrier_level": "Overall",
    "barrier_count": "Barrier Count",
    "barrier_summary": "Summary",
    "technical_barrier": "Technical",
    "semantic_barrier": "Semantic",
    "legal_barrier": "Legal",
    "governance_barrier": "Governance",
    "accessibility_barrier": "Accessibility",
    "organisational_barrier": "Organisational",
    "integration_class": "Classification",
    "rank": "Rank",
    "priority": "Priority",
    "criterion": "Criterion",
    "recommendation": "Recommendation",
    "barrier": "Barrier",
    "severity": "Severity",
    "reason": "Reason",
    "official_website": "Official Site",
    "research_publication": "Publication",
    "data_source": "Source",
    "date_accessed": "Accessed",
    "url": "Evidence Source",
    "doi": "DOI",
    "publisher": "Publisher",
}


def columns_from_ids(col_ids: list[str], markdown_cols: frozenset[str] = frozenset()) -> list[dict]:
    columns = []
    for col_id in col_ids:
        col = {"name": COLUMN_LABELS.get(col_id, col_id.replace("_", " ").title()), "id": col_id}
        if col_id in markdown_cols:
            col["presentation"] = "markdown"
        columns.append(col)
    return columns


def _column_width_styles(column_ids: list[str]) -> list[dict]:
    styles = []
    for col_id in column_ids:
        if col_id in ID_COLS:
            styles.append({"if": {"column_id": col_id}, "width": "90px", "minWidth": "80px", "maxWidth": "100px"})
        elif col_id in NUMERIC_COLS:
            styles.append(
                {
                    "if": {"column_id": col_id},
                    "width": "110px",
                    "minWidth": "90px",
                    "maxWidth": "120px",
                    "textAlign": "right",
                }
            )
        elif col_id in NAME_COLS:
            styles.append({"if": {"column_id": col_id}, "width": "300px", "minWidth": "180px", "maxWidth": "320px"})
        elif col_id in TEXT_COLS or "summary" in col_id or "recommendation" in col_id:
            styles.append(
                {
                    "if": {"column_id": col_id},
                    "width": "380px",
                    "minWidth": "240px",
                    "maxWidth": "420px",
                    "whiteSpace": "normal",
                }
            )
        elif col_id in BARRIER_SEVERITY_COLS:
            styles.append({"if": {"column_id": col_id}, "width": "90px", "minWidth": "80px", "maxWidth": "100px"})
        elif col_id in MEDIUM_COLS:
            styles.append({"if": {"column_id": col_id}, "width": "140px", "minWidth": "110px", "maxWidth": "160px"})
    return styles


def make_table(
    data: list[dict],
    columns: list[dict] | None = None,
    *,
    extra_style_data_conditional: list[dict] | None = None,
    extra_style_cell_conditional: list[dict] | None = None,
    wrapper_style: dict | None = None,
    **kwargs,
) -> html.Div:
    if columns is None and data:
        columns = columns_from_ids(list(data[0]))
    columns = columns or []
    col_ids = [c["id"] for c in columns]

    style_data_conditional = list(ZEBRA_STRIPE)
    if "readiness_class" in col_ids:
        style_data_conditional.extend(READINESS_CELL_STYLES)
    if "integration_class" in col_ids:
        style_data_conditional.extend(INTEGRATION_CELL_STYLES)
    if "barrier_level" in col_ids:
        style_data_conditional.extend(BARRIER_LEVEL_CELL_STYLES)
    if extra_style_data_conditional:
        style_data_conditional.extend(extra_style_data_conditional)

    cell_styles = _column_width_styles(col_ids)
    if extra_style_cell_conditional:
        cell_styles.extend(extra_style_cell_conditional)

    if any(c.get("presentation") == "markdown" for c in columns):
        kwargs.setdefault("markdown_options", {"link_target": "_blank"})

    # Handle custom table style from wrapper_style if overflowX is specified
    table_style = dict(TABLE_STYLE)
    if wrapper_style and wrapper_style.get("overflowX") == "visible":
        table_style.update({
            "width": "100%",
            "maxWidth": "100%", 
            "overflowX": "visible",
        })
    
    table = dash_table.DataTable(
        data=data,
        columns=columns,
        style_cell=TABLE_CELL,
        style_header=TABLE_HEADER,
        style_cell_conditional=cell_styles,
        style_data_conditional=style_data_conditional,
        style_as_list_view=True,
        style_table=table_style,
        **kwargs,
    )

    wrap = {**TABLE_WRAPPER, **(wrapper_style or {})}
    return html.Div(style=wrap, children=[table])
