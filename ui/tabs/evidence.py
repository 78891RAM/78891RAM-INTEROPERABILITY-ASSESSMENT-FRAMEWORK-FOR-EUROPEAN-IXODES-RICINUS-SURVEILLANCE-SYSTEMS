"""Research evidence tab layout."""

from __future__ import annotations

from dash import html

from config import EVIDENCE_COLUMNS, EVIDENCE_UNAVAILABLE
from data.pipeline import FrameworkSnapshot
from ui.styles import MUTED
from ui.tables import columns_from_ids, make_table


def layout(snapshot: FrameworkSnapshot) -> html.Div:
    ev = snapshot.evidence
    if ev.empty:
        return html.P("No evidence records.")

    cols = [c for c in EVIDENCE_COLUMNS if c in ev.columns]

    missing = (
        int((ev.get("url", ev.get("official_website", "")) == EVIDENCE_UNAVAILABLE).sum())
        if "url" in ev.columns
        else 0
    )

    note = html.P(
        [
            f"Evidence traceability: {len(ev) - missing}/{len(ev)} systems have a documented URL. "
            "Test-retest reliability is scoring consistency — not expert validation.",
            html.Br(),
            "\"Official Site\" is the system's homepage; \"Evidence Source\" is the specific page "
            "the citation in this row was drawn from — they differ where the strongest evidence "
            "wasn't on the homepage (e.g. a dataset page or a news article).",
            html.Br(),
            "Evidence table adapts to your screen size: full view on desktop, key columns on mobile/tablet.",
        ],
        style={**MUTED, "marginBottom": "16px"},
    )

    link_cols = {"url", "official_website"} & set(cols)
    records = ev[cols].to_dict("records")
    for row in records:
        for col in link_cols:
            value = row.get(col, "")
            if value and value != EVIDENCE_UNAVAILABLE:
                row[col] = f"[{value}]({value})"

    # Responsive table with progressive disclosure
    table = html.Div([
        # Desktop/Large screens: Show all columns
        html.Div(
            make_table(
                data=records,
                columns=columns_from_ids(cols, markdown_cols=frozenset(link_cols)),
                page_size=15,
                wrapper_style={
                    "maxWidth": "1150px",
                    "overflowX": "auto",  # Allow scroll on desktop for wide content
                    "width": "100%",
                },
            ),
            className="evidence-table-desktop",
            style={
                "display": "block",
                "@media screen and (max-width: 1024px)": {"display": "none"},
            },
        ),
        
        # Tablet/Medium screens: Priority columns only
        html.Div([
            html.P(
                "📱 Simplified view for smaller screens. Key columns shown below:",
                style={**MUTED, "fontSize": "0.85rem", "marginBottom": "12px", "fontStyle": "italic"}
            ),
            make_table(
                data=records,
                columns=columns_from_ids(
                    # Priority columns for mobile: ID, official site, and evidence source
                    [col for col in ["system_id", "official_website", "research_publication", "doi", "url"] if col in cols],
                    markdown_cols=frozenset(link_cols)
                ),
                page_size=15,
                wrapper_style={
                    "width": "100%",
                    "overflowX": "visible",
                },
            ),
        ],
        className="evidence-table-mobile",
        style={"display": "none"},
        ),
    ])

    return html.Div([note, table])
