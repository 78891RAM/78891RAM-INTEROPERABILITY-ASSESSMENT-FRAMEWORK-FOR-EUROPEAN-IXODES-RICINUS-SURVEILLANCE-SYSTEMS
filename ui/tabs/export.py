"""Export tab layout."""

from __future__ import annotations

import base64
from io import BytesIO

import pandas as pd
from dash import html

from data.pipeline import FrameworkSnapshot
from ui.styles import BLOCK, MUTED


def layout(snapshot: FrameworkSnapshot) -> html.Div:
    if not snapshot.ok:
        return html.P("Nothing to export.")

    downloads = []
    sheets = {
        "scores": snapshot.systems,
        "barriers": snapshot.barriers,
        "barrier_details": snapshot.barrier_details,
        "integration": snapshot.integration,
        "recommendations": snapshot.recommendations,
        "evidence": snapshot.evidence,
    }

    for name, df in sheets.items():
        if df is None:
            continue
        if df.empty:
            continue
        downloads.append(
            _csv_download(df, f"{name}.csv", f"Download {name.replace('_', ' ').title()}")
        )

    xlsx = _excel_download(sheets, "interoperability_export.xlsx")
    if xlsx:
        downloads.insert(1, xlsx)

    return html.Div([
        html.P(
            "Download analytical layers as CSV or a combined Excel workbook.",
            style={**MUTED, "marginBottom": "16px"},
        ),
        html.Div(downloads, style=BLOCK),
    ])


def _csv_download(df: pd.DataFrame, filename: str, label: str) -> html.A:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return html.A(
        label,
        href=f"data:text/csv;base64,{b64}",
        download=filename,
        style={"display": "block", "margin": "8px 0", "color": "#2980b9"},
    )


def _excel_download(sheets: dict[str, pd.DataFrame], filename: str) -> html.A | None:
    try:
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            for name, df in sheets.items():
                if df is None:
                    continue
                if df.empty:
                    continue
                df.to_excel(writer, sheet_name=name[:31], index=False)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return html.A(
            "Download all (Excel)",
            href=f"data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}",
            download=filename,
            style={"display": "block", "margin": "8px 0", "fontWeight": "600", "color": "#1a5276"},
        )
    except Exception:
        return None
