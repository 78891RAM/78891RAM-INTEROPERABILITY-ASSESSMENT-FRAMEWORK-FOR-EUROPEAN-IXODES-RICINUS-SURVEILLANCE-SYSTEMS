"""Methodology tab layout — merged with the former Roadmap tab (see UI audit)."""

from __future__ import annotations

from dash import html

from ui.styles import MUTED, THEME_GREEN


def layout(_snapshot=None) -> html.Div:
    """Build dissertation methodology + repository roadmap workflow."""
    phases = [
        ("Literature Review", "Survey European tick surveillance infrastructures."),
        ("System Identification", "Select representative national and pan-European systems (VectorNet, CiTIQUE, national schemes, and others)."),
        ("Metadata Extraction", "Manual curation into data/systems.csv: coverage, access, licence, standards, governance."),
        ("Interoperability Scoring", "10-criterion scorecard (0–2 each), reported as a total plus technical and governance sub-scores — see Scores."),
        ("Scoring Consistency Check", "Test-retest comparison — not blind expert validation."),
        ("Barrier Classification & Hard Gate", "Score combined with barrier severity across five dimensions (technical, semantic, legal, governance, accessibility). A system scoring zero on a genuinely blocking criterion — starting with API availability — is capped below \"High integration ready\" by an explicit hard-gate rule, regardless of total score. This barrier-adjusted integration class, not the raw score, is the app's headline verdict — see Integration."),
        ("Standardisation Target", "Darwin Core, FAIR principles, REST APIs, and shared vocabularies — the interoperability benchmark the scorecard measures systems against."),
        ("Ecological Suitability Modelling", "A field-validated tick presence/absence model, independent from the interoperability scorecard, shown on its own Ecological Suitability dashboard tab."),
        ("Visual Analytics", "Interactive Dash dashboard: Overview, Map, Scores, Barriers, Integration, Ecological Suitability, Recommendations, Evidence, and Export."),
        ("Repository & Policy Recommendations", "Roadmap toward a harmonised European metadata repository — a central catalogue with provenance and access policies — plus evidence-based policy recommendations for EU tick surveillance interoperability."),
    ]

    return html.Div([
        html.Pre(
            "Literature → Identification → Extraction → Scoring → Consistency → "
            "Barriers & Hard Gate → Standardisation → Suitability Modelling → Analytics → Repository & Policy",
            style={"background": "#f4f6f7", "padding": "12px", "borderRadius": "6px", "marginBottom": "20px"},
        ),
        html.Ul([html.Li([html.B(title), f": {desc}"]) for title, desc in phases]),
        html.P(
            "Scores are derived from documented metadata via a manual extraction protocol; "
            "assessment is at the metadata level only.",
            style={
                **MUTED,
                "marginTop": "20px",
                "fontStyle": "italic",
                "borderLeft": f"3px solid {THEME_GREEN}",
                "paddingLeft": "12px",
            },
        ),
    ])
