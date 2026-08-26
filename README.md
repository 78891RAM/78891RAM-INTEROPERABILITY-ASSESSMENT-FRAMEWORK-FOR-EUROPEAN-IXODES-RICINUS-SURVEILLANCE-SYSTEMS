# Interoperability Assessment Framework for European *Ixodes ricinus* Surveillance Systems

An interactive Dash dashboard supporting the MSc dissertation *"Development of an
Interoperability Assessment Framework for European Ixodes ricinus Surveillance
Systems."* It scores and compares 14 European tick surveillance systems on a
10-criterion interoperability scorecard, classifies technical/legal/semantic/
governance/accessibility barriers, derives a barrier- and hard-gate-adjusted
integration readiness verdict, and presents an ecological suitability model
(trained on Austria and Croatia, externally validated on Estonia and Ireland)
alongside its cross-system transfer and validation results.

## Dashboard tabs

| Tab | Content |
|---|---|
| Overview | Headline KPIs, readiness distribution, score distribution |
| Map | Interoperability readiness by country, across the 14 assessed systems |
| Ecological Suitability | Predicted suitability surface, occurrence data, model performance, transfer matrix, external validation, feature importance |
| Scores | Per-criterion scores, system ranking, criterion × system heatmap |
| Barriers | Technical/semantic/legal/governance/accessibility barrier severity |
| Integration | Barrier-adjusted integration readiness — the headline classification |
| Recommendations | Rule-based recommendations per system per weak criterion |
| Evidence | Source traceability for every system's scorecard entry |
| Methodology | Scoring methodology and criteria definitions |
| Export | CSV / Excel export of every analytical layer |

## Running locally

```bash
python3.13 -m venv venv
venv/bin/python -m pip install -r requirements.txt
venv/bin/python app.py            # dev server, http://127.0.0.1:8050
# or
venv/bin/python -m gunicorn -w 1 -b 0.0.0.0:8051 app:server   # production-style
```

Note: use `venv/bin/python -m <tool>` (not the bare `pip`/`gunicorn` commands)
if you ever recreate the venv from a copied/moved folder — a wrapper script's
shebang bakes in an absolute path at install time and breaks if the folder is
later moved or copied.

## Generating dissertation figures

```bash
venv/bin/python export_dissertation_figures.py
```

Exports every dashboard chart and table as publication-resolution PNG/SVG to
`../dissertation_figures/` (see `export_dissertation_figures.py`'s docstring).

## Deploying (Render)

This repo is self-contained — no files outside `tick/` are needed at runtime.
`render.yaml` defines the service:

- Build: `pip install -r requirements.txt`
- Start: `gunicorn app:server`

Connect the repo in the Render dashboard (New → Blueprint, or a Web Service
pointing at this repo) and it picks up `render.yaml` automatically.

## Data

- `data/systems.csv`, `data/evidence.csv` — the 14-system scorecard and its
  source evidence (author-compiled, 2026).
- `outputs/` — ecological suitability model outputs actually used by the
  dashboard (suitability grid, occurrence layer, model performance, transfer
  matrix, external validation, feature importance). This is a small subset of
  the full modelling pipeline's outputs — only what the Ecological
  Suitability tab reads.

## Tech stack

Dash / Plotly, pandas, scikit-learn (model training happens offline, in the
notebook pipeline — the dashboard only reads its saved outputs), Shapely
(country-boundary filtering for the suitability map), Kaleido (server-side
PNG export for the "Download figure" buttons and the dissertation export
script).
