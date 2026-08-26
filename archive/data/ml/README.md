# Ecological suitability modelling inputs

The ML module does **not** ship training data. Place your own CSV files in
`data/ml/input/` and run the training pipeline.

## Training CSV (single file)

Required columns:

| Column | Description |
|--------|-------------|
| `latitude` | WGS84 latitude (`lat` accepted) |
| `longitude` | WGS84 longitude (`lon` accepted) |
| `tick_presence` | `1` = tick presence, `0` = background / pseudo-absence |
| `temperature` | Mean or sampled temperature |
| `rainfall` | Precipitation |
| `ndvi` | Normalised Difference Vegetation Index |
| `land_cover` | Land-cover category (optional if not used) |

Example:

```bash
python -m lib.ml.train_model --training-csv data/ml/input/training.csv
```

## Prediction grid (optional)

Unlabelled rows with the same feature columns (no `label` column) for mapping:

```bash
python -m lib.ml.train_model \
  --training-csv data/ml/input/training.csv \
  --prediction-csv data/ml/input/prediction_grid.csv
```

If omitted, suitability scores are generated for the training locations.

## Outputs (`data/ml/artifacts/`)

| File | Description |
|------|-------------|
| `model.joblib` | Trained sklearn pipeline |
| `metrics.json` | Test + 5-fold CV metrics |
| `feature_importance.csv` | Random Forest importances |
| `predictions.csv` | `latitude`, `longitude`, `probability`, `predicted_class` |
| `confusion_matrix.csv` | Test-set confusion matrix |
| `roc_curve.csv` | ROC curve points |
| `training_features.csv` | Cleaned training rows for dashboard plots |
| `metadata.json` | Feature list and run configuration |

The **Ecological Suitability** dashboard tab loads these files automatically.
