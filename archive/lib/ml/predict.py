"""Suitability prediction functions for external grid CSV files."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from lib.ml.config import ColumnConfig
from lib.ml.data_loader import load_model, load_prediction_csv, save_dataframe
from lib.ml.feature_engineering import FeatureSpec

def predict_suitability(model, df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Return coordinates, 0–1 probability and predicted class."""
    output = df[["latitude", "longitude"]].copy()
    output["probability"] = model.predict_proba(df[spec.all_features])[:, 1]
    output["suitability"] = output["probability"]  # Compatibility alias for early dashboard exports.
    output["predicted_class"] = model.predict(df[spec.all_features]).astype(int)
    return output

def predict_from_csv(model_path: Path | str, prediction_csv: Path | str, output_csv: Path | str,
                     *, feature_spec: FeatureSpec, columns: ColumnConfig = ColumnConfig()) -> pd.DataFrame:
    """Predict on a user grid and save its continuous suitability probabilities."""
    grid = load_prediction_csv(prediction_csv, columns=columns, feature_columns=feature_spec.all_features)
    predictions = predict_suitability(load_model(Path(model_path)), grid, feature_spec)
    save_dataframe(Path(output_csv), predictions)
    return predictions
