"""Load validated user-supplied ecological data and ML artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd

from lib.ml.config import ARTIFACT_PATH, INPUT_PATH, ColumnConfig

ML_OUTPUT_DIR = ARTIFACT_PATH  # Backwards-compatible public alias.
ML_INPUT_DIR = INPUT_PATH
MODEL_FILENAME = "model.joblib"
METRICS_FILENAME = "metrics.json"
FEATURE_IMPORTANCE_FILENAME = "feature_importance.csv"
PREDICTIONS_FILENAME = "predictions.csv"
CONFUSION_MATRIX_FILENAME = "confusion_matrix.csv"
ROC_CURVE_FILENAME = "roc_curve.csv"
TRAINING_FEATURES_FILENAME = "training_features.csv"
METADATA_FILENAME = "metadata.json"
CLASSIFICATION_REPORT_FILENAME = "classification_report.json"
LABEL_COLUMN = "tick_presence"
VARIABLE_STATS_FILENAME = "variable_stats.csv"


@dataclass(frozen=True)
class MLArtifacts:
    """Outputs loaded by the dashboard; no connection to assessment data."""
    metrics: dict
    feature_importance: pd.DataFrame
    predictions: pd.DataFrame
    confusion_matrix: pd.DataFrame
    roc_curve: pd.DataFrame
    variable_stats: pd.DataFrame
    training_features: pd.DataFrame
    metadata: dict
    model_path: Path


def _required(columns: Iterable[str], frame: pd.DataFrame) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_ecological_frame(df: pd.DataFrame, *, columns: ColumnConfig = ColumnConfig(),
                           require_target: bool = True,
                           feature_columns: list[str] | None = None,
                           require_label: bool | None = None) -> pd.DataFrame:
    """Validate, clean and return canonical geographic ecological data.

    Numeric missing values are retained for pipeline median imputation; missing
    coordinates, target values and categorical values are removed.
    """
    if require_label is not None:  # Compatibility with the initial module API.
        require_target = require_label
    if df.empty:
        raise ValueError("Input dataset is empty.")
    # Common aliases retain compatibility with established surveillance extracts;
    # custom providers should use ColumnConfig for fully explicit mapping.
    aliases = {"lat": columns.latitude, "lon": columns.longitude, "label": columns.target,
               "land_use": "land_cover"}
    work_source = df.rename(columns={old: new for old, new in aliases.items()
                                     if old in df.columns and new not in df.columns})
    features = ([aliases.get(column, column) for column in feature_columns]
                if feature_columns else [column for column in columns.feature_columns if column in work_source.columns])
    if not features:
        raise ValueError(
            "No configured feature columns were supplied. Configure ColumnConfig to match "
            f"the CSV headers. Available columns: {list(work_source.columns)}"
        )
    _required([columns.latitude, columns.longitude, *features], work_source)
    if require_target:
        _required([columns.target], work_source)
    rename = {columns.latitude: "latitude", columns.longitude: "longitude"}
    if require_target:
        rename[columns.target] = LABEL_COLUMN
    work = work_source.rename(columns=rename).copy()
    work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
    work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce")
    numeric = [c for c in features if c in columns.numeric_features]
    categorical = [c for c in features if c in columns.categorical_features]
    for column in numeric:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    for column in categorical:
        work[column] = work[column].astype("string").str.strip().replace("", pd.NA)
    required = ["latitude", "longitude", *categorical]
    if require_target:
        work[LABEL_COLUMN] = pd.to_numeric(work[LABEL_COLUMN], errors="coerce")
        required.append(LABEL_COLUMN)
    work = work.dropna(subset=required)
    if require_target:
        work[LABEL_COLUMN] = work[LABEL_COLUMN].astype(int)
        if not set(work[LABEL_COLUMN].unique()).issubset({0, 1}):
            raise ValueError(f"'{columns.target}' must contain only 0 and 1.")
    if work.empty:
        raise ValueError("No valid rows remain after cleaning.")
    return work[["latitude", "longitude", *features, *([LABEL_COLUMN] if require_target else [])]]


def load_training_csv(path: Path | str, *, columns: ColumnConfig = ColumnConfig(),
                      feature_columns: list[str] | None = None) -> pd.DataFrame:
    """Load a labelled training dataset from CSV."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {file_path}")
    return clean_ecological_frame(pd.read_csv(file_path), columns=columns,
                                  feature_columns=feature_columns)


def load_prediction_csv(path: Path | str, *, columns: ColumnConfig = ColumnConfig(),
                        feature_columns: list[str] | None = None) -> pd.DataFrame:
    """Load an unlabelled prediction grid from CSV."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {file_path}")
    return clean_ecological_frame(pd.read_csv(file_path), columns=columns,
                                  feature_columns=feature_columns, require_target=False)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def save_dataframe(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def artifacts_available(output_dir: Path | None = None) -> bool:
    """Return whether a complete minimum set of dashboard artifacts exists."""
    root = output_dir or ARTIFACT_PATH
    return all((root / name).exists() for name in (METRICS_FILENAME, FEATURE_IMPORTANCE_FILENAME,
                                                    PREDICTIONS_FILENAME, METADATA_FILENAME))


def load_artifacts(output_dir: Path | None = None) -> MLArtifacts | None:
    """Load saved artifacts, returning ``None`` when no trained model is available."""
    root = output_dir or ARTIFACT_PATH
    if not artifacts_available(root):
        return None
    def csv(name: str) -> pd.DataFrame:
        path = root / name
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    return MLArtifacts(json.loads((root / METRICS_FILENAME).read_text()), csv(FEATURE_IMPORTANCE_FILENAME),
        csv(PREDICTIONS_FILENAME), csv(CONFUSION_MATRIX_FILENAME), csv(ROC_CURVE_FILENAME),
        csv(VARIABLE_STATS_FILENAME), csv(TRAINING_FEATURES_FILENAME),
        json.loads((root / METADATA_FILENAME).read_text()), root / MODEL_FILENAME)


def load_model(model_path: Path | None = None):
    """Load a serialised sklearn pipeline."""
    path = model_path or ARTIFACT_PATH / MODEL_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)
