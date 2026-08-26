"""Tests for the standalone ecological suitability ML module."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from lib.ml.data_loader import artifacts_available, clean_ecological_frame, load_artifacts
from lib.ml.train_model import TrainingConfig, run_training_pipeline


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ml"


@pytest.fixture()
def tiny_training_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame({
        "latitude": [48.0, 48.1, 47.9, 48.05, 48.15, 47.85, 52.0, 52.1, 51.9, 52.05, 52.15, 51.85],
        "longitude": [2.0, 2.1, 1.9, 2.05, 2.15, 1.85, 5.0, 5.1, 4.9, 5.05, 5.15, 4.85],
        "temperature": [15.0, 16.0, 14.5, 15.2, 15.8, 14.8, 10.0, 11.0, 9.5, 10.2, 10.8, 9.8],
        "rainfall": [600, 620, 580, 610, 630, 590, 900, 880, 910, 895, 870, 920],
        "ndvi": [0.7, 0.72, 0.68, 0.71, 0.73, 0.69, 0.4, 0.42, 0.38, 0.41, 0.43, 0.37],
        "land_use": ["forest"] * 6 + ["urban"] * 6,
        "label": [1] * 6 + [0] * 6,
    })
    path = tmp_path / "training.csv"
    df.to_csv(path, index=False)
    return path


def test_clean_ecological_frame_drops_incomplete_rows() -> None:
    df = pd.DataFrame({
        "lat": [48.0, None],
        "lon": [2.0, 2.1],
        "temperature": [15.0, 16.0],
        "rainfall": [600, 620],
        "ndvi": [0.7, 0.72],
        "land_use": ["forest", "forest"],
        "label": [1, 0],
    })
    cleaned = clean_ecological_frame(
        df,
        feature_columns=["temperature", "rainfall", "ndvi", "land_use"],
    )
    assert len(cleaned) == 1
    assert "latitude" in cleaned.columns


def test_training_pipeline_writes_artifacts(tiny_training_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "ml_out"
    config = TrainingConfig(
        training_csv=tiny_training_csv,
        output_dir=out,
        cv_folds=5,
        n_estimators=50,
        random_state=0,
    )
    result = run_training_pipeline(config)

    assert result.model_path.exists()
    assert result.metrics_path.exists()
    assert result.feature_importance_path.exists()
    assert result.predictions_path.exists()

    payload = json.loads(result.metrics_path.read_text())
    assert "test" in payload
    assert "cross_validation" in payload
    assert payload["cross_validation"]["folds"] >= 2
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert key in payload["test"]

    preds = pd.read_csv(result.predictions_path)
    assert {"latitude", "longitude", "probability", "predicted_class"}.issubset(preds.columns)
    assert preds["probability"].between(0, 1).all()


def test_load_artifacts_for_dashboard(tiny_training_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "ml_out"
    run_training_pipeline(TrainingConfig(
        training_csv=tiny_training_csv,
        output_dir=out,
        cv_folds=5,
        n_estimators=50,
    ))
    assert artifacts_available(out)
    bundle = load_artifacts(out)
    assert bundle is not None
    assert not bundle.feature_importance.empty
    assert not bundle.predictions.empty
