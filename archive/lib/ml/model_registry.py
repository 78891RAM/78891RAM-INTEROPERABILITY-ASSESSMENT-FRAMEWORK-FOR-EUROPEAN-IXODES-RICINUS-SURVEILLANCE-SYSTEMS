"""Create metadata records for reproducible ML training runs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lib.ml.config import MODEL_NAME, RANDOM_STATE
from lib.ml.data_loader import save_json


def build_metadata(
    *, dataset_name: str, training_size: int, testing_size: int,
    hyperparameters: dict[str, Any], metrics: dict[str, Any],
    feature_columns: list[str], random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Return a serialisable model-registry entry."""
    return {
        "model_name": MODEL_NAME,
        "version": "1.0",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "training_sample_size": training_size,
        "testing_sample_size": testing_size,
        "random_seed": random_state,
        "feature_columns": feature_columns,
        "hyperparameters": hyperparameters,
        "evaluation_metrics": metrics,
    }


def save_metadata(path: Path, metadata: dict[str, Any]) -> None:
    """Write the model-registry metadata JSON."""
    save_json(path, metadata)
