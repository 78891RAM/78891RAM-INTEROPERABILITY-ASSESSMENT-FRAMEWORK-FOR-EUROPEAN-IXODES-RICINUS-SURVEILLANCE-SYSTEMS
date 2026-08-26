"""Central configuration for the independent ecological ML module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "ml" / "input"
ARTIFACT_PATH = PROJECT_ROOT / "data" / "ml" / "artifacts"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
N_ESTIMATORS = 300
MODEL_NAME = "RandomForest"
PARAM_GRID = {
    "classifier__max_depth": [None, 10],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2],
    "classifier__max_features": ["sqrt", None],
}


@dataclass(frozen=True)
class ColumnConfig:
    """Column names used by a supplied ecological dataset.

    Adjust this object when a data provider uses different field names.
    """

    latitude: str = "latitude"
    longitude: str = "longitude"
    target: str = "tick_presence"
    numeric_features: tuple[str, ...] = (
        "temperature", "humidity", "rainfall", "elevation", "ndvi", "month",
    )
    categorical_features: tuple[str, ...] = ("land_cover",)

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self.numeric_features + self.categorical_features
