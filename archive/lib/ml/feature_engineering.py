"""Feature preparation for ecological suitability modelling."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from lib.ml.config import ColumnConfig
from lib.ml.data_loader import LABEL_COLUMN


@dataclass(frozen=True)
class FeatureSpec:
    """Configured numeric and categorical model features."""
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    label_column: str = LABEL_COLUMN
    @property
    def all_features(self) -> list[str]:
        return [*self.numeric_features, *self.categorical_features]


def default_feature_spec(df: pd.DataFrame, *, columns: ColumnConfig = ColumnConfig()) -> FeatureSpec:
    """Derive the modelling specification from supplied column configuration."""
    numeric = tuple(c for c in columns.numeric_features if c in df.columns)
    categorical = tuple(c for c in columns.categorical_features if c in df.columns)
    if not numeric and not categorical:
        raise ValueError("No configured modelling features found in the dataset.")
    return FeatureSpec(numeric, categorical)


def split_features_labels(df: pd.DataFrame, spec: FeatureSpec) -> tuple[pd.DataFrame, pd.Series]:
    """Extract X and binary y from a cleaned training frame."""
    return df[spec.all_features].copy(), df[spec.label_column].astype(int)


def build_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    """Build imputation and one-hot encoding without leakage outside a pipeline."""
    transforms = []
    if spec.numeric_features:
        transforms.append(("numeric", SimpleImputer(strategy="median"), list(spec.numeric_features)))
    if spec.categorical_features:
        transforms.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), list(spec.categorical_features)))
    return ColumnTransformer(transforms)


def transformed_feature_names(preprocessor: ColumnTransformer, spec: FeatureSpec) -> list[str]:
    """Return post-preprocessing feature names."""
    return list(preprocessor.get_feature_names_out())
