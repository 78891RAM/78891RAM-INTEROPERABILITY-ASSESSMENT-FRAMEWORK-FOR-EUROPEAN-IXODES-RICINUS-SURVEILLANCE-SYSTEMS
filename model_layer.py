"""
Field-validated tick presence model — loads a pre-trained classifier and
scores it against the 34-site field survey (field_clean.csv). Never retrains
at app startup; only trains a fallback model if no saved model file exists.

Data + model files live in data/field_model/:
  - model.pkl              saved sklearn Pipeline (preprocessing + classifier)
  - field_clean.csv        REQUIRED — the only file with real presence/absence
  - occurrence_layer.csv   optional — display-only occurrence backdrop
  - environment_layer.csv  optional — display-only environmental overlay
  - dashboard_cells.csv    optional — display-only occurrence density grid

Any missing file degrades gracefully: ModelLayerData.error explains what's
missing so the UI can show a message and skip the layer instead of crashing.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent
FIELD_MODEL_DIR = PROJECT_ROOT / "data" / "field_model"

MODEL_PATH = FIELD_MODEL_DIR / "model.pkl"
MODEL_METADATA_PATH = FIELD_MODEL_DIR / "model_metadata.json"
FIELD_CSV = FIELD_MODEL_DIR / "field_clean.csv"
OCCURRENCE_CSV = FIELD_MODEL_DIR / "occurrence_layer.csv"
ENVIRONMENT_CSV = FIELD_MODEL_DIR / "environment_layer.csv"
CELLS_CSV = FIELD_MODEL_DIR / "dashboard_cells.csv"

# Set this to the site-grouped (GroupKFold-by-site) AUC from your original
# training run, e.g. SITE_GROUPED_AUC = 0.83, to display it without
# recomputing anything. Left as None until you provide it.
SITE_GROUPED_AUC: float | None = None

# Canonical feature key -> lowercase substring used to resolve the real column
# name in field_clean.csv (headers contain odd characters like '°)').
FEATURE_FRAGMENTS: dict[str, str] = {
    "humidity": "rh at sample point",
    "temperature_sample_point": "temperature at sample point",
    "wind_speed": "wind speed",
    "elevation": "elevation",
    "temperature_max": "temperature max",
    "max_rainfall": "max rainfall",
    "ndvi": "ndvi",
    "urban_fabric": "discontinuous urban fabric",
    "month": "month",
    "land_use": "land use",
}
CATEGORICAL_FEATURE_KEYS = {"land_use"}

# Fields that must never end up in the predictor matrix (identifiers or the
# label itself). Checked with an assertion after column resolution.
BANNED_FRAGMENTS = ["site", "count_ticks", "present", "vegetation sample point", "life stage"]


@dataclass
class ModelLayerData:
    """Everything the dashboard needs to render the field tick-model section."""

    field: pd.DataFrame
    occurrence: pd.DataFrame | None
    environment: pd.DataFrame | None
    cells: pd.DataFrame | None
    auc: float | None
    auc_is_insample: bool
    n_sites: int
    n_points: int
    model_source: str  # "loaded" | "trained_fallback" | "unavailable"
    feature_columns: list[str]
    error: str | None = None
    warning: str | None = None

    @property
    def available(self) -> bool:
        return self.error is None and "pred_prob" in self.field.columns


def _resolve_feature_columns(columns: list[str]) -> dict[str, str]:
    """Map canonical feature keys -> actual dataframe column names by fragment match."""
    lower_to_actual = {c.lower(): c for c in columns}
    resolved: dict[str, str] = {}
    for key, fragment in FEATURE_FRAGMENTS.items():
        match = next((actual for low, actual in lower_to_actual.items() if fragment in low), None)
        if match:
            resolved[key] = match
    return resolved


def _assert_no_banned_fields(resolved_columns: list[str]) -> None:
    lowered = [c.lower() for c in resolved_columns]
    for banned in BANNED_FRAGMENTS:
        offenders = [c for c in lowered if banned in c]
        assert not offenders, (
            f"Banned field fragment '{banned}' found in predictor columns {offenders} — "
            "this leaks the label or is a non-predictive identifier and must be excluded."
        )


def _try_load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _load_saved_model(path: Path):
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        pass
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def _build_fallback_preprocessor(numeric_cols: list[str], categorical_cols: list[str]):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))
    return ColumnTransformer(transformers)


def _split_feature_keys(resolved: dict[str, str]) -> tuple[list[str], list[str]]:
    numeric_cols = [v for k, v in resolved.items() if k not in CATEGORICAL_FEATURE_KEYS]
    categorical_cols = [v for k, v in resolved.items() if k in CATEGORICAL_FEATURE_KEYS]
    return numeric_cols, categorical_cols


def _train_fallback_model(field_df: pd.DataFrame, resolved: dict[str, str]) -> tuple[Pipeline, float | None]:
    """Only runs when MODEL_PATH is missing. Trains + saves, validated with GroupKFold by site."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict

    numeric_cols, categorical_cols = _split_feature_keys(resolved)
    X = field_df[numeric_cols + categorical_cols]
    y = field_df["present"].astype(int)
    groups = field_df["site"]

    pipeline = Pipeline([
        ("prep", _build_fallback_preprocessor(numeric_cols, categorical_cols)),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=0)),
    ])

    n_groups = groups.nunique()
    auc: float | None = None
    if n_groups >= 2 and y.nunique() >= 2:
        cv = GroupKFold(n_splits=min(5, n_groups))
        oof_proba = cross_val_predict(
            pipeline, X, y, groups=groups, cv=cv, method="predict_proba",
        )[:, 1]
        try:
            auc = float(roc_auc_score(y, oof_proba))
        except ValueError:
            auc = None

    pipeline.fit(X, y)
    FIELD_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    _write_model_metadata(auc=auc, n_sites=n_groups, cv_folds=min(5, n_groups) if n_groups >= 2 else 0)
    return pipeline, auc


def _write_model_metadata(*, auc: float | None, n_sites: int, cv_folds: int) -> None:
    """
    Persist the GroupKFold-by-site AUC computed at training time. Without this,
    a later run that only *loads* the saved model has no honest generalization
    estimate left — scoring the loaded model against the rows it was fit on
    gives a leaky, near-perfect in-sample number instead (see the fallback in
    load_model_layer_data), which is what motivated saving this sidecar.
    """
    import json
    from datetime import datetime, timezone

    payload = {
        "site_grouped_auc": auc,
        "n_sites": n_sites,
        "cv_folds": cv_folds,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    MODEL_METADATA_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_model_metadata() -> dict | None:
    if not MODEL_METADATA_PATH.exists():
        return None
    import json

    try:
        return json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _error_result(
    field_df: pd.DataFrame,
    occurrence_df: pd.DataFrame | None,
    environment_df: pd.DataFrame | None,
    cells_df: pd.DataFrame | None,
    model_source: str,
    feature_columns: list[str],
    error: str,
) -> ModelLayerData:
    return ModelLayerData(
        field=field_df,
        occurrence=occurrence_df,
        environment=environment_df,
        cells=cells_df,
        auc=None,
        auc_is_insample=False,
        n_sites=int(field_df["site"].nunique()) if "site" in field_df.columns else 0,
        n_points=len(field_df),
        model_source=model_source,
        feature_columns=feature_columns,
        error=error,
    )


def load_model_layer_data() -> ModelLayerData:
    """
    Load the saved model + field/display data and return field predictions
    plus metadata. Never raises — any failure is captured in `.error` so the
    dashboard can show a message and skip the layer instead of crashing.
    """
    field_df = _try_load_csv(FIELD_CSV)
    if field_df is None:
        return _error_result(
            pd.DataFrame(), None, None, None, "unavailable", [],
            f"{FIELD_CSV.name} not found — place it in {FIELD_MODEL_DIR}",
        )

    occurrence_df = _try_load_csv(OCCURRENCE_CSV)
    environment_df = _try_load_csv(ENVIRONMENT_CSV)
    cells_df = _try_load_csv(CELLS_CSV)

    resolved = _resolve_feature_columns(list(field_df.columns))
    missing = [key for key in FEATURE_FRAGMENTS if key not in resolved]
    if missing or "present" not in field_df.columns or "site" not in field_df.columns:
        detail = f"missing predictor columns for: {', '.join(missing)}" if missing else "missing 'present' or 'site' column"
        return _error_result(
            field_df, occurrence_df, environment_df, cells_df, "unavailable", [],
            f"{FIELD_CSV.name} is not in the expected shape — {detail}",
        )

    feature_columns = list(resolved.values())
    _assert_no_banned_fields(feature_columns)
    numeric_cols, categorical_cols = _split_feature_keys(resolved)
    X = field_df[numeric_cols + categorical_cols]

    model = _load_saved_model(MODEL_PATH)
    model_source = "loaded"
    auc = SITE_GROUPED_AUC
    warning = None

    if model is not None and auc is None:
        metadata = _read_model_metadata()
        if metadata is not None:
            auc = metadata.get("site_grouped_auc")

    if model is None:
        try:
            model, trained_auc = _train_fallback_model(field_df, resolved)
            model_source = "trained_fallback"
            if auc is None:
                auc = trained_auc
        except Exception as exc:
            return _error_result(
                field_df, occurrence_df, environment_df, cells_df, "unavailable", feature_columns,
                f"No saved model at {MODEL_PATH.name} and fallback training failed: {exc}",
            )

    try:
        if isinstance(model, Pipeline):
            proba = model.predict_proba(X)[:, 1]
        else:
            preprocessor = _build_fallback_preprocessor(numeric_cols, categorical_cols)
            X_transformed = preprocessor.fit_transform(X)
            proba = model.predict_proba(X_transformed)[:, 1]
            warning = (
                "Loaded model is not an sklearn Pipeline — preprocessing was rebuilt here and "
                "may not exactly match training. Verify predictions before trusting them."
            )
    except Exception as exc:
        return _error_result(
            field_df, occurrence_df, environment_df, cells_df, model_source, feature_columns,
            f"Model loaded but prediction failed: {exc}",
        )

    field_df = field_df.copy()
    field_df["pred_prob"] = proba

    auc_is_insample = False
    if auc is None:
        from sklearn.metrics import roc_auc_score

        try:
            auc = float(roc_auc_score(field_df["present"].astype(int), field_df["pred_prob"]))
            auc_is_insample = True
            warning = (warning + " " if warning else "") + (
                "AUC shown is in-sample (scored on the same rows the model already saw), "
                "not a held-out/site-grouped estimate — set SITE_GROUPED_AUC in model_layer.py "
                "once you have the real cross-validated figure."
            )
        except ValueError:
            auc = None

    return ModelLayerData(
        field=field_df,
        occurrence=occurrence_df,
        environment=environment_df,
        cells=cells_df,
        auc=auc,
        auc_is_insample=auc_is_insample,
        n_sites=int(field_df["site"].nunique()),
        n_points=len(field_df),
        model_source=model_source,
        feature_columns=feature_columns,
        warning=warning,
    )
