"""Train and persist a tuned Random Forest ecological suitability model."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from lib.ml.config import ARTIFACT_PATH, CV_FOLDS, N_ESTIMATORS, PARAM_GRID, RANDOM_STATE, TEST_SIZE, ColumnConfig
from lib.ml.data_loader import (CLASSIFICATION_REPORT_FILENAME, CONFUSION_MATRIX_FILENAME,
    FEATURE_IMPORTANCE_FILENAME, METADATA_FILENAME, METRICS_FILENAME, MODEL_FILENAME,
    PREDICTIONS_FILENAME, ROC_CURVE_FILENAME, TRAINING_FEATURES_FILENAME, VARIABLE_STATS_FILENAME,
    LABEL_COLUMN, load_prediction_csv, load_training_csv, save_dataframe, save_json)
from lib.ml.evaluation import (confusion_matrix_frame, evaluate_classifier, evaluation_to_metrics_dict,
    feature_importance_frame, roc_curve_frame, run_cross_validation, variable_distribution_stats)
from lib.ml.feature_engineering import FeatureSpec, build_preprocessor, default_feature_spec, split_features_labels, transformed_feature_names
from lib.ml.model_registry import build_metadata, save_metadata
from lib.ml.predict import predict_suitability

@dataclass(frozen=True)
class TrainingConfig:
    """Inputs and all configurable model-training choices."""
    training_csv: Path
    prediction_csv: Path | None = None
    output_dir: Path = ARTIFACT_PATH
    columns: ColumnConfig = ColumnConfig()
    test_size: float = TEST_SIZE
    random_state: int = RANDOM_STATE
    cv_folds: int = CV_FOLDS
    n_estimators: int = N_ESTIMATORS

@dataclass(frozen=True)
class TrainingResult:
    """Important persisted files from a successful training run."""
    model_path: Path; metrics_path: Path; feature_importance_path: Path; predictions_path: Path; metadata_path: Path

def _pipeline(spec: FeatureSpec, config: TrainingConfig) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor(spec)), ("classifier", RandomForestClassifier(
        random_state=config.random_state, class_weight="balanced_subsample", n_jobs=1))])

def run_training_pipeline(config: TrainingConfig) -> TrainingResult:
    """Run validation, tuning, cross-validation, evaluation and artifact creation."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    train_df = load_training_csv(config.training_csv, columns=config.columns)
    spec = default_feature_spec(train_df, columns=config.columns)
    x, y = split_features_labels(train_df, spec)
    if y.nunique() != 2 or y.value_counts().min() < 3:
        raise ValueError("Training data needs at least three rows for each target class.")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=config.test_size,
        random_state=config.random_state, stratify=y)
    model = _pipeline(spec, config)
    folds = min(config.cv_folds, int(y_train.value_counts().min()))
    parameter_grid = {**PARAM_GRID, "classifier__n_estimators": [config.n_estimators]}
    tuning = GridSearchCV(model, parameter_grid,
        scoring="roc_auc", cv=folds, n_jobs=1, refit=True)
    tuning.fit(x_train, y_train)
    best_model = tuning.best_estimator_
    cv_result = run_cross_validation(best_model, x_train, y_train, folds=folds)
    y_pred = best_model.predict(x_test); y_score = best_model.predict_proba(x_test)[:, 1]
    evaluation = evaluate_classifier(y_test, y_pred, y_score)
    metrics = evaluation_to_metrics_dict(evaluation, cv_result)
    metrics["best_parameters"] = tuning.best_params_
    importance = feature_importance_frame(transformed_feature_names(best_model.named_steps["preprocessor"], spec),
        best_model.named_steps["classifier"].feature_importances_)
    grid = (load_prediction_csv(config.prediction_csv, columns=config.columns, feature_columns=spec.all_features)
            if config.prediction_csv else train_df)
    predictions = predict_suitability(best_model, grid, spec)
    joblib.dump(best_model, config.output_dir / MODEL_FILENAME)
    save_json(config.output_dir / METRICS_FILENAME, metrics)
    save_json(config.output_dir / CLASSIFICATION_REPORT_FILENAME, {"report": evaluation.classification_report})
    save_dataframe(config.output_dir / FEATURE_IMPORTANCE_FILENAME, importance)
    save_dataframe(config.output_dir / PREDICTIONS_FILENAME, predictions)
    save_dataframe(config.output_dir / CONFUSION_MATRIX_FILENAME, confusion_matrix_frame(evaluation.confusion_matrix))
    save_dataframe(config.output_dir / ROC_CURVE_FILENAME, roc_curve_frame(evaluation.roc_fpr, evaluation.roc_tpr))
    save_dataframe(config.output_dir / VARIABLE_STATS_FILENAME, variable_distribution_stats(train_df, spec.all_features, label_column=LABEL_COLUMN))
    save_dataframe(config.output_dir / TRAINING_FEATURES_FILENAME, train_df)
    metadata = build_metadata(dataset_name=Path(config.training_csv).name, training_size=len(y_train), testing_size=len(y_test),
        hyperparameters=tuning.best_params_, metrics=metrics, feature_columns=spec.all_features, random_state=config.random_state)
    metadata.update({"numeric_features": list(spec.numeric_features), "categorical_features": list(spec.categorical_features)})
    save_metadata(config.output_dir / METADATA_FILENAME, metadata)
    return TrainingResult(*(config.output_dir / n for n in (MODEL_FILENAME, METRICS_FILENAME, FEATURE_IMPORTANCE_FILENAME, PREDICTIONS_FILENAME, METADATA_FILENAME)))

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-csv", required=True, type=Path); parser.add_argument("--prediction-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACT_PATH)
    args = parser.parse_args(argv)
    run_training_pipeline(TrainingConfig(args.training_csv, args.prediction_csv, args.output_dir))
    return 0
if __name__ == "__main__": raise SystemExit(main())
