"""Model evaluation metrics and tabular exports for the dashboard."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    classification_report,
)
from sklearn.model_selection import cross_validate


@dataclass(frozen=True)
class EvaluationResult:
    """Hold-out test metrics and arrays for plotting."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: np.ndarray
    roc_fpr: np.ndarray
    roc_tpr: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    y_score: np.ndarray
    classification_report: dict


@dataclass(frozen=True)
class CrossValidationResult:
    """Summary of 5-fold cross-validation on the training split."""

    accuracy_mean: float
    accuracy_std: float
    precision_mean: float
    precision_std: float
    recall_mean: float
    recall_std: float
    f1_mean: float
    f1_std: float
    roc_auc_mean: float
    roc_auc_std: float
    folds: int


def evaluate_classifier(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
) -> EvaluationResult:
    """Compute classification metrics on a held-out test set."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    ys = np.asarray(y_score)

    cm = confusion_matrix(yt, yp, labels=[0, 1])
    fpr, tpr, _ = roc_curve(yt, ys)

    return EvaluationResult(
        accuracy=float(accuracy_score(yt, yp)),
        precision=float(precision_score(yt, yp, zero_division=0)),
        recall=float(recall_score(yt, yp, zero_division=0)),
        f1=float(f1_score(yt, yp, zero_division=0)),
        roc_auc=float(roc_auc_score(yt, ys)),
        confusion_matrix=cm,
        roc_fpr=fpr,
        roc_tpr=tpr,
        y_true=yt,
        y_pred=yp,
        y_score=ys,
        classification_report=classification_report(yt, yp, output_dict=True, zero_division=0),
    )


def effective_cv_folds(y_train: pd.Series, requested: int) -> int:
    """Cap folds so stratified CV remains valid for imbalanced/small training splits."""
    counts = y_train.value_counts()
    if counts.empty:
        return max(2, min(requested, 2))
    min_class = int(counts.min())
    if min_class < 2:
        raise ValueError(
            "Training split has fewer than 2 rows in at least one class; "
            "cannot run cross-validation. Provide more labelled data."
        )
    return max(2, min(requested, min_class))


def run_cross_validation(model, x_train: pd.DataFrame, y_train: pd.Series, *, folds: int = 5) -> CrossValidationResult:
    """Run stratified k-fold cross-validation on the training split."""
    folds = effective_cv_folds(y_train, folds)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    cv = cross_validate(
        model,
        x_train,
        y_train,
        cv=folds,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
    )

    def _mean_std(key: str) -> tuple[float, float]:
        values = cv[f"test_{key}"]
        return float(np.mean(values)), float(np.std(values))

    acc_m, acc_s = _mean_std("accuracy")
    pre_m, pre_s = _mean_std("precision")
    rec_m, rec_s = _mean_std("recall")
    f1_m, f1_s = _mean_std("f1")
    auc_m, auc_s = _mean_std("roc_auc")

    return CrossValidationResult(
        accuracy_mean=acc_m,
        accuracy_std=acc_s,
        precision_mean=pre_m,
        precision_std=pre_s,
        recall_mean=rec_m,
        recall_std=rec_s,
        f1_mean=f1_m,
        f1_std=f1_s,
        roc_auc_mean=auc_m,
        roc_auc_std=auc_s,
        folds=folds,
    )


def evaluation_to_metrics_dict(
    test_eval: EvaluationResult,
    cv_eval: CrossValidationResult,
) -> dict:
    """JSON-serialisable metrics bundle."""
    return {
        "test": {
            "accuracy": test_eval.accuracy,
            "precision": test_eval.precision,
            "recall": test_eval.recall,
            "f1": test_eval.f1,
            "roc_auc": test_eval.roc_auc,
        },
        "cross_validation": {
            "folds": cv_eval.folds,
            "accuracy_mean": cv_eval.accuracy_mean,
            "accuracy_std": cv_eval.accuracy_std,
            "precision_mean": cv_eval.precision_mean,
            "precision_std": cv_eval.precision_std,
            "recall_mean": cv_eval.recall_mean,
            "recall_std": cv_eval.recall_std,
            "f1_mean": cv_eval.f1_mean,
            "f1_std": cv_eval.f1_std,
            "roc_auc_mean": cv_eval.roc_auc_mean,
            "roc_auc_std": cv_eval.roc_auc_std,
        },
    }


def confusion_matrix_frame(cm: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"],
    ).reset_index(names="actual")


def roc_curve_frame(fpr: np.ndarray, tpr: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"fpr": fpr, "tpr": tpr})


def feature_importance_frame(names: list[str], values: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"feature": names, "importance": values})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def variable_distribution_stats(
    df: pd.DataFrame,
    feature_columns: list[str],
    *,
    label_column: str = "label",
) -> pd.DataFrame:
    """Summary statistics for dashboard variable distribution plots."""
    rows = []
    has_label = label_column in df.columns
    for col in feature_columns:
        if col not in df.columns:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            for label in ([0, 1] if has_label else [None]):
                subset = df.loc[df[label_column] == label, col] if label is not None else series
                rows.append({
                    "variable": col,
                    "class": label if label is not None else "all",
                    "count": int(subset.count()),
                    "mean": float(subset.mean()) if subset.count() else None,
                    "std": float(subset.std()) if subset.count() > 1 else None,
                    "min": float(subset.min()) if subset.count() else None,
                    "max": float(subset.max()) if subset.count() else None,
                })
        else:
            counts = series.value_counts(dropna=False)
            for value, count in counts.items():
                rows.append({
                    "variable": col,
                    "class": "all",
                    "count": int(count),
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "category": str(value),
                })
    return pd.DataFrame(rows)
