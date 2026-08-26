"""Plotly figures for the Ecological Suitability dashboard tab."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from lib.ml.data_loader import MLArtifacts
from ui.styles import CHART_MARGIN, THEME_BLUE


def suitability_map_figure(predictions: pd.DataFrame) -> go.Figure:
    """Europe scatter map coloured by suitability score."""
    if predictions.empty:
        return go.Figure()

    df = predictions.copy()
    if "probability" not in df.columns:
        return go.Figure()

    fig = px.scatter_geo(
        df,
        lat="latitude",
        lon="longitude",
        color="probability",
        color_continuous_scale="RdYlGn",
        range_color=(0, 1),
        hover_data={
            "latitude": ":.4f",
            "longitude": ":.4f",
            "probability": ":.3f",
            "predicted_class": True,
        },
        title="Ecological Suitability Map",
    )
    fig.update_geos(
        scope="europe",
        projection_type="natural earth",
        showcountries=True,
        showland=True,
        showocean=True,
        landcolor="#f4f6f7",
        oceancolor="#e8f4f8",
        countrycolor="#bdc3c7",
        lataxis_range=[34, 72],
        lonaxis_range=[-25, 45],
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        margin=CHART_MARGIN,
        height=560,
        coloraxis_colorbar=dict(title="Probability"),
        font=dict(family="system-ui, sans-serif", size=13, color="#2c3e50"),
    )
    return fig


def feature_importance_figure(feature_importance: pd.DataFrame, *, top_n: int = 15) -> go.Figure:
    if feature_importance.empty:
        return go.Figure()

    df = feature_importance.head(top_n).sort_values("importance", ascending=True)
    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        title="Feature Importance",
        labels={"importance": "Importance", "feature": "Feature"},
    )
    fig.update_traces(marker_color=THEME_BLUE)
    fig.update_layout(margin=CHART_MARGIN, height=420, yaxis_title="")
    return fig


def confusion_matrix_figure(confusion_matrix: pd.DataFrame) -> go.Figure:
    if confusion_matrix.empty:
        return go.Figure()

    value_cols = [c for c in confusion_matrix.columns if c.startswith("Predicted")]
    if not value_cols:
        return go.Figure()

    z = confusion_matrix[value_cols].to_numpy()
    x_labels = [c.replace("Predicted ", "") for c in value_cols]
    y_labels = confusion_matrix["actual"].astype(str).str.replace("Actual ", "", regex=False).tolist()

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        colorscale="Blues",
        text=z,
        texttemplate="%{text}",
        hovertemplate="Actual %{y}<br>Predicted %{x}<br>Count %{z}<extra></extra>",
    ))
    fig.update_layout(
        title="Confusion Matrix (Test Set)",
        margin=CHART_MARGIN,
        height=380,
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    return fig


def roc_curve_figure(roc_curve: pd.DataFrame, *, auc: float | None = None) -> go.Figure:
    if roc_curve.empty:
        return go.Figure()

    title = "ROC Curve (Test Set)"
    if auc is not None:
        title = f"ROC Curve (Test Set) — AUC = {auc:.3f}"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=roc_curve["fpr"],
        y=roc_curve["tpr"],
        mode="lines",
        name="ROC",
        line=dict(color=THEME_BLUE, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Chance",
        line=dict(color="#95a5a6", dash="dash"),
    ))
    fig.update_layout(
        title=title,
        margin=CHART_MARGIN,
        height=380,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    return fig


def variable_distribution_figure(
    training_features: pd.DataFrame,
    metadata: dict | None = None,
) -> go.Figure:
    """Overlaid histograms of numeric ecological variables by presence class."""
    if training_features.empty or "label" not in training_features.columns:
        return go.Figure()

    numeric = (metadata or {}).get("numeric_features", [])
    present = [c for c in numeric if c in training_features.columns]
    if not present:
        return go.Figure()

    col = present[0]
    plot_df = training_features.copy()
    plot_df["class"] = plot_df["label"].map({0: "Background", 1: "Presence"})
    fig = px.histogram(
        plot_df,
        x=col,
        color="class",
        barmode="overlay",
        opacity=0.65,
        title=f"{col.title()} Distribution by Class (Training Data)",
        labels={col: col, "class": "Class"},
    )
    fig.update_layout(margin=CHART_MARGIN, height=360, bargap=0.05)
    return fig


def suitability_distribution_figure(predictions: pd.DataFrame) -> go.Figure:
    """Histogram of predicted suitability scores."""
    if predictions.empty or "probability" not in predictions.columns:
        return go.Figure()

    fig = px.histogram(
        predictions,
        x="probability",
        nbins=30,
        title="Predicted Suitability Distribution",
        labels={"probability": "Suitability probability (0–1)"},
    )
    fig.update_traces(marker_color=THEME_BLUE)
    fig.update_layout(margin=CHART_MARGIN, height=360, bargap=0.05)
    return fig


def build_dashboard_figures(artifacts: MLArtifacts) -> dict[str, go.Figure]:
    """Convenience bundle for the suitability tab."""
    test_auc = artifacts.metrics.get("test", {}).get("roc_auc")
    return {
        "map": suitability_map_figure(artifacts.predictions),
        "importance": feature_importance_figure(artifacts.feature_importance),
        "confusion_matrix": confusion_matrix_figure(artifacts.confusion_matrix),
        "roc": roc_curve_figure(artifacts.roc_curve, auc=test_auc),
        "distribution": variable_distribution_figure(
            artifacts.training_features,
            metadata=artifacts.metadata,
        ),
        "suitability_distribution": suitability_distribution_figure(artifacts.predictions),
    }
