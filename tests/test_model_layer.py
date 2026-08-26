"""Tests for the field-validated tick model module (model_layer.py)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import model_layer


def _tiny_field_frame() -> pd.DataFrame:
    sites = [f"site_{i}" for i in range(6)]
    rows = []
    for i, site in enumerate(sites):
        for j in range(4):
            rows.append({
                "site": site,
                "lat": 48.0 + i * 0.1,
                "lon": 2.0 + i * 0.1,
                "present": int((i + j) % 2 == 0),
                "count_ticks": (i + j) % 3,
                "RH at sample point (%)": 70 + j,
                "Temperature at sample point °C)": 15 + j,
                "Wind speed at sample point (kph)": 5 + j,
                "Elevation (metres)": 100 + i * 10,
                "Temperature max (Degrees Celsius) for day": 18 + j,
                "Max rainfall (mm) reading for day": 2.0 + j,
                "NDVI": 0.5 + 0.01 * j,
                "Land use": "forest" if i % 2 == 0 else "pasture",
                "Discontinuous urban fabric": 0,
                "Month": 5 + (j % 3),
            })
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _patch_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    field_dir = tmp_path / "field_model"
    field_dir.mkdir()
    monkeypatch.setattr(model_layer, "FIELD_MODEL_DIR", field_dir)
    monkeypatch.setattr(model_layer, "MODEL_PATH", field_dir / "model.pkl")
    monkeypatch.setattr(model_layer, "MODEL_METADATA_PATH", field_dir / "model_metadata.json")
    monkeypatch.setattr(model_layer, "FIELD_CSV", field_dir / "field_clean.csv")
    monkeypatch.setattr(model_layer, "OCCURRENCE_CSV", field_dir / "occurrence_layer.csv")
    monkeypatch.setattr(model_layer, "ENVIRONMENT_CSV", field_dir / "environment_layer.csv")
    monkeypatch.setattr(model_layer, "CELLS_CSV", field_dir / "dashboard_cells.csv")
    monkeypatch.setattr(model_layer, "SITE_GROUPED_AUC", None)
    yield field_dir


def test_missing_field_csv_degrades_gracefully(_patch_paths: Path) -> None:
    result = model_layer.load_model_layer_data()
    assert result.available is False
    assert result.error is not None
    assert "field_clean.csv" in result.error


def test_resolve_feature_columns_excludes_banned_fields() -> None:
    columns = list(_tiny_field_frame().columns)
    resolved = model_layer._resolve_feature_columns(columns)
    assert set(resolved) == set(model_layer.FEATURE_FRAGMENTS)
    model_layer._assert_no_banned_fields(list(resolved.values()))


def test_assert_no_banned_fields_raises_on_leak() -> None:
    with pytest.raises(AssertionError):
        model_layer._assert_no_banned_fields(["Vegetation sample point", "NDVI"])


def test_fallback_training_when_model_missing(_patch_paths: Path) -> None:
    field_dir = _patch_paths
    df = _tiny_field_frame()
    df.to_csv(field_dir / "field_clean.csv", index=False)

    result = model_layer.load_model_layer_data()

    assert result.error is None
    assert result.available is True
    assert result.model_source == "trained_fallback"
    assert (field_dir / "model.pkl").exists()
    assert result.field["pred_prob"].between(0, 1).all()
    assert result.n_sites == 6
    assert result.n_points == 24


def test_loaded_model_is_not_retrained(_patch_paths: Path) -> None:
    field_dir = _patch_paths
    df = _tiny_field_frame()
    df.to_csv(field_dir / "field_clean.csv", index=False)

    first = model_layer.load_model_layer_data()
    assert first.model_source == "trained_fallback"
    saved_mtime = (field_dir / "model.pkl").stat().st_mtime_ns

    second = model_layer.load_model_layer_data()
    assert second.model_source == "loaded"
    assert (field_dir / "model.pkl").stat().st_mtime_ns == saved_mtime
    # Loading (not retraining) must still surface the honest CV AUC from the
    # persisted sidecar, not a leaky in-sample number recomputed on load.
    assert second.auc_is_insample is False
    assert second.auc == first.auc


def test_metadata_sidecar_is_written_on_fallback_training(_patch_paths: Path) -> None:
    field_dir = _patch_paths
    _tiny_field_frame().to_csv(field_dir / "field_clean.csv", index=False)

    model_layer.load_model_layer_data()

    assert (field_dir / "model_metadata.json").exists()
    import json

    payload = json.loads((field_dir / "model_metadata.json").read_text())
    assert "site_grouped_auc" in payload
    assert payload["n_sites"] == 6


def test_display_only_layers_load_when_present(_patch_paths: Path) -> None:
    field_dir = _patch_paths
    _tiny_field_frame().to_csv(field_dir / "field_clean.csv", index=False)
    pd.DataFrame({"lat": [48.0], "lon": [2.0], "source": ["x"]}).to_csv(
        field_dir / "occurrence_layer.csv", index=False
    )

    result = model_layer.load_model_layer_data()
    assert result.occurrence is not None
    assert result.environment is None
    assert result.cells is None
