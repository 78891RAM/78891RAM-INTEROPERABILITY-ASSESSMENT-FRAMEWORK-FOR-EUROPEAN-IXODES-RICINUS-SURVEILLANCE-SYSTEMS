# Archive

Code and data moved here deliberately, not deleted, because it's no longer part of the live app but still has reference value.

## `lib/ml`, `data/ml`, `research/ml`, `tests/test_ml.py`

The previous ecological-suitability ML pipeline: a general-purpose Random Forest trainer (`lib/ml/train_model.py`) that took a user-supplied training CSV in `data/ml/input/`, ran grid search + cross-validation, and wrote artifacts to `data/ml/artifacts/` for the dashboard to load.

**Superseded** by `model_layer.py` (project root) and `data/field_model/`, which load a pre-trained, field-validated model instead of training generically — see `docs/architecture.md`. The Ecological Suitability tab (`ui/tabs/suitability.py`) has not imported `lib.ml` since that change; this archive move only makes that explicit.

Kept as-is (imports, tests, and behaviour unchanged) in case the generic-training approach is useful again later — e.g. as a fallback path or a comparison baseline. Nothing in `ui/`, `app.py`, or `data/pipeline.py` references this archive; it is not wired into the running app.

To run its tests standalone: `cd archive && python -m pytest tests/test_ml.py` (needs `archive/` on the Python path, since it re-establishes the original `lib/ml` package layout).
