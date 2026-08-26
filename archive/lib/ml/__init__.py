"""
Ecological suitability modelling — independent from interoperability scoring.

Train with::

    python -m lib.ml.train_model --training-csv path/to/training.csv

Optional prediction grid::

    python -m lib.ml.train_model --training-csv path/to/training.csv \\
        --prediction-csv path/to/grid.csv
"""

__all__ = ["MLArtifacts", "TrainingConfig", "artifacts_available", "load_artifacts", "run_training_pipeline"]


def __getattr__(name: str):
    """Lazily expose public API without pre-importing the CLI module."""
    if name in {"MLArtifacts", "artifacts_available", "load_artifacts"}:
        from lib.ml import data_loader
        return getattr(data_loader, name)
    if name in {"TrainingConfig", "run_training_pipeline"}:
        from lib.ml import train_model
        return getattr(train_model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
