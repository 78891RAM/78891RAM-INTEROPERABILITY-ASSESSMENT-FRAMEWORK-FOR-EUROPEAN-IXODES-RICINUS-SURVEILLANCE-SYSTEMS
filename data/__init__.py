"""Data loading layer — sole disk I/O entry points."""

from data.pipeline import FrameworkSnapshot, build_framework
from data.loaders import load_evidence, load_systems

__all__ = ["FrameworkSnapshot", "build_framework", "load_evidence", "load_systems"]
