"""Build immutable framework snapshot at startup."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.barrier_details import build_barrier_detail_table
from core.barriers import classify_barriers
from core.evidence import merge_evidence
from core.geo import MapBuildResult, build_map_dataframe
from core.integration import classify_integration_readiness, enrich_map_dataframe
from core.preparation import DataQualityReport, attach_map_quality, prepare_systems_dataframe
from core.recommendations import generate_recommendations
from core.validation import validate_systems_df
from data.loaders import load_evidence, load_systems


@dataclass(frozen=True)
class FrameworkSnapshot:
    """Immutable bundle of all analytical layers for the Dash UI."""

    systems: pd.DataFrame
    quality: DataQualityReport
    map_result: MapBuildResult
    barriers: pd.DataFrame
    integration: pd.DataFrame
    map_scatter: pd.DataFrame
    barrier_details: pd.DataFrame
    recommendations: pd.DataFrame
    evidence: pd.DataFrame

    @property
    def ok(self) -> bool:
        return not self.systems.empty


def _empty_snapshot() -> FrameworkSnapshot:
    return FrameworkSnapshot(
        systems=pd.DataFrame(),
        quality=DataQualityReport(),
        map_result=MapBuildResult(),
        barriers=pd.DataFrame(),
        integration=pd.DataFrame(),
        map_scatter=pd.DataFrame(),
        barrier_details=pd.DataFrame(),
        recommendations=pd.DataFrame(),
        evidence=pd.DataFrame(),
    )


def build_framework() -> FrameworkSnapshot:
    try:
        raw = load_systems()
        validation = validate_systems_df(raw, require_complete_scores=False)
        systems, quality = prepare_systems_dataframe(validation.df, auto_score_if_empty=False)
        if systems.empty:
            return _empty_snapshot()

        map_result = build_map_dataframe(systems)
        attach_map_quality(quality, map_result)

        barriers = classify_barriers(systems)
        integration = classify_integration_readiness(systems, barriers)
        map_scatter = enrich_map_dataframe(map_result.scatter_df, integration)
        barrier_details = build_barrier_detail_table(systems, barriers)
        recommendations = generate_recommendations(systems)
        evidence = merge_evidence(systems, load_evidence())

        return FrameworkSnapshot(
            systems=systems,
            quality=quality,
            map_result=map_result,
            barriers=barriers,
            integration=integration,
            map_scatter=map_scatter,
            barrier_details=barrier_details,
            recommendations=recommendations,
            evidence=evidence,
        )
    except Exception as exc:
        quality = DataQualityReport(errors=[f"Load failed: {exc}"])
        return FrameworkSnapshot(
            systems=pd.DataFrame(),
            quality=quality,
            map_result=MapBuildResult(),
            barriers=pd.DataFrame(),
            integration=pd.DataFrame(),
            map_scatter=pd.DataFrame(),
            barrier_details=pd.DataFrame(),
            recommendations=pd.DataFrame(),
            evidence=pd.DataFrame(),
        )
