"""Application configuration — paths, criteria, thresholds, colours."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SYSTEMS_CSV = DATA_DIR / "systems.csv"
EVIDENCE_CSV = DATA_DIR / "evidence.csv"
RETEST_CSV = DATA_DIR / "retest_scores.csv"

EVIDENCE_UNAVAILABLE = "Evidence unavailable"

CRITERIA = [
    "api_availability",
    "schema_completeness",
    "semantic_alignment",
    "update_frequency",
    "spatial_resolution",
    "temporal_coverage",
    "license_openness",
    "data_quality",
    "documentation_quality",
    "governance_gdpr_clarity",
]

IDENTITY_COLUMNS = ["system_id", "system_name"]

METADATA_COLUMNS = [
    "operator_org",
    "countries_covered",
    "data_type",
    "access_method",
    "update_freq_desc",
    "spatial_res_desc",
    "temporal_cov_desc",
    "license_type",
    "governance_body",
    "standards_used",
    "notes",
]

SCORE_MIN = 0
SCORE_MAX = 2

READINESS_HIGH_MIN = 15
READINESS_MEDIUM_MIN = 10

COUNTRY_COLUMN = "countries_covered"

CRITERIA_LABELS = {
    "api_availability": "API / Data Availability",
    "schema_completeness": "Schema Completeness",
    "semantic_alignment": "Semantic Alignment",
    "update_frequency": "Update Frequency",
    "spatial_resolution": "Spatial Resolution",
    "temporal_coverage": "Temporal Coverage",
    "license_openness": "License Openness",
    "data_quality": "Data Quality",
    "documentation_quality": "Documentation Quality",
    "governance_gdpr_clarity": "Governance / GDPR Clarity",
}

READINESS_COLORS = {
    "High": "#27ae60",    # Professional forest green
    "Medium": "#f39c12",  # Warm amber (unchanged)
    "Low": "#e74c3c",     # Professional red (unchanged)  
    "Unknown": "#7f8c8d", # Neutral grey
}

INTEGRATION_COLORS = {
    "High integration ready": "#27ae60",
    "Medium integration ready": "#f39c12",
    "Low integration ready": "#c0392b",
}

BARRIER_SEVERITY_COLORS = {
    "High": "#c0392b",
    "Medium": "#f39c12",
    "Low": "#27ae60",
    "Unknown": "#95a5a6",
}

EVIDENCE_COLUMNS = [
    "system_id",
    "official_website",
    "research_publication",
    "doi",
    "data_source",
    "publisher",
    "reference",
    "date_accessed",
    "url",
]

DASH_HOST = "127.0.0.1"
DASH_PORT = 8050
