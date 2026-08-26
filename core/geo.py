# system-level metadata only — never biological records

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

COUNTRY_NAMES: dict[str, str] = {
    "FRA": "France",
    "DEU": "Germany",
    "GBR": "United Kingdom",
    "NLD": "Netherlands",
    "CHE": "Switzerland",
    "HUN": "Hungary",
    "DNK": "Denmark",
    "AUT": "Austria",
    "BEL": "Belgium",
    "ESP": "Spain",
    "ITA": "Italy",
    "POL": "Poland",
    "SWE": "Sweden",
    "NOR": "Norway",
    "FIN": "Finland",
    "IRL": "Ireland",
    "PRT": "Portugal",
    "CZE": "Czechia",
    "ROU": "Romania",
    "GRC": "Greece",
    "SVK": "Slovakia",
    "SVN": "Slovenia",
    "HRV": "Croatia",
    "BGR": "Bulgaria",
    "EST": "Estonia",
    "LVA": "Latvia",
    "LTU": "Lithuania",
    "LUX": "Luxembourg",
    "EUR": "Europe (EU-wide)",
}

COUNTRY_CENTROIDS: dict[str, tuple[float, float]] = {
    "FRA": (46.23, 2.21),
    "DEU": (51.17, 10.45),
    "GBR": (54.00, -2.50),
    "NLD": (52.13, 5.29),
    "CHE": (46.82, 8.23),
    "HUN": (47.16, 19.50),
    "DNK": (56.26, 9.50),
    "AUT": (47.52, 14.55),
    "BEL": (50.50, 4.47),
    "ESP": (40.46, -3.75),
    "ITA": (41.87, 12.57),
    "POL": (51.92, 19.15),
    "SWE": (60.13, 18.64),
    "NOR": (60.47, 8.47),
    "FIN": (61.92, 25.75),
    "IRL": (53.41, -8.24),
    "PRT": (39.40, -8.22),
    "CZE": (49.82, 15.47),
    "ROU": (45.94, 24.97),
    "GRC": (39.07, 21.82),
    "SVK": (48.67, 19.70),
    "SVN": (46.15, 14.99),
    "HRV": (45.10, 15.20),
    "BGR": (42.73, 25.49),
    "EST": (58.60, 25.01),
    "LVA": (56.88, 24.60),
    "LTU": (55.17, 23.88),
    "LUX": (49.82, 6.13),
    "EUR": (54.53, 15.00),
}

COUNTRY_ALIASES: dict[str, str] = {
    "united kingdom": "GBR",
    "great britain": "GBR",
    "england and wales": "GBR",
    "england": "GBR",
    "wales": "GBR",
    "scotland": "GBR",
    "northern ireland": "GBR",
    "britain": "GBR",
    "ukhsa": "GBR",
    "netherlands": "NLD",
    "holland": "NLD",
    "dutch": "NLD",
    "tekenradar": "NLD",
    "france": "FRA",
    "french": "FRA",
    "citique": "FRA",
    "germany": "DEU",
    "german": "DEU",
    "deutschland": "DEU",
    "länder": "DEU",
    "switzerland": "CHE",
    "swiss": "CHE",
    "hungary": "HUN",
    "hungarian": "HUN",
    "tickwatcher": "HUN",
    "denmark": "DNK",
    "danish": "DNK",
    "austria": "AUT",
    "belgium": "BEL",
    "belgian": "BEL",
    "spain": "ESP",
    "italy": "ITA",
    "poland": "POL",
    "sweden": "SWE",
    "norway": "NOR",
    "finland": "FIN",
    "ireland": "IRL",
    "portugal": "PRT",
    "czech": "CZE",
    "czechia": "CZE",
    "romania": "ROU",
    "greece": "GRC",
    "slovakia": "SVK",
    "slovenia": "SVN",
    "croatia": "HRV",
    "bulgaria": "BGR",
    "estonia": "EST",
    "latvia": "LVA",
    "lithuania": "LTU",
    "luxembourg": "LUX",
    # ISO-2 fragments often seen in metadata
    "fr": "FRA",
    "de": "DEU",
    "nl": "NLD",
    "dk": "DNK",
    "hu": "HUN",
    "ch": "CHE",
    "uk": "GBR",
}

EU_WIDE_PATTERNS = (
    "eu-wide",
    "eu wide",
    "european union",
    "eu +",
    "eu+",
    " eea",
    "efta",
    "27 states",
    "27 member",
    "pan-european",
    "pan european",
    "pan european coordination",
    "europe-wide",
    "europe wide",
    "multi-country eu",
    "eu-wide (",
    "efsa / ecdc",
    "vectornet",
)

_SKIP_PATTERNS = ("varies", "not applicable", "unknown", "tbd")
_ALIAS_KEYS_SORTED = sorted(COUNTRY_ALIASES.keys(), key=len, reverse=True)
_FUZZY_THRESHOLD = 0.80

_UK_PATTERN = re.compile(r"\buk\b|\bu\.k\.?\b", re.IGNORECASE)


@dataclass
class MapBuildResult:
    """Outcome of map dataframe construction — never silent failure."""

    scatter_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    choropleth_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    systems_loaded: int = 0
    systems_mapped: int = 0
    systems_skipped: int = 0
    skipped_details: list[dict[str, str]] = field(default_factory=list)
    unmapped_system_ids: list[str] = field(default_factory=list)

    @property
    def has_points(self) -> bool:
        return not self.scatter_df.empty


def _normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s&;,\-/]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _is_empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return str(value).strip() == ""


def _is_eu_wide(text: str) -> bool:
    norm = _normalize(text)
    return any(p in norm for p in EU_WIDE_PATTERNS)


def _tokenize_countries(text: str) -> list[str]:
    norm = _normalize(text)
    if not norm:
        return []
    parts = re.split(r"[;/,&]|\band\b", norm)
    return [p.strip() for p in parts if p.strip()]


def _fuzzy_alias_lookup(token: str) -> str | None:
    token = _normalize(token)
    if not token:
        return None
    if any(s in token for s in _SKIP_PATTERNS) and len(token) < 12:
        return None
    if _UK_PATTERN.search(token):
        return "GBR"

    for alias in _ALIAS_KEYS_SORTED:
        if alias == token or f" {alias} " in f" {token} " or token.startswith(f"{alias} "):
            return COUNTRY_ALIASES[alias]
        if len(alias) >= 4 and alias in token:
            return COUNTRY_ALIASES[alias]

    best_iso: str | None = None
    best_ratio = 0.0
    for alias, iso3 in COUNTRY_ALIASES.items():
        for candidate in [token] + token.split():
            if len(candidate) < 2:
                continue
            ratio = SequenceMatcher(None, candidate, alias).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_iso = iso3

    if best_ratio >= _FUZZY_THRESHOLD:
        return best_iso
    return None


def resolve_iso3_list(text) -> list[str]:
    """Resolve free text to ISO-3 code list."""
    if _is_empty(text):
        return []
    raw = str(text).strip()
    if _is_eu_wide(raw):
        return ["EUR"]

    tokens = _tokenize_countries(raw) or [raw]
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        if _is_eu_wide(token):
            if "EUR" not in seen:
                seen.add("EUR")
                result.append("EUR")
            continue
        iso3 = _fuzzy_alias_lookup(token)
        if iso3 and iso3 not in seen:
            seen.add(iso3)
            result.append(iso3)
    return result


def resolve_system_location(
    countries_covered=None,
    system_name=None,
    operator_org=None,
    notes=None,
) -> tuple[list[str], str]:
    """
    Multi-source location resolver.

    Returns (iso3_list, source_field) where source_field explains which column matched.
    """
    sources = [
        ("countries_covered", countries_covered),
        ("system_name", system_name),
        ("operator_org", operator_org),
        ("notes", notes),
    ]
    for field_name, value in sources:
        codes = resolve_iso3_list(value)
        if codes:
            return codes, field_name
    return [], "none"


def country_label_from_iso3_all(iso3_all) -> str:
    """Turn comma-separated ISO-3 codes into readable country name(s)."""
    if _is_empty(iso3_all):
        return ""
    codes = [c.strip() for c in str(iso3_all).split(",") if c.strip()]
    if not codes:
        return ""
    names = [COUNTRY_NAMES.get(code, code) for code in codes]
    return ", ".join(names)


ISO3_TO_ISO2: dict[str, str] = {
    "FRA": "FR", "DEU": "DE", "GBR": "GB", "NLD": "NL", "CHE": "CH",
    "HUN": "HU", "DNK": "DK", "AUT": "AT", "BEL": "BE", "ESP": "ES",
    "ITA": "IT", "POL": "PL", "SWE": "SE", "NOR": "NO", "FIN": "FI",
    "IRL": "IE", "PRT": "PT", "CZE": "CZ", "ROU": "RO", "GRC": "GR",
    "SVK": "SK", "SVN": "SI", "HRV": "HR", "BGR": "BG", "EST": "EE",
    "LVA": "LV", "LTU": "LT", "LUX": "LU", "EUR": "EU",
}


def short_country_label_from_iso3_all(iso3_all) -> str:
    """Compact ISO-2 form for on-map text (e.g. "FR, DK, NL") — full names stay in hover text."""
    if _is_empty(iso3_all):
        return ""
    codes = [c.strip() for c in str(iso3_all).split(",") if c.strip()]
    if not codes:
        return ""
    return ", ".join(ISO3_TO_ISO2.get(code, code) for code in codes)


def centroid_for_iso3_list(iso3_list: list[str]) -> tuple[float | None, float | None]:
    if not iso3_list:
        return None, None
    coords = [COUNTRY_CENTROIDS[c] for c in iso3_list if c in COUNTRY_CENTROIDS]
    if not coords:
        return None, None
    arr = np.array(coords)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def _resolve_row_locations(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized-first resolution with column-wise fallback for unmapped rows."""
    out = df.copy()
    n = len(out)
    out["_iso3_list"] = pd.Series([[] for _ in range(n)], dtype=object)
    out["_location_source"] = ""

    fallback_cols = [
        ("countries_covered", "countries_covered"),
        ("country", "countries_covered"),
        ("system_name", "system_name"),
        ("operator_org", "operator_org"),
        ("notes", "notes"),
    ]

    pending = pd.Series(True, index=out.index)
    for col, source_label in fallback_cols:
        if col not in out.columns:
            continue
        mask = pending & out[col].notna() & (out[col].astype(str).str.strip() != "")
        if not mask.any():
            continue
        resolved = out.loc[mask, col].map(resolve_iso3_list)
        hit_idx = resolved[resolved.map(len) > 0].index
        out.loc[hit_idx, "_iso3_list"] = resolved.loc[hit_idx]
        out.loc[hit_idx, "_location_source"] = source_label
        pending = out["_iso3_list"].map(len) == 0

    return out


def build_map_dataframe(
    df: pd.DataFrame,
    country_col: str = "countries_covered",
) -> MapBuildResult:
    """
    Build scatter and choropleth map data with explicit skip diagnostics.
    """
    result = MapBuildResult()
    if df is None or df.empty:
        result.skipped_details.append({"system_id": "—", "reason": "Input dataframe is empty"})
        return result

    if "system_id" not in df.columns:
        result.skipped_details.append({"system_id": "—", "reason": "Missing system_id column"})
        return result

    work = _resolve_row_locations(df)
    result.systems_loaded = len(work)

    coords = work["_iso3_list"].map(centroid_for_iso3_list)
    work = work.assign(
        lat=coords.map(lambda c: c[0]),
        lon=coords.map(lambda c: c[1]),
    )

    mappable = work["lat"].notna() & work["lon"].notna()
    unmapped = work.loc[~mappable]
    country_series = (
        unmapped[country_col]
        if country_col in unmapped.columns
        else unmapped.get("countries_covered", pd.Series("", index=unmapped.index))
    )

    for sid, src, countries in zip(
        unmapped.get("system_id", pd.Series(dtype=str)),
        unmapped.get("_location_source", pd.Series(dtype=str)),
        country_series,
    ):
        reason = "No recognisable country in metadata"
        if _is_empty(countries) and (not src or src == ""):
            reason = "Missing countries_covered and no fallback match in system_name"
        elif src == "none" or not src:
            reason = f"Unrecognised location text: {countries!r}"
        else:
            reason = f"Could not geocode resolved codes from {src}"
        entry = {"system_id": str(sid), "reason": reason}
        result.skipped_details.append(entry)
        result.unmapped_system_ids.append(str(sid))

    result.systems_skipped = int((~mappable).sum())
    result.systems_mapped = int(mappable.sum())

    if not mappable.any():
        return result

    mapped = work.loc[mappable].copy()

    if "readiness_class" in mapped.columns:
        readiness = mapped["readiness_class"].replace("", np.nan).fillna("Unknown")
    else:
        readiness = pd.Series("Unknown", index=mapped.index)

    mapped = mapped.assign(
        iso3=mapped["_iso3_list"].map(lambda x: x[0] if x else None),
        iso3_all=mapped["_iso3_list"].map(lambda x: ",".join(x) if x else ""),
        map_total_score=pd.to_numeric(
            mapped.get("total_score", 0), errors="coerce"
        ).fillna(0),
        map_readiness=readiness,
    )

    country_series = mapped[country_col] if country_col in mapped.columns else mapped.get(
        "countries_covered", ""
    )
    result.scatter_df = pd.DataFrame(
        {
            "system_id": mapped["system_id"],
            "system_name": mapped.get("system_name", ""),
            "countries_covered": country_series,
            "iso3": mapped["iso3"],
            "iso3_all": mapped["iso3_all"],
            "lat": mapped["lat"],
            "lon": mapped["lon"],
            "total_score": mapped["map_total_score"],
            "readiness_class": mapped["map_readiness"],
            "location_source": mapped["_location_source"],
        }
    )

    multi = result.scatter_df["iso3_all"].str.contains(",", na=False)
    choro_mask = (result.scatter_df["iso3"] != "EUR") & ~multi
    result.choropleth_df = result.scatter_df.loc[choro_mask].copy()

    return result
