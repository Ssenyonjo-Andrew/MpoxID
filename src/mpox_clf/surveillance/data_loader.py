"""
Mpox Surveillance Data Loader & Preprocessing Pipeline.

Loads, caches, cleans, and enriches Our World in Data (OWID) Monkeypox data.
Supports direct live GitHub URL fetching with automatic offline caching and fallback.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OWID_MPOX_URL = "https://raw.githubusercontent.com/owid/monkeypox/main/owid-monkeypox-data.csv"
DEFAULT_LOCAL_PATH = Path(__file__).resolve().parents[3] / "data" / "raw" / "owid-monkeypox-data.csv"

# Known endemic and epidemic clade zones for public-health epidemiology mapping
CLADE_EPIDEMIOLOGY_MAPPING: Dict[str, Dict[str, str]] = {
    # Clade I & Ib (Central / East Africa - Higher virulence / respiratory & household / sexual transmission)
    "COD": {"primary_clade": "Clade Ia / Ib", "risk_tier": "Critical Hotspot", "notes": "Epicenter of Clade Ia (endemic/zoonotic) & Clade Ib (sustained human-to-human transmission)"},
    "BDI": {"primary_clade": "Clade Ib", "risk_tier": "High Alert", "notes": "Clade Ib cross-border spread from eastern DRC"},
    "UGA": {"primary_clade": "Clade Ib", "risk_tier": "High Alert", "notes": "Clade Ib transmission in urban/transit hubs"},
    "RWA": {"primary_clade": "Clade Ib", "risk_tier": "Monitored", "notes": "Clade Ib cross-border containment zone"},
    "KEN": {"primary_clade": "Clade Ib", "risk_tier": "Monitored", "notes": "Clade Ib border screening & monitoring"},
    "COG": {"primary_clade": "Clade Ia", "risk_tier": "Active", "notes": "Congo Basin Clade Ia reservoir"},
    "CAF": {"primary_clade": "Clade Ia", "risk_tier": "Active", "notes": "Central African Republic Clade Ia endemic zone"},
    "CMR": {"primary_clade": "Clade I / IIa", "risk_tier": "Active", "notes": "Geographic suture zone where Clade I and Clade II overlap"},
    # Clade IIa (West Africa)
    "NGA": {"primary_clade": "Clade IIa / IIb", "risk_tier": "Active", "notes": "Historical origin of the 2017–2018 resurgence (Clade IIb progenitor)"},
    "GHA": {"primary_clade": "Clade IIa", "risk_tier": "Monitored", "notes": "West African Clade IIa endemic area"},
    "SLE": {"primary_clade": "Clade IIa", "risk_tier": "Active", "notes": "West African transmission"},
    "LBR": {"primary_clade": "Clade IIa", "risk_tier": "Active", "notes": "West African transmission"},
    # Clade IIb (Global 2022-2024 human-to-human outbreak)
    "USA": {"primary_clade": "Clade IIb", "risk_tier": "Surveillance", "notes": "2022–2023 Clade IIb global outbreak; travel-associated Clade Ib monitoring"},
    "BRA": {"primary_clade": "Clade IIb", "risk_tier": "Surveillance", "notes": "2022–2023 Clade IIb global outbreak"},
    "ESP": {"primary_clade": "Clade IIb", "risk_tier": "Surveillance", "notes": "2022–2023 European epicenter (Clade IIb)"},
    "GBR": {"primary_clade": "Clade IIb", "risk_tier": "Surveillance", "notes": "2022 index cases in Europe (Clade IIb)"},
    "DEU": {"primary_clade": "Clade IIb", "risk_tier": "Surveillance", "notes": "Clade IIb global wave"},
    "FRA": {"primary_clade": "Clade IIb", "risk_tier": "Surveillance", "notes": "Clade IIb global wave"},
    "CAN": {"primary_clade": "Clade IIb", "risk_tier": "Surveillance", "notes": "Clade IIb global wave"},
    "MEX": {"primary_clade": "Clade IIb", "risk_tier": "Surveillance", "notes": "Clade IIb global wave"},
    "COL": {"primary_clade": "Clade IIb", "risk_tier": "Surveillance", "notes": "Clade IIb South American transmission"},
    "PER": {"primary_clade": "Clade IIb", "risk_tier": "Surveillance", "notes": "Clade IIb South American transmission"},
}

CONTINENT_MAP: Dict[str, str] = {
    # Africa
    "DZA": "Africa", "AGO": "Africa", "BEN": "Africa", "BWA": "Africa", "BFA": "Africa",
    "BDI": "Africa", "CPV": "Africa", "CMR": "Africa", "CAF": "Africa", "TCD": "Africa",
    "COM": "Africa", "COG": "Africa", "COD": "Africa", "CIV": "Africa", "DJI": "Africa",
    "EGY": "Africa", "GNQ": "Africa", "ERI": "Africa", "SWZ": "Africa", "ETH": "Africa",
    "GAB": "Africa", "GMB": "Africa", "GHA": "Africa", "GIN": "Africa", "GNB": "Africa",
    "KEN": "Africa", "LSO": "Africa", "LBR": "Africa", "LBY": "Africa", "MDG": "Africa",
    "MWI": "Africa", "MLI": "Africa", "MRT": "Africa", "MUS": "Africa", "MAR": "Africa",
    "MOZ": "Africa", "NAM": "Africa", "NER": "Africa", "NGA": "Africa", "RWA": "Africa",
    "STP": "Africa", "SEN": "Africa", "SYC": "Africa", "SLE": "Africa", "SOM": "Africa",
    "ZAF": "Africa", "SSD": "Africa", "SDN": "Africa", "TZA": "Africa", "TGO": "Africa",
    "TUN": "Africa", "UGA": "Africa", "ZMB": "Africa", "ZWE": "Africa",
    # Europe
    "ALB": "Europe", "AND": "Europe", "AUT": "Europe", "BLR": "Europe", "BEL": "Europe",
    "BIH": "Europe", "BGR": "Europe", "HRV": "Europe", "CYP": "Europe", "CZE": "Europe",
    "DNK": "Europe", "EST": "Europe", "FIN": "Europe", "FRA": "Europe", "DEU": "Europe",
    "GRC": "Europe", "HUN": "Europe", "ISL": "Europe", "IRL": "Europe", "ITA": "Europe",
    "LVA": "Europe", "LIE": "Europe", "LTU": "Europe", "LUX": "Europe", "MLT": "Europe",
    "MDA": "Europe", "MCO": "Europe", "MNE": "Europe", "NLD": "Europe", "MKD": "Europe",
    "NOR": "Europe", "POL": "Europe", "PRT": "Europe", "ROU": "Europe", "RUS": "Europe",
    "SMR": "Europe", "SRB": "Europe", "SVK": "Europe", "SVN": "Europe", "ESP": "Europe",
    "SWE": "Europe", "CHE": "Europe", "UKR": "Europe", "GBR": "Europe", "VAT": "Europe",
    # Americas
    "USA": "North America", "CAN": "North America", "MEX": "North America", "GTM": "North America",
    "BLZ": "North America", "SLV": "North America", "HND": "North America", "NIC": "North America",
    "CRI": "North America", "PAN": "North America", "CUB": "North America", "JAM": "North America",
    "HTI": "North America", "DOM": "North America", "BHS": "North America", "BRB": "North America",
    "TTO": "North America", "PRI": "North America", "GLP": "North America", "MTQ": "North America",
    "BRA": "South America", "ARG": "South America", "COL": "South America", "PER": "South America",
    "CHL": "South America", "ECU": "South America", "VEN": "South America", "BOL": "South America",
    "PRY": "South America", "URY": "South America", "GUY": "South America", "SUR": "South America",
    # Asia
    "CHN": "Asia", "IND": "Asia", "JPN": "Asia", "KOR": "Asia", "SGP": "Asia", "THA": "Asia",
    "VNM": "Asia", "MYS": "Asia", "IDN": "Asia", "PHL": "Asia", "TWN": "Asia", "HKG": "Asia",
    "ISR": "Asia", "SAU": "Asia", "ARE": "Asia", "QAT": "Asia", "TUR": "Asia", "IRN": "Asia",
    "IRQ": "Asia", "JOR": "Asia", "LBN": "Asia", "PAK": "Asia", "BGD": "Asia", "LKA": "Asia",
    # Oceania
    "AUS": "Oceania", "NZL": "Oceania", "FJI": "Oceania", "PNG": "Oceania", "NCL": "Oceania", "PYF": "Oceania",
}


def load_owid_mpox_data(
    source_url: str = OWID_MPOX_URL,
    local_path: Optional[Path | str] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Load the Our World in Data (OWID) Monkeypox dataset.
    Tries live download first (unless offline or file exists and force_refresh is False).
    Falls back gracefully to local disk cache.
    """
    target_cache = Path(local_path) if local_path else DEFAULT_LOCAL_PATH
    target_cache.parent.mkdir(parents=True, exist_ok=True)

    df: Optional[pd.DataFrame] = None

    # Step 1: Try fetching online if force_refresh or cache doesn't exist
    if force_refresh or not target_cache.exists():
        try:
            logger.info("Fetching live OWID Mpox data from %s...", source_url)
            req = urllib.request.Request(
                source_url,
                headers={"User-Agent": "MegaMpox-Surveillance-Dashboard/1.0"}
            )
            with urllib.request.urlopen(req, timeout=12) as response:
                content = response.read()
                target_cache.write_bytes(content)
                logger.info("Saved fresh dataset to %s (%d bytes)", target_cache, len(content))
        except Exception as exc:
            logger.warning("Online fetch failed (%s). Attempting local offline cache...", exc)

    # Step 2: Read from local cache or direct URL fallback
    if target_cache.exists():
        try:
            df = pd.read_csv(target_cache)
            logger.info("Loaded OWID dataset from local cache: %s (rows: %d)", target_cache, len(df))
        except Exception as exc:
            logger.error("Failed to read local cache %s: %s", target_cache, exc)

    if df is None:
        logger.info("Reading directly via pd.read_csv(%s)...", source_url)
        df = pd.read_csv(source_url)

    # Step 3: Clean and enrich data
    df = clean_and_enrich_mpox_data(df)
    return df


def clean_and_enrich_mpox_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean columns, repair missing ISO codes (Burundi -> BDI), compute CFR,
    continent mapping, and clade epidemiological context.
    """
    df = df.copy()

    # Convert dates to datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    # Fix known missing ISOs in OWID
    df.loc[(df["location"] == "Burundi") & (df["iso_code"].isna()), "iso_code"] = "BDI"

    # Fill numeric NaNs with 0.0 for calculations
    numeric_cols = [
        "total_cases", "total_deaths", "new_cases", "new_deaths",
        "new_cases_smoothed", "new_deaths_smoothed", "new_cases_per_million",
        "total_cases_per_million", "new_cases_smoothed_per_million",
        "new_deaths_per_million", "total_deaths_per_million",
        "new_deaths_smoothed_per_million"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Distinguish country rows from aggregates
    df["is_aggregate"] = (
        df["iso_code"].isna() |
        df["iso_code"].str.startswith("OWID", na=False) |
        df["location"].isin(["World", "Africa", "Asia", "Europe", "North America", "Oceania", "South America"])
    )

    # Compute Case Fatality Rate (CFR %)
    df["case_fatality_rate"] = np.where(
        df["total_cases"] > 0,
        (df["total_deaths"] / df["total_cases"]) * 100.0,
        0.0
    )
    df["case_fatality_rate"] = df["case_fatality_rate"].round(2)

    # Map continents
    df["continent"] = df["iso_code"].map(CONTINENT_MAP)
    df.loc[df["location"] == "World", "continent"] = "Global Aggregate"

    # Map Clade Epidemiological Context
    df["primary_clade"] = df["iso_code"].map(lambda x: CLADE_EPIDEMIOLOGY_MAPPING.get(x, {}).get("primary_clade", "Clade IIb (Global/Unassigned)"))
    df["risk_tier"] = df["iso_code"].map(lambda x: CLADE_EPIDEMIOLOGY_MAPPING.get(x, {}).get("risk_tier", "Standard Surveillance"))
    df["clade_notes"] = df["iso_code"].map(lambda x: CLADE_EPIDEMIOLOGY_MAPPING.get(x, {}).get("notes", "No unique clade variant annotations"))

    # Sort chronological
    df = df.sort_values(by=["location", "date"]).reset_index(drop=True)

    return df


def get_latest_country_snapshot(df: pd.DataFrame, target_date: Optional[str] = None) -> pd.DataFrame:
    """
    Returns the latest single row per country for mapping and ranking.
    Excludes aggregate entities (World, Continents) unless requested.
    """
    country_df = df[~df["is_aggregate"]].copy()

    if target_date:
        t_dt = pd.to_datetime(target_date)
        country_df = country_df[country_df["date"] <= t_dt]

    latest = country_df.groupby("iso_code", as_index=False).last()

    latest["active_status"] = np.where(
        latest["new_cases_smoothed"] > 50, "Critical Surge",
        np.where(latest["new_cases_smoothed"] > 10, "Active Surge",
        np.where(latest["new_cases_smoothed"] > 0, "Low Transmission", "Baseline / Dormant"))
    )

    latest = latest.sort_values(by="total_cases", ascending=False).reset_index(drop=True)
    latest["global_rank"] = latest.index + 1

    return latest


def get_timeline_sampled_data(df: pd.DataFrame, sample_freq: str = "W") -> pd.DataFrame:
    """
    Returns country-level data sampled by weekly or monthly intervals for fast animated geo maps.
    """
    country_df = df[~df["is_aggregate"]].copy()
    country_df["period"] = country_df["date"].dt.to_period(sample_freq).dt.to_timestamp()

    sampled = country_df.groupby(["iso_code", "period"], as_index=False).last()
    sampled["date_str"] = sampled["period"].dt.strftime("%Y-%m-%d")
    return sampled.sort_values(by=["period", "iso_code"]).reset_index(drop=True)


def get_location_timeseries(df: pd.DataFrame, location_or_iso: str) -> pd.DataFrame:
    """
    Extracts complete time-series curve for a specific country or aggregate.
    """
    match = df[
        (df["location"].str.lower() == location_or_iso.lower()) |
        (df["iso_code"].str.upper() == location_or_iso.upper())
    ].copy()

    return match.sort_values(by="date").reset_index(drop=True)
