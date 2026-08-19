"""
Epidemiological Analytics, Global KPIs & Outbreak Hotspot Ranking.

Calculates key indicators, 7-day velocity deltas, regional aggregates,
and clade-specific surveillance metrics.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


def calculate_global_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes high-level global KPIs:
    - Total Cumulative Cases
    - Total Cumulative Deaths
    - Global Case Fatality Rate (CFR %)
    - 7-Day Smoothed New Cases
    - Active Outbreak Countries (countries with new cases in past 14 days)
    - Epicenter Region & Highest Impact Country
    """
    # Prefer World row if present, otherwise sum country snapshots
    world_rows = df[df["location"] == "World"].sort_values(by="date")

    if not world_rows.empty:
        latest_world = world_rows.iloc[-1]
        prior_week_world = world_rows.iloc[-8] if len(world_rows) >= 8 else world_rows.iloc[0]

        total_cases = int(latest_world["total_cases"])
        total_deaths = int(latest_world["total_deaths"])
        new_cases_7d = float(latest_world["new_cases_smoothed"])
        new_deaths_7d = float(latest_world["new_deaths_smoothed"])
        cfr = float(latest_world["case_fatality_rate"])

        delta_cases_7d = total_cases - int(prior_week_world["total_cases"])
        delta_deaths_7d = total_deaths - int(prior_week_world["total_deaths"])
        last_date = latest_world["date_str"]
    else:
        # Fallback: aggregate from latest country snapshot
        country_df = df[~df["is_aggregate"]].groupby("iso_code").last().reset_index()
        total_cases = int(country_df["total_cases"].sum())
        total_deaths = int(country_df["total_deaths"].sum())
        new_cases_7d = float(country_df["new_cases_smoothed"].sum())
        new_deaths_7d = float(country_df["new_deaths_smoothed"].sum())
        cfr = round((total_deaths / total_cases * 100.0), 2) if total_cases > 0 else 0.0
        delta_cases_7d = int(new_cases_7d * 7)
        delta_deaths_7d = int(new_deaths_7d * 7)
        last_date = df["date"].max().strftime("%Y-%m-%d")

    # Count countries with active cases
    country_snapshot = df[~df["is_aggregate"]].groupby("iso_code").last().reset_index()
    active_countries_count = int((country_snapshot["new_cases_smoothed"] > 0).sum())
    total_reporting_countries = len(country_snapshot)

    # Top country by total cases
    top_country = country_snapshot.sort_values(by="total_cases", ascending=False).iloc[0] if not country_snapshot.empty else None
    top_country_name = top_country["location"] if top_country is not None else "N/A"
    top_country_cases = int(top_country["total_cases"]) if top_country is not None else 0

    return {
        "total_cases": total_cases,
        "total_deaths": total_deaths,
        "cfr": cfr,
        "new_cases_7d": new_cases_7d,
        "new_deaths_7d": new_deaths_7d,
        "delta_cases_7d": delta_cases_7d,
        "delta_deaths_7d": delta_deaths_7d,
        "active_countries_count": active_countries_count,
        "total_reporting_countries": total_reporting_countries,
        "top_country_name": top_country_name,
        "top_country_cases": top_country_cases,
        "last_reported_date": last_date,
    }


def get_top_hotspots(df_snapshot: pd.DataFrame, top_n: int = 10, sort_by: str = "total_cases") -> pd.DataFrame:
    """
    Returns top N affected countries sorted by chosen metric.
    """
    df = df_snapshot[~df_snapshot["is_aggregate"]].copy()
    sorted_df = df.sort_values(by=sort_by, ascending=False).head(top_n).reset_index(drop=True)
    return sorted_df[[
        "global_rank", "location", "iso_code", "continent", "total_cases",
        "total_deaths", "case_fatality_rate", "new_cases_smoothed",
        "total_cases_per_million", "primary_clade", "risk_tier"
    ]]


def get_continent_summary(df_snapshot: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates aggregated figures grouped by continent.
    """
    df = df_snapshot[~df_snapshot["is_aggregate"]].copy()
    grouped = df.groupby("continent").agg(
        total_cases=("total_cases", "sum"),
        total_deaths=("total_deaths", "sum"),
        active_smoothed_cases=("new_cases_smoothed", "sum"),
        countries_affected=("location", "count"),
    ).reset_index()

    grouped["case_fatality_rate"] = np.where(
        grouped["total_cases"] > 0,
        (grouped["total_deaths"] / grouped["total_cases"]) * 100.0,
        0.0
    ).round(2)

    return grouped.sort_values(by="total_cases", ascending=False).reset_index(drop=True)


def get_clade_breakdown_summary(df_snapshot: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates metrics by primary clade profile.
    """
    df = df_snapshot[~df_snapshot["is_aggregate"]].copy()
    grouped = df.groupby("primary_clade").agg(
        total_cases=("total_cases", "sum"),
        total_deaths=("total_deaths", "sum"),
        countries=("location", "count"),
    ).reset_index()

    grouped["case_fatality_rate"] = np.where(
        grouped["total_cases"] > 0,
        (grouped["total_deaths"] / grouped["total_cases"]) * 100.0,
        0.0
    ).round(2)

    return grouped.sort_values(by="total_cases", ascending=False).reset_index(drop=True)
