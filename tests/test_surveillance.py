"""
Unit tests for Mpox Surveillance and Geo Visualizations.
"""

import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

from mpox_clf.surveillance.data_loader import (
    clean_and_enrich_mpox_data,
    get_latest_country_snapshot,
    get_timeline_sampled_data,
    get_location_timeseries,
    load_owid_mpox_data,
)
from mpox_clf.surveillance.analytics import (
    calculate_global_kpis,
    get_top_hotspots,
    get_continent_summary,
    get_clade_breakdown_summary,
)
from mpox_clf.surveillance.geo_viz import (
    create_choropleth_map,
    create_bubble_geo_map,
    create_animated_timeline_map,
    create_clade_epidemiology_map,
    create_country_drilldown_chart,
    create_multi_country_comparison,
)


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Creates a realistic sample OWID dataset for testing."""
    return pd.DataFrame({
        "location": [
            "Democratic Republic of Congo", "Democratic Republic of Congo",
            "Burundi", "Burundi",
            "United States", "United States",
            "World", "World",
            "Africa", "Africa",
        ],
        "date": [
            "2024-01-01", "2024-01-08",
            "2024-01-01", "2024-01-08",
            "2024-01-01", "2024-01-08",
            "2024-01-01", "2024-01-08",
            "2024-01-01", "2024-01-08",
        ],
        "iso_code": [
            "COD", "COD",
            None, None,  # Burundi missing ISO in raw OWID
            "USA", "USA",
            None, None,
            "OWID_AFR", "OWID_AFR",
        ],
        "total_cases": [100.0, 150.0, 10.0, 25.0, 500.0, 600.0, 610.0, 775.0, 110.0, 175.0],
        "total_deaths": [5.0, 8.0, 0.0, 1.0, 2.0, 3.0, 7.0, 12.0, 5.0, 9.0],
        "new_cases": [10.0, 50.0, 5.0, 15.0, 20.0, 100.0, 35.0, 165.0, 15.0, 65.0],
        "new_deaths": [1.0, 3.0, 0.0, 1.0, 0.0, 1.0, 1.0, 5.0, 1.0, 4.0],
        "new_cases_smoothed": [1.4, 7.1, 0.7, 2.1, 2.8, 14.2, 5.0, 23.5, 2.1, 9.2],
        "new_deaths_smoothed": [0.1, 0.4, 0.0, 0.1, 0.0, 0.1, 0.1, 0.7, 0.1, 0.5],
        "new_cases_per_million": [0.1, 0.5, 0.4, 1.2, 0.06, 0.3, 0.04, 0.2, 0.01, 0.05],
        "total_cases_per_million": [1.0, 1.5, 0.8, 2.0, 1.5, 1.8, 0.07, 0.1, 0.08, 0.13],
        "new_cases_smoothed_per_million": [0.01, 0.07, 0.05, 0.17, 0.01, 0.04, 0.0, 0.0, 0.0, 0.0],
        "new_deaths_per_million": [0.01, 0.03, 0.0, 0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "total_deaths_per_million": [0.05, 0.08, 0.0, 0.08, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0],
        "new_deaths_smoothed_per_million": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    })


def test_clean_and_enrich_mpox_data(sample_raw_df: pd.DataFrame):
    enriched = clean_and_enrich_mpox_data(sample_raw_df)

    # 1. Date conversion
    assert pd.api.types.is_datetime64_any_dtype(enriched["date"])

    # 2. Burundi ISO fixed
    burundi_iso = enriched[enriched["location"] == "Burundi"]["iso_code"].unique()
    assert list(burundi_iso) == ["BDI"]

    # 3. Case Fatality Rate calculation
    cod_latest = enriched[(enriched["location"] == "Democratic Republic of Congo") & (enriched["date"] == "2024-01-08")].iloc[0]
    expected_cfr = round((8.0 / 150.0) * 100.0, 2)
    assert cod_latest["case_fatality_rate"] == expected_cfr

    # 4. Aggregates identified correctly
    world_row = enriched[enriched["location"] == "World"].iloc[0]
    assert bool(world_row["is_aggregate"]) is True

    usa_row = enriched[enriched["location"] == "United States"].iloc[0]
    assert bool(usa_row["is_aggregate"]) is False
    assert usa_row["continent"] == "North America"

    # 5. Clade mapping
    assert "Clade I" in cod_latest["primary_clade"]
    assert "Clade IIb" in usa_row["primary_clade"]


def test_get_latest_country_snapshot(sample_raw_df: pd.DataFrame):
    enriched = clean_and_enrich_mpox_data(sample_raw_df)
    snapshot = get_latest_country_snapshot(enriched)

    # Aggregates like World and Africa should not be in country snapshot
    assert "World" not in snapshot["location"].values
    assert "Africa" not in snapshot["location"].values

    # Should contain exactly 3 countries (COD, BDI, USA)
    assert len(snapshot) == 3
    assert "global_rank" in snapshot.columns
    # Top country should be USA (600 cases)
    assert snapshot.iloc[0]["location"] == "United States"
    assert snapshot.iloc[0]["global_rank"] == 1


def test_get_timeline_sampled_data(sample_raw_df: pd.DataFrame):
    enriched = clean_and_enrich_mpox_data(sample_raw_df)
    sampled = get_timeline_sampled_data(enriched, sample_freq="W")
    assert not sampled.empty
    assert "period" in sampled.columns
    assert "date_str" in sampled.columns


def test_get_location_timeseries(sample_raw_df: pd.DataFrame):
    enriched = clean_and_enrich_mpox_data(sample_raw_df)
    ts = get_location_timeseries(enriched, "Burundi")
    assert len(ts) == 2
    assert ts.iloc[0]["total_cases"] == 10.0
    assert ts.iloc[1]["total_cases"] == 25.0


def test_calculate_global_kpis(sample_raw_df: pd.DataFrame):
    enriched = clean_and_enrich_mpox_data(sample_raw_df)
    kpis = calculate_global_kpis(enriched)

    assert kpis["total_cases"] == 775
    assert kpis["total_deaths"] == 12
    assert kpis["active_countries_count"] == 3
    assert kpis["top_country_name"] == "United States"


def test_get_top_hotspots_and_continent(sample_raw_df: pd.DataFrame):
    enriched = clean_and_enrich_mpox_data(sample_raw_df)
    snapshot = get_latest_country_snapshot(enriched)

    hotspots = get_top_hotspots(snapshot, top_n=2)
    assert len(hotspots) == 2

    continent = get_continent_summary(snapshot)
    assert "Africa" in continent["continent"].values
    assert "North America" in continent["continent"].values

    clade_sum = get_clade_breakdown_summary(snapshot)
    assert not clade_sum.empty


def test_geo_visualizations_render(sample_raw_df: pd.DataFrame):
    enriched = clean_and_enrich_mpox_data(sample_raw_df)
    snapshot = get_latest_country_snapshot(enriched)

    # 1. Choropleth Map
    fig_choro = create_choropleth_map(snapshot, metric="total_cases")
    assert isinstance(fig_choro, go.Figure)
    assert len(fig_choro.data) > 0

    # 2. Bubble Scatter Geo Map
    fig_bubble = create_bubble_geo_map(snapshot, size_metric="total_cases", color_metric="case_fatality_rate")
    assert isinstance(fig_bubble, go.Figure)

    # 3. Animated Timeline Map
    sampled = get_timeline_sampled_data(enriched, sample_freq="W")
    fig_anim = create_animated_timeline_map(sampled, metric="total_cases")
    assert isinstance(fig_anim, go.Figure)

    # 4. Clade Epidemiology Map
    fig_clade = create_clade_epidemiology_map(snapshot)
    assert isinstance(fig_clade, go.Figure)

    # 5. Country Drilldown Chart
    ts = get_location_timeseries(enriched, "COD")
    fig_drill = create_country_drilldown_chart(ts, "Democratic Republic of Congo")
    assert isinstance(fig_drill, go.Figure)

    # 6. Multi-Country Comparison
    fig_comp = create_multi_country_comparison(enriched, countries=["United States", "Burundi"])
    assert isinstance(fig_comp, go.Figure)


def test_load_owid_mpox_data_from_local_cache():
    """Verify local cached file loading works without internet."""
    local_cache = Path(__file__).resolve().parents[1] / "data" / "raw" / "owid-monkeypox-data.csv"
    if local_cache.exists():
        df = load_owid_mpox_data(local_path=local_cache, force_refresh=False)
        assert not df.empty
        assert "location" in df.columns
        assert "case_fatality_rate" in df.columns
        assert "primary_clade" in df.columns
        assert df["location"].nunique() > 100
