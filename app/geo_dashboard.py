"""
Global Mpox Epidemiological Surveillance & Interactive Geo-Intelligence Dashboard.

Interactive public-health operations dashboard powered by Our World in Data (OWID).
Features dynamic choropleth maps, proportional bubble maps, animated timelines,
rich hover cards, trajectory comparisons, and clade epidemiology cross-referencing.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# Ensure src/ and project root are on sys.path
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from mpox_clf.surveillance import (
    load_owid_mpox_data,
    get_latest_country_snapshot,
    get_timeline_sampled_data,
    get_location_timeseries,
    calculate_global_kpis,
    get_top_hotspots,
    get_continent_summary,
    get_clade_breakdown_summary,
    create_choropleth_map,
    create_bubble_geo_map,
    create_animated_timeline_map,
    create_clade_epidemiology_map,
    create_country_drilldown_chart,
    create_multi_country_comparison,
    METRIC_LABELS,
    SCOPE_MAPPING,
    OWID_MPOX_URL,
)


@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_surveillance_data(force_refresh: bool = False) -> pd.DataFrame:
    """Loads and caches the enriched OWID dataset."""
    local_cache = ROOT / "data" / "raw" / "owid-monkeypox-data.csv"
    return load_owid_mpox_data(
        source_url=OWID_MPOX_URL,
        local_path=local_cache,
        force_refresh=force_refresh,
    )


def render_surveillance_css():
    st.markdown(
        """
        <style>
        .geo-header {
            background: linear-gradient(135deg, #0b132b 0%, #1c2541 100%);
            border: 1px solid #3a506b;
            padding: 1.2rem 1.6rem;
            border-radius: 10px;
            margin-bottom: 1.2rem;
            box-shadow: 0 4px 14px rgba(0,0,0,0.3);
        }
        .geo-title {
            color: #48dbfb;
            font-size: 1.65rem;
            font-weight: 700;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .geo-subtitle {
            color: #cbd5e1;
            font-size: 0.88rem;
            margin-top: 6px;
            margin-bottom: 0;
        }
        .kpi-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin-bottom: 1.2rem;
        }
        .kpi-card {
            background: #111827;
            border: 1px solid #1f2937;
            border-top: 3px solid #38bdf8;
            padding: 14px 16px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.25);
        }
        .kpi-card-danger {
            border-top-color: #ef4444;
        }
        .kpi-card-warning {
            border-top-color: #f59e0b;
        }
        .kpi-card-success {
            border-top-color: #10b981;
        }
        .kpi-title {
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8;
            font-weight: 600;
            margin-bottom: 4px;
        }
        .kpi-value {
            font-size: 1.55rem;
            font-weight: 800;
            color: #f8fafc;
            line-height: 1.2;
        }
        .kpi-delta {
            font-size: 0.75rem;
            margin-top: 4px;
            color: #38bdf8;
            font-weight: 600;
        }
        .hover-card-preview {
            background: #0f172a;
            border: 1px solid #334155;
            border-left: 4px solid #38bdf8;
            padding: 12px 16px;
            border-radius: 6px;
            font-size: 0.85rem;
            color: #e2e8f0;
            margin-top: 10px;
        }
        .clade-tag {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.72rem;
            font-weight: bold;
            color: #ffffff;
            margin-right: 4px;
        }
        .tag-clade1 { background-color: #dc2626; }
        .tag-clade1b { background-color: #e11d48; }
        .tag-clade2a { background-color: #0284c7; }
        .tag-clade2b { background-color: #7c3aed; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_surveillance_dashboard():
    """Main render function for the Surveillance Dashboard."""
    render_surveillance_css()

    # Sidebar controls for refresh and data provenance
    with st.sidebar:
        st.markdown("### Surveillance Data Feed")
        refresh_data = st.button("Check & Fetch Latest OWID Feed", use_container_width=True)
        if refresh_data:
            st.cache_data.clear()

        local_file = ROOT / "data" / "raw" / "owid-monkeypox-data.csv"
        cache_status = "Cached Offline" if local_file.exists() else "Live Only"
        st.caption(f"**Data Status**: {cache_status}")
        st.caption(f"**Source**: [Our World in Data (GitHub)]({OWID_MPOX_URL})")

    # Load dataset
    with st.spinner("Ingesting and calculating epidemiological metrics..."):
        try:
            df = get_cached_surveillance_data(force_refresh=refresh_data)
        except Exception as exc:
            st.error(f"Failed to load surveillance data: {exc}")
            return

    if df.empty:
        st.warning("Surveillance dataset is empty.")
        return

    # Calculate Global KPIs
    kpis = calculate_global_kpis(df)
    latest_snapshot = get_latest_country_snapshot(df)

    # Header
    st.markdown(
        f"""
        <div class="geo-header">
            <h2 class="geo-title">Mpox Global Surveillance & Geo-Intelligence Hub</h2>
            <p class="geo-subtitle">
                Real-Time Geographic Tracking, Interactive Spatial Outbreak Maps, Clade Epidemiology & Epicurves (Data through <b>{kpis['last_reported_date']}</b>)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI Metric Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric(
            label="Global Cumulative Cases",
            value=f"{kpis['total_cases']:,}",
            delta=f"+{kpis['delta_cases_7d']:,} (7d)",
        )
    with c2:
        st.metric(
            label="Global Confirmed Fatalities",
            value=f"{kpis['total_deaths']:,}",
            delta=f"+{kpis['delta_deaths_7d']:,} (7d)",
            delta_color="inverse",
        )
    with c3:
        st.metric(
            label="Case Fatality Rate (CFR)",
            value=f"{kpis['cfr']:.2f}%",
            delta="Severity Index",
            delta_color="off",
        )
    with c4:
        st.metric(
            label="Active Transmission Hotspots",
            value=f"{kpis['active_countries_count']} / {kpis['total_reporting_countries']}",
            delta="Reporting Countries",
            delta_color="off",
        )
    with c5:
        st.metric(
            label="Top Impact Epicenter",
            value=f"{kpis['top_country_name']}",
            delta=f"{kpis['top_country_cases']:,} cases",
            delta_color="off",
        )

    st.markdown("---")

    # Map Visual Options & Filtering
    st.subheader("Interactive Spatial Outbreak Map & Live Hover Card")

    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.2, 1.2, 1.0, 1.0])

    with ctrl_col1:
        selected_metric = st.selectbox(
            "Primary Metric:",
            options=[
                "total_cases",
                "total_deaths",
                "new_cases_smoothed",
                "total_cases_per_million",
                "new_cases_per_million",
                "case_fatality_rate",
            ],
            format_func=lambda x: METRIC_LABELS.get(x, x),
            index=0,
        )

    with ctrl_col2:
        map_mode = st.selectbox(
            "Visualization Mode:",
            options=[
                "Choropleth Density Map",
                "Proportional Bubble Map",
                "Animated Timeline Evolution (2022–Present)",
                "Clade Lineage & Risk Tier Map",
            ],
            index=0,
        )

    with ctrl_col3:
        selected_region = st.selectbox(
            "Geographic Region:",
            options=list(SCOPE_MAPPING.keys()),
            index=0,
        )
        scope_val = SCOPE_MAPPING[selected_region]

    with ctrl_col4:
        projection_choice = st.selectbox(
            "Map Projection:",
            options=["natural earth", "orthographic", "mercator", "equirectangular", "robinson"],
            index=0,
        )

    # Advanced toggles
    adv_c1, adv_c2, adv_c3 = st.columns([1, 1, 2])
    with adv_c1:
        log_scale = st.checkbox("Logarithmic Scale (log10)", value=False)
    with adv_c2:
        palette_choice = st.selectbox("Color Palette:", ["Turbo", "Plasma", "Viridis", "YlOrRd", "Reds", "Purples"], index=0)

    # Render Map Based on Selected Mode
    with st.spinner("Rendering interactive spatial map with custom hovercards..."):
        if map_mode == "Choropleth Density Map":
            fig_map = create_choropleth_map(
                df_snapshot=latest_snapshot,
                metric=selected_metric,
                palette=palette_choice,
                projection=projection_choice,
                scope=scope_val,
                log_scale=log_scale,
            )
            st.plotly_chart(fig_map, use_container_width=True)

        elif map_mode == "Proportional Bubble Map":
            fig_map = create_bubble_geo_map(
                df_snapshot=latest_snapshot,
                size_metric=selected_metric,
                color_metric="case_fatality_rate" if selected_metric != "case_fatality_rate" else "new_cases_smoothed",
                projection=projection_choice,
                scope=scope_val,
            )
            st.plotly_chart(fig_map, use_container_width=True)

        elif map_mode == "Animated Timeline Evolution (2022–Present)":
            st.info("Click **Play** on the timeline below or drag the slider to watch the global transmission waves over time.")
            sampled_df = get_timeline_sampled_data(df, sample_freq="M")
            fig_map = create_animated_timeline_map(
                sampled_df=sampled_df,
                metric=selected_metric,
                palette=palette_choice,
                projection=projection_choice,
                scope=scope_val,
            )
            st.plotly_chart(fig_map, use_container_width=True)

        elif map_mode == "Clade Lineage & Risk Tier Map":
            fig_map = create_clade_epidemiology_map(
                df_snapshot=latest_snapshot,
                projection=projection_choice,
                scope=scope_val,
            )
            st.plotly_chart(fig_map, use_container_width=True)

    # Hover card explanation hint
    st.markdown(
        """
        <div class="hover-card-preview">
            <b>Interactive Hover Card Guide</b>: Hover over any country or territory above to reveal its 
            <b>detailed operational card</b> including confirmed cases, fatalities, Case Fatality Rate (CFR), 
            7-day smoothed incidence, population rate, epidemic risk tier, and dominant <b>Mpox Clade profile</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Section 2: Country Deep Dive & Epicurves
    st.subheader("Country-Specific Epidemiological Drilldown & Epicurves")

    d_col1, d_col2 = st.columns([1, 2])

    with d_col1:
        all_countries = sorted(latest_snapshot["location"].unique().tolist())
        default_index = all_countries.index("Democratic Republic of Congo") if "Democratic Republic of Congo" in all_countries else 0
        selected_country = st.selectbox(
            "Select Country for Deep-Dive Diagnostic:",
            options=all_countries,
            index=default_index,
        )

        # Country Summary Card
        c_data = latest_snapshot[latest_snapshot["location"] == selected_country].iloc[0]
        st.markdown(
            f"""
            <div style="background:#1e293b; padding:16px; border-radius:8px; border:1px solid #334155; margin-top:10px;">
                <h4 style="margin:0 0 8px 0; color:#38bdf8;">{c_data['location']} ({c_data['iso_code']})</h4>
                <p style="margin:2px 0; font-size:0.85rem;"><b>Global Rank:</b> #{c_data.get('global_rank', 'N/A')}</p>
                <p style="margin:2px 0; font-size:0.85rem;"><b>Total Confirmed Cases:</b> <span style="color:#f59e0b; font-weight:bold;">{c_data['total_cases']:,.0f}</span></p>
                <p style="margin:2px 0; font-size:0.85rem;"><b>Total Deaths:</b> <span style="color:#ef4444; font-weight:bold;">{c_data['total_deaths']:,.0f}</span></p>
                <p style="margin:2px 0; font-size:0.85rem;"><b>Case Fatality Rate:</b> {c_data['case_fatality_rate']:.2f}%</p>
                <p style="margin:2px 0; font-size:0.85rem;"><b>7-Day Smoothed Daily Cases:</b> {c_data['new_cases_smoothed']:,.1f}/day</p>
                <p style="margin:2px 0; font-size:0.85rem;"><b>Cases / Million Population:</b> {c_data['total_cases_per_million']:,.1f}</p>
                <hr style="border-color:#475569; margin:8px 0;"/>
                <p style="margin:2px 0; font-size:0.85rem;"><b>Surveillance Alert:</b> {c_data.get('active_status', 'N/A')}</p>
                <p style="margin:2px 0; font-size:0.85rem;"><b>Associated Clade:</b> <span style="color:#ec4899; font-weight:bold;">{c_data.get('primary_clade', 'N/A')}</span></p>
                <p style="margin:4px 0 0 0; font-size:0.8rem; color:#94a3b8;"><i>{c_data.get('clade_notes', '')}</i></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with d_col2:
        c_timeseries = get_location_timeseries(df, selected_country)
        if not c_timeseries.empty:
            fig_drilldown = create_country_drilldown_chart(c_timeseries, selected_country)
            st.plotly_chart(fig_drilldown, use_container_width=True)
        else:
            st.info(f"No time-series history available for {selected_country}.")

    st.markdown("---")

    # Section 3: Multi-Country Trajectory Comparison
    st.subheader("Multi-Country Outbreak Trajectory Comparison")
    comp_col1, comp_col2, comp_col3 = st.columns([2.5, 1, 1])

    with comp_col1:
        default_compare = [
            c for c in ["Democratic Republic of Congo", "Burundi", "Uganda", "United States", "Spain", "Brazil"]
            if c in all_countries
        ]
        compare_countries = st.multiselect(
            "Select Countries to Overlay on Timeline:",
            options=all_countries,
            default=default_compare,
        )

    with comp_col2:
        compare_metric = st.selectbox(
            "Comparison Metric:",
            options=["total_cases", "total_cases_per_million", "new_cases_smoothed", "total_deaths"],
            format_func=lambda x: METRIC_LABELS.get(x, x),
            index=0,
        )

    with comp_col3:
        comp_log = st.checkbox("Logarithmic Vertical Axis", value=False, key="comp_log")

    if compare_countries:
        fig_compare = create_multi_country_comparison(
            df=df,
            countries=compare_countries,
            metric=compare_metric,
            log_scale=comp_log,
        )
        st.plotly_chart(fig_compare, use_container_width=True)

    st.markdown("---")

    # Section 4: Hotspots Ranking Table & Continent Summary
    tab_table, tab_continent, tab_clade_summary = st.tabs([
        "Global Country Rankings & Hotspots Table",
        "Regional / Continent Breakdown",
        "Clade Public-Health Profile Summary",
    ])

    with tab_table:
        st.markdown("##### Outbreak Hotspots Leaderboard")
        hotspots_df = get_top_hotspots(latest_snapshot, top_n=len(latest_snapshot))

        # Format dataframe for display
        display_df = hotspots_df.rename(columns={
            "global_rank": "Rank",
            "location": "Country / Territory",
            "iso_code": "ISO-3",
            "continent": "Continent",
            "total_cases": "Total Cases",
            "total_deaths": "Total Deaths",
            "case_fatality_rate": "CFR (%)",
            "new_cases_smoothed": "7d Smoothed Cases",
            "total_cases_per_million": "Cases / 1M",
            "primary_clade": "Dominant Clade",
            "risk_tier": "Risk Tier",
        })

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Total Cases": st.column_config.NumberColumn(format="%d"),
                "Total Deaths": st.column_config.NumberColumn(format="%d"),
                "CFR (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "7d Smoothed Cases": st.column_config.NumberColumn(format="%.1f"),
                "Cases / 1M": st.column_config.NumberColumn(format="%.1f"),
            }
        )

        csv_bytes = hotspots_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Complete Surveillance Table as CSV",
            data=csv_bytes,
            file_name=f"mpox_surveillance_snapshot_{kpis['last_reported_date']}.csv",
            mime="text/csv",
        )

    with tab_continent:
        st.markdown("##### Continental Burden Overview")
        continent_df = get_continent_summary(latest_snapshot)
        st.dataframe(
            continent_df.rename(columns={
                "continent": "Continent",
                "total_cases": "Total Cases",
                "total_deaths": "Total Deaths",
                "active_smoothed_cases": "Active 7d Smoothed Cases",
                "countries_affected": "Affected Countries",
                "case_fatality_rate": "CFR (%)",
            }),
            use_container_width=True,
            hide_index=True,
        )

    with tab_clade_summary:
        st.markdown("##### Epidemiological Breakdown by Mpox Clade Profile")
        clade_df = get_clade_breakdown_summary(latest_snapshot)
        st.dataframe(
            clade_df.rename(columns={
                "primary_clade": "Clade Lineage",
                "total_cases": "Reported Cases",
                "total_deaths": "Reported Deaths",
                "countries": "Countries / Territories",
                "case_fatality_rate": "CFR (%)",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            """
            > **Epidemiological Context Note**:
            > - **Clade I (Ia / Ib)**: Endemic to the Congo Basin / Central & East Africa. Historically associated with higher case fatality (~1–10%) and severe clinical outcomes. Clade Ib has shown sustained human-to-human transmission.
            > - **Clade IIa**: Endemic to West Africa with lower mortality (~0.1–1%).
            > - **Clade IIb**: Responsible for the 2022–2023 global multi-country outbreak, driven by APOBEC3-mediated mutations.
            """
        )


def main():
    st.set_page_config(
        page_title="Mpox Global Surveillance & Geo-Intelligence Hub",
        layout="wide",
    )
    render_surveillance_dashboard()


if __name__ == "__main__":
    main()
