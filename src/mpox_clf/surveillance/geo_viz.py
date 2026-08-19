"""
Interactive Geographic Visualizations & Rich Hover Cards for Mpox Surveillance.

Uses Plotly Express and Graph Objects to produce high-contrast, responsive,
publication-grade choropleth maps, proportional bubble maps, and animated timelines.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


COLOR_PALETTES = {
    "flame": ["#0f172a", "#0284c7", "#f59e0b", "#ef4444", "#991b1b"],
    "turbo": "Turbo",
    "plasma": "Plasma",
    "viridis": "Viridis",
    "ylorrd": "YlOrRd",
    "reds": "Reds",
    "purples": "Purples",
    "clade_categorical": {
        "Clade Ia / Ib": "#dc2626",
        "Clade Ib": "#e11d48",
        "Clade Ia": "#ea580c",
        "Clade I / IIa": "#d97706",
        "Clade IIa / IIb": "#2563eb",
        "Clade IIa": "#0284c7",
        "Clade IIb": "#7c3aed",
        "Clade IIb (Global/Unassigned)": "#64748b",
    }
}

SCOPE_MAPPING = {
    "Global": "world",
    "Africa (Epicenter)": "africa",
    "Europe": "europe",
    "North America": "north america",
    "South America": "south america",
    "Asia": "asia",
}

METRIC_LABELS = {
    "total_cases": "Total Cumulative Cases",
    "total_deaths": "Total Cumulative Deaths",
    "new_cases_smoothed": "7-Day Smoothed Daily Cases",
    "new_deaths_smoothed": "7-Day Smoothed Daily Deaths",
    "total_cases_per_million": "Cases per Million Population",
    "new_cases_per_million": "New Cases per Million",
    "case_fatality_rate": "Case Fatality Rate (CFR %)",
}


def build_custom_hover_template(metric: str) -> str:
    """
    Builds a rich, beautifully structured HTML hovercard template for Plotly geo layers.
    """
    metric_title = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
    
    template = (
        "<b><span style='font-size:15px; color:#38bdf8;'>%{customdata[0]} (%{customdata[1]})</span></b><br>"
        "<b>Global Rank:</b> #%{customdata[2]}<br>"
        "----------------------------------------<br>"
        f"<b><span style='color:#f59e0b;'>{metric_title}:</span></b> %{{customdata[3]:,.2f}}<br>"
        "<b>Total Confirmed Cases:</b> %{customdata[4]:,}<br>"
        "<b>Total Fatalities:</b> %{customdata[5]:,}<br>"
        "<b>Case Fatality Rate (CFR):</b> %{customdata[6]:.2f}%<br>"
        "<b>7-Day Smoothed Cases:</b> %{customdata[7]:,.1f} / day<br>"
        "<b>Cases / Million:</b> %{customdata[8]:,.1f}<br>"
        "----------------------------------------<br>"
        "<b>Surveillance Status:</b> %{customdata[9]}<br>"
        "<b>Dominant Clade Profile:</b> <span style='color:#ec4899; font-weight:bold;'>%{customdata[10]}</span><br>"
        "<b>Epidemiology Notes:</b> <i>%{customdata[11]}</i>"
        "<extra></extra>"
    )
    return template


def prepare_customdata_array(df: pd.DataFrame, metric: str) -> np.ndarray:
    """
    Prepares customdata matrix matching build_custom_hover_template indexes.
    """
    return np.column_stack([
        df["location"].astype(str),                         # 0: Country name
        df["iso_code"].astype(str),                         # 1: ISO-3
        df.get("global_rank", pd.Series(range(1, len(df)+1))), # 2: Rank
        df[metric].fillna(0.0),                             # 3: Active metric
        df["total_cases"].fillna(0).astype(int),            # 4: Total cases
        df["total_deaths"].fillna(0).astype(int),           # 5: Total deaths
        df["case_fatality_rate"].fillna(0.0),               # 6: CFR %
        df["new_cases_smoothed"].fillna(0.0),               # 7: Smoothed new cases
        df["total_cases_per_million"].fillna(0.0),          # 8: Cases per million
        df.get("active_status", pd.Series(["Active"] * len(df))), # 9: Status
        df.get("primary_clade", pd.Series(["Clade IIb"] * len(df))), # 10: Clade
        df.get("clade_notes", pd.Series(["Standard surveillance"] * len(df))), # 11: Notes
    ])


def create_choropleth_map(
    df_snapshot: pd.DataFrame,
    metric: str = "total_cases",
    palette: str = "Turbo",
    projection: str = "natural earth",
    scope: str = "world",
    log_scale: bool = False,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Renders an interactive publication-quality choropleth map.
    """
    df = df_snapshot[~df_snapshot["is_aggregate"]].copy()

    plot_metric = metric
    if log_scale:
        df["log_metric"] = np.log10(np.maximum(df[metric], 0.1))
        plot_metric = "log_metric"

    custom_data = prepare_customdata_array(df, metric)

    fig = go.Figure(
        data=go.Choropleth(
            locations=df["iso_code"],
            locationmode="ISO-3",
            z=df[plot_metric],
            colorscale=palette,
            marker_line_color="#334155",
            marker_line_width=0.6,
            customdata=custom_data,
            hovertemplate=build_custom_hover_template(metric),
            colorbar=dict(
                title=dict(text=METRIC_LABELS.get(metric, metric), font=dict(color="#e2e8f0", size=12)),
                tickfont=dict(color="#cbd5e1", size=10),
                thickness=14,
                len=0.75,
                bgcolor="rgba(15,23,42,0.85)",
                bordercolor="#475569",
                borderwidth=1,
            ),
        )
    )

    fig.update_geos(
        showframe=False,
        showcoastlines=True,
        coastlinecolor="#64748b",
        coastlinewidth=0.8,
        showcountries=True,
        countrycolor="#475569",
        countrywidth=0.5,
        showocean=True,
        oceancolor="#0f172a",
        showland=True,
        landcolor="#1e293b",
        showlakes=True,
        lakecolor="#0f172a",
        projection_type=projection,
        scope=scope,
        bgcolor="#0b0f19",
    )

    metric_name = METRIC_LABELS.get(metric, metric)
    fig.update_layout(
        title=dict(
            text=title or f"Global Mpox Geographic Distribution — <b>{metric_name}</b>",
            font=dict(color="#f8fafc", size=16, family="Segoe UI, Roboto, Helvetica"),
            x=0.02,
            y=0.96,
        ),
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        margin=dict(l=10, r=10, t=50, b=10),
        height=580,
        hoverlabel=dict(
            bgcolor="#0f172a",
            bordercolor="#38bdf8",
            font=dict(color="#f8fafc", size=12, family="Segoe UI, Roboto, monospace"),
        ),
    )

    return fig


def create_bubble_geo_map(
    df_snapshot: pd.DataFrame,
    size_metric: str = "total_cases",
    color_metric: str = "case_fatality_rate",
    projection: str = "natural earth",
    scope: str = "world",
    max_bubble_size: int = 40,
) -> go.Figure:
    """
    Renders a proportional symbol / bubble map with dual metric mapping:
    - Bubble Radius ~ size_metric (e.g. Total Cases)
    - Bubble Hue ~ color_metric (e.g. Case Fatality Rate % or Smoothed New Cases)
    """
    df = df_snapshot[~df_snapshot["is_aggregate"] & (df_snapshot[size_metric] > 0)].copy()

    custom_data = prepare_customdata_array(df, size_metric)

    # Scale sizing smoothly
    raw_vals = df[size_metric].values
    sqrt_vals = np.sqrt(raw_vals)
    max_sqrt = sqrt_vals.max() if len(sqrt_vals) > 0 and sqrt_vals.max() > 0 else 1.0
    normalized_sizes = (sqrt_vals / max_sqrt) * max_bubble_size + 4

    fig = go.Figure(
        data=go.Scattergeo(
            locations=df["iso_code"],
            locationmode="ISO-3",
            text=df["location"],
            customdata=custom_data,
            hovertemplate=build_custom_hover_template(size_metric),
            marker=dict(
                size=normalized_sizes,
                color=df[color_metric],
                colorscale="YlOrRd",
                showscale=True,
                colorbar=dict(
                    title=dict(text=METRIC_LABELS.get(color_metric, color_metric), font=dict(color="#e2e8f0", size=11)),
                    tickfont=dict(color="#cbd5e1", size=10),
                    thickness=12,
                    len=0.7,
                    bgcolor="rgba(15,23,42,0.85)",
                    bordercolor="#475569",
                ),
                line=dict(width=1.2, color="#ffffff"),
                opacity=0.88,
            ),
        )
    )

    fig.update_geos(
        showframe=False,
        showcoastlines=True,
        coastlinecolor="#64748b",
        showcountries=True,
        countrycolor="#334155",
        showocean=True,
        oceancolor="#0f172a",
        showland=True,
        landcolor="#1e293b",
        showlakes=True,
        lakecolor="#0f172a",
        projection_type=projection,
        scope=scope,
        bgcolor="#0b0f19",
    )

    fig.update_layout(
        title=dict(
            text=f"Proportional Bubble Map: <b>{METRIC_LABELS.get(size_metric, size_metric)}</b> (Bubble Size) & <b>{METRIC_LABELS.get(color_metric, color_metric)}</b> (Hue)",
            font=dict(color="#f8fafc", size=15),
            x=0.02,
            y=0.96,
        ),
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        margin=dict(l=10, r=10, t=50, b=10),
        height=580,
        hoverlabel=dict(
            bgcolor="#0f172a",
            bordercolor="#f59e0b",
            font=dict(color="#f8fafc", size=12),
        ),
    )

    return fig


def create_animated_timeline_map(
    sampled_df: pd.DataFrame,
    metric: str = "total_cases",
    palette: str = "Turbo",
    projection: str = "natural earth",
    scope: str = "world",
) -> go.Figure:
    """
    Builds a timeline-animated choropleth map allowing the user to play
    or scrub through the global Mpox outbreak waves from 2022 to present.
    """
    df = sampled_df[~sampled_df["is_aggregate"]].copy()

    # Pre-calculate global range for stable color scaling
    max_val = df[metric].quantile(0.98) if len(df) > 0 else 1000
    if max_val <= 0:
        max_val = 100

    fig = px.choropleth(
        df,
        locations="iso_code",
        locationmode="ISO-3",
        color=metric,
        hover_name="location",
        animation_frame="date_str",
        color_continuous_scale=palette,
        range_color=[0, max_val],
        projection=projection,
        scope=scope,
        labels={metric: METRIC_LABELS.get(metric, metric)},
    )

    fig.update_geos(
        showframe=False,
        showcoastlines=True,
        coastlinecolor="#64748b",
        showcountries=True,
        countrycolor="#475569",
        showocean=True,
        oceancolor="#0f172a",
        showland=True,
        landcolor="#1e293b",
        showlakes=True,
        lakecolor="#0f172a",
        bgcolor="#0b0f19",
    )

    fig.update_layout(
        title=dict(
            text=f"Spatiotemporal Evolution of Mpox Outbreak — <b>{METRIC_LABELS.get(metric, metric)}</b>",
            font=dict(color="#f8fafc", size=15),
            x=0.02,
        ),
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        margin=dict(l=10, r=10, t=50, b=10),
        height=600,
        font=dict(color="#cbd5e1"),
    )

    # Style animation slider
    if fig.layout.sliders:
        fig.layout.sliders[0].currentvalue = dict(
            prefix="Outbreak Date: ",
            font=dict(color="#38bdf8", size=13),
            visible=True,
        )
        fig.layout.sliders[0].font = dict(color="#cbd5e1", size=10)

    return fig


def create_clade_epidemiology_map(
    df_snapshot: pd.DataFrame,
    projection: str = "natural earth",
    scope: str = "world",
) -> go.Figure:
    """
    Renders an informative categorical choropleth showing known Clade I (Ia/Ib),
    Clade IIa, and Clade IIb geographic endemicity & outbreak zones.
    """
    df = df_snapshot[~df_snapshot["is_aggregate"]].copy()

    fig = px.choropleth(
        df,
        locations="iso_code",
        locationmode="ISO-3",
        color="primary_clade",
        hover_name="location",
        hover_data={
            "iso_code": True,
            "total_cases": ":,",
            "total_deaths": ":,",
            "risk_tier": True,
            "clade_notes": True,
            "primary_clade": False,
        },
        color_discrete_map=COLOR_PALETTES["clade_categorical"],
        projection=projection,
        scope=scope,
    )

    fig.update_geos(
        showframe=False,
        showcoastlines=True,
        coastlinecolor="#64748b",
        showcountries=True,
        countrycolor="#475569",
        showocean=True,
        oceancolor="#0f172a",
        showland=True,
        landcolor="#1e293b",
        showlakes=True,
        lakecolor="#0f172a",
        bgcolor="#0b0f19",
    )

    fig.update_layout(
        title=dict(
            text="Geographic Distribution of <b>Mpox Clades (Clade I, Ib, IIa, IIb)</b> & Risk Tiers",
            font=dict(color="#f8fafc", size=15),
            x=0.02,
        ),
        legend=dict(
            title=dict(text="Clade Lineage", font=dict(color="#e2e8f0", size=12)),
            font=dict(color="#cbd5e1", size=11),
            bgcolor="rgba(15,23,42,0.8)",
            bordercolor="#334155",
            borderwidth=1,
        ),
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        margin=dict(l=10, r=10, t=50, b=10),
        height=580,
    )

    return fig


def create_country_drilldown_chart(df_country: pd.DataFrame, country_name: str) -> go.Figure:
    """
    Generates a dual-panel epidemiological chart for a single country:
    Panel 1: Daily New Cases (bars) + 7-Day Moving Average (line)
    Panel 2: Cumulative Total Cases & Cumulative Total Deaths
    """
    df = df_country.sort_values(by="date").copy()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            f"Daily Incidence & 7-Day Moving Average — {country_name}",
            f"Cumulative Epidemic Trajectory (Cases & Deaths) — {country_name}",
        ),
    )

    # Top: Daily Cases (Bar) + Smoothed (Line)
    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df["new_cases"],
            name="Daily New Cases",
            marker=dict(color="rgba(56, 189, 248, 0.4)", line=dict(color="#0284c7", width=0.5)),
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Daily Cases:</b> %{y:,}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["new_cases_smoothed"],
            mode="lines",
            name="7-Day Smoothed Trend",
            line=dict(color="#f59e0b", width=2.5),
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>7-Day Avg:</b> %{y:,.1f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Bottom: Cumulative Total Cases (Line) + Deaths (Line)
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["total_cases"],
            mode="lines",
            name="Cumulative Cases",
            line=dict(color="#38bdf8", width=2.5),
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Total Cases:</b> %{y:,}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["total_deaths"],
            mode="lines",
            name="Cumulative Deaths",
            line=dict(color="#ef4444", width=2.0, dash="dot"),
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Total Deaths:</b> %{y:,}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#111827",
        font=dict(color="#cbd5e1", size=11),
        margin=dict(l=50, r=20, t=50, b=30),
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(15,23,42,0.7)",
        ),
        hovermode="x unified",
    )

    fig.update_xaxes(showgrid=True, gridcolor="#1f2937", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#1f2937", zeroline=False)

    return fig


def create_multi_country_comparison(
    df: pd.DataFrame,
    countries: List[str],
    metric: str = "total_cases",
    log_scale: bool = False,
) -> go.Figure:
    """
    Overlays epidemic curves for multiple selected countries.
    """
    fig = go.Figure()

    palette = px.colors.qualitative.Bold

    for idx, c in enumerate(countries):
        cdf = df[df["location"] == c].sort_values(by="date")
        if cdf.empty:
            continue

        color = palette[idx % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=cdf["date"],
                y=cdf[metric],
                mode="lines",
                name=c,
                line=dict(width=2.4, color=color),
                hovertemplate=f"<b>{c}</b><br><b>Date:</b> %{{x|%Y-%m-%d}}<br><b>{METRIC_LABELS.get(metric, metric)}:</b> %{{y:,.2f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"Multi-Country Epidemic Trajectory Comparison (<b>{METRIC_LABELS.get(metric, metric)}</b>)",
            font=dict(color="#f8fafc", size=15),
        ),
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#111827",
        font=dict(color="#cbd5e1"),
        margin=dict(l=50, r=20, t=50, b=40),
        height=480,
        hovermode="x unified",
        legend=dict(
            bgcolor="rgba(15,23,42,0.8)",
            bordercolor="#334155",
            borderwidth=1,
        ),
    )

    fig.update_xaxes(showgrid=True, gridcolor="#1f2937")
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#1f2937",
        type="log" if log_scale else "linear",
    )

    return fig
