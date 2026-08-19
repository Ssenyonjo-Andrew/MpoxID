"""
Mpox Surveillance & Geo-Intelligence Package.
"""

from mpox_clf.surveillance.data_loader import (
    load_owid_mpox_data,
    clean_and_enrich_mpox_data,
    get_latest_country_snapshot,
    get_timeline_sampled_data,
    get_location_timeseries,
    OWID_MPOX_URL,
    CLADE_EPIDEMIOLOGY_MAPPING,
)
from mpox_clf.surveillance.geo_viz import (
    create_choropleth_map,
    create_bubble_geo_map,
    create_animated_timeline_map,
    create_clade_epidemiology_map,
    create_country_drilldown_chart,
    create_multi_country_comparison,
    METRIC_LABELS,
    SCOPE_MAPPING,
)
from mpox_clf.surveillance.analytics import (
    calculate_global_kpis,
    get_top_hotspots,
    get_continent_summary,
    get_clade_breakdown_summary,
)

__all__ = [
    "load_owid_mpox_data",
    "clean_and_enrich_mpox_data",
    "get_latest_country_snapshot",
    "get_timeline_sampled_data",
    "get_location_timeseries",
    "OWID_MPOX_URL",
    "CLADE_EPIDEMIOLOGY_MAPPING",
    "create_choropleth_map",
    "create_bubble_geo_map",
    "create_animated_timeline_map",
    "create_clade_epidemiology_map",
    "create_country_drilldown_chart",
    "create_multi_country_comparison",
    "METRIC_LABELS",
    "SCOPE_MAPPING",
    "calculate_global_kpis",
    "get_top_hotspots",
    "get_continent_summary",
    "get_clade_breakdown_summary",
]
