"""
ChargeSense — Folium map rendering.

Builds the interactive map: road network, existing stations, ranked
recommendations, a legend and layer controls. Every recommendation marker
carries the full detail panel from spec 31.

No scoring happens here. This module renders what the engine produced — which
is what keeps Folium and Streamlit showing identical results.

OSMnx 2.x note: `ox.plot_graph_folium` was removed. Roads are drawn by handing
the edges GeoDataFrame to `folium.GeoJson`.
"""

from __future__ import annotations

import logging

import folium
import numpy as np
from folium.plugins import MarkerCluster

from config import (
    ADDRESS_UNAVAILABLE,
    DATA_SOURCE_NOTE,
    DIMENSION_LABELS,
    TRAFFIC_PROXY_NOTE,
    city_config,
)
from landmarks import DISTANCE_KIND, format_lines

log = logging.getLogger("chargesense.visualization")

# Simplifying the road geometry keeps the output file openable. Full-detail
# edges for a large city produce a page too heavy for a browser to render.
ROAD_SIMPLIFY_TOLERANCE = 0.0001
MAX_ROAD_EDGES = 20000


def _score_colour(score: float) -> str:
    if score >= 75:
        return "darkred"
    if score >= 60:
        return "red"
    if score >= 45:
        return "orange"
    return "lightred"


def _popup_html(row) -> str:
    """The spec 31 detail panel for one recommendation."""
    address = str(row.get("address", "") or "") or ADDRESS_UNAVAILABLE
    rank = int(row.get("rank", 0))
    overall = float(row.get("overall_score", 0.0))

    breakdown = "".join(
        f"<tr><td style='padding:1px 10px 1px 0;color:#555'>{label}</td>"
        f"<td style='text-align:right;font-variant-numeric:tabular-nums'>"
        f"{float(row[key]):.0f}</td></tr>"
        for key, label in DIMENSION_LABELS.items()
        if key in row.index and np.isfinite(float(row.get(key, np.nan)))
    )

    landmarks = "".join(f"<div>{line}</div>" for line in format_lines(row))

    conflicts = str(row.get("conflicts", "") or "")
    band = str(row.get("feasibility_band", "") or "")
    feas_line = (
        f"&#10007; {conflicts}" if conflicts else "&#10003; No major land-use conflict"
    )

    km_station = row.get("km_to_station", np.inf)
    try:
        km_station = float(km_station)
        station_line = (
            f"{km_station:.1f} km" if np.isfinite(km_station) else "none mapped nearby"
        )
    except (TypeError, ValueError):
        station_line = "unknown"

    return f"""
    <div style="font-family:system-ui,sans-serif;font-size:12px;width:290px;line-height:1.45">
      <div style="font-weight:700;font-size:13px;border-bottom:2px solid #222;padding-bottom:4px;margin-bottom:6px">
        #{rank} &middot; RECOMMENDED CHARGING SITE
      </div>

      <div style="color:#555">Address</div>
      <div style="margin-bottom:5px">{address}</div>

      <div style="color:#555">Coordinates</div>
      <div style="margin-bottom:6px;font-variant-numeric:tabular-nums">
        {float(row['latitude']):.6f}, {float(row['longitude']):.6f}
      </div>

      <div style="background:#f2f4f3;padding:5px 7px;margin-bottom:6px">
        <b>Overall Score: {overall:.1f} / 100</b>
      </div>

      <div style="color:#555;margin-bottom:2px">Score Breakdown</div>
      <table style="width:100%;font-size:11px;margin-bottom:6px">{breakdown}</table>

      <div style="color:#555;margin-bottom:2px">
        Nearby Landmarks <span style="font-size:10px">({DISTANCE_KIND})</span>
      </div>
      <div style="font-size:11px;margin-bottom:6px">{landmarks}</div>

      <div style="color:#555;margin-bottom:2px">Nearest Existing Charger</div>
      <div style="font-size:11px;margin-bottom:6px">{station_line}</div>

      <div style="color:#555;margin-bottom:2px">Site Feasibility &mdash; {band}</div>
      <div style="font-size:11px;margin-bottom:6px">{feas_line}</div>

      <div style="color:#555;margin-bottom:2px">Why Recommended</div>
      <div style="font-size:11px">{row.get('reason', '')}</div>

      <div style="font-size:9px;color:#888;margin-top:7px;border-top:1px solid #ddd;padding-top:4px">
        {TRAFFIC_PROXY_NOTE}
      </div>
    </div>
    """


def _legend_html() -> str:
    return f"""
    <div style="position:fixed;bottom:22px;left:22px;z-index:9999;
                background:rgba(255,255,255,.95);border:1px solid #bbb;
                padding:10px 13px;font-family:system-ui,sans-serif;font-size:12px;
                line-height:1.6;max-width:265px;box-shadow:0 2px 8px rgba(0,0,0,.15)">
      <div style="font-weight:700;margin-bottom:5px">ChargeSense</div>
      <div><span style="color:green;font-size:15px">&#9679;</span> Existing charging station (OSM-tagged)</div>
      <div><span style="color:darkred;font-size:15px">&#9679;</span> Recommended site &mdash; ranked</div>
      <div><span style="color:#777;font-size:15px">&#9473;</span> Road network</div>
      <div style="font-size:10px;color:#666;margin-top:6px;border-top:1px solid #ddd;padding-top:5px">
        {DATA_SOURCE_NOTE}
      </div>
    </div>
    """


def build_map(city: str, recommendations, stations=None, edges=None, show_roads: bool = True):
    """Assemble the full Folium map for one city."""
    cfg = city_config(city)
    fmap = folium.Map(
        location=[cfg["latitude"], cfg["longitude"]],
        zoom_start=11,
        tiles="cartodbpositron",
    )

    # --- roads
    if show_roads and edges is not None and len(edges):
        try:
            subset = edges
            if len(subset) > MAX_ROAD_EDGES:
                log.info("[%s] simplifying %d edges for display", city, len(subset))
                subset = subset.sample(MAX_ROAD_EDGES, random_state=42)
            geom = subset[["geometry"]].copy()
            geom["geometry"] = geom.geometry.simplify(ROAD_SIMPLIFY_TOLERANCE)
            roads = folium.FeatureGroup(name="Road network", show=True)
            folium.GeoJson(
                geom.to_json(),
                style_function=lambda _: {"color": "#888", "weight": 0.7, "opacity": 0.55},
            ).add_to(roads)
            roads.add_to(fmap)
        except Exception as exc:
            log.warning("[%s] could not render roads: %s", city, exc)

    # --- existing stations
    if stations is not None and len(stations):
        group = folium.FeatureGroup(name="Existing charging stations", show=True)
        for _, row in stations.iterrows():
            point = row.geometry
            if point is None or point.is_empty:
                continue
            name = str(row.get("name", "") or "").strip()
            folium.Marker(
                [point.y, point.x],
                popup=folium.Popup(
                    f"<b>Existing charging station</b><br>{name or 'Unnamed'}"
                    "<br><span style='font-size:10px;color:#777'>Source: OpenStreetMap</span>",
                    max_width=240,
                ),
                icon=folium.Icon(color="green", icon="bolt", prefix="fa"),
            ).add_to(group)
        group.add_to(fmap)

    # --- recommendations
    if recommendations is not None and len(recommendations):
        group = folium.FeatureGroup(name="Recommended sites", show=True)
        for _, row in recommendations.iterrows():
            folium.Marker(
                [float(row["latitude"]), float(row["longitude"])],
                popup=folium.Popup(_popup_html(row), max_width=320),
                tooltip=f"#{int(row['rank'])} — {float(row['overall_score']):.1f}/100",
                icon=folium.Icon(
                    color=_score_colour(float(row["overall_score"])),
                    icon="bolt",
                    prefix="fa",
                ),
            ).add_to(group)
        group.add_to(fmap)

        # --- cluster view as a separate toggleable layer
        try:
            clustered = folium.FeatureGroup(name="Recommended sites (clustered)", show=False)
            mc = MarkerCluster().add_to(clustered)
            for _, row in recommendations.iterrows():
                folium.Marker(
                    [float(row["latitude"]), float(row["longitude"])],
                    tooltip=f"#{int(row['rank'])}",
                ).add_to(mc)
            clustered.add_to(fmap)
        except Exception as exc:
            log.debug("cluster layer unavailable: %s", exc)

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.get_root().html.add_child(folium.Element(_legend_html()))
    return fmap


def save_map(fmap, path) -> str:
    fmap.save(str(path))
    log.info("map written to %s", path)
    return str(path)
