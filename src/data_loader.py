"""
ChargeSense — cached geospatial data access.

Every expensive OpenStreetMap download happens here exactly once per city and
is then persisted to disk. Nothing downstream touches the network.

That separation is what makes the Streamlit weight sliders usable: moving a
slider recomputes a score from cached features, it never triggers a download.
"""

from __future__ import annotations

import logging
import warnings

import geopandas as gpd
import osmnx as ox
import pandas as pd

from config import (
    POI_CATEGORIES,
    RESTRICTED_LANDUSE,
    city_config,
    city_dir,
)

log = logging.getLogger("chargesense.data")

ox.settings.use_cache = True
ox.settings.log_console = False

_MEM: dict = {}


# ------------------------------------------------------------------ helpers
def _sanitize(gdf: gpd.GeoDataFrame, keep=("name",)) -> gpd.GeoDataFrame:
    """
    OSM feature frames carry list-valued columns that no file driver accepts.
    Keep geometry plus a few string columns and drop the rest.
    """
    cols = ["geometry"] + [c for c in keep if c in gdf.columns]
    out = gdf[cols].copy()
    for c in cols:
        if c != "geometry":
            out[c] = out[c].astype(str)
    return out


def to_points(gdf: gpd.GeoDataFrame, utm_epsg: int) -> gpd.GeoDataFrame:
    """
    Reduce mixed point/line/polygon features to representative points.

    Centroids are taken in a projected CRS (metres), not in degrees — taking a
    centroid in EPSG:4326 is geometrically wrong and emits a warning.
    """
    if gdf is None or gdf.empty:
        return gdf

    out = gdf.copy()
    out = out[out.geometry.notna()]
    if out.empty:
        return out

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        projected = out.to_crs(epsg=utm_epsg)
        projected["geometry"] = projected.geometry.centroid
        out = projected.to_crs(epsg=4326)

    return out[out.geometry.notna()]


def points_to_frame(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Point GeoDataFrame -> plain latitude/longitude/name DataFrame."""
    if gdf is None or gdf.empty:
        return pd.DataFrame(columns=["latitude", "longitude", "name"])
    return pd.DataFrame(
        {
            "latitude": gdf.geometry.y.to_numpy(),
            "longitude": gdf.geometry.x.to_numpy(),
            "name": (
                gdf["name"].astype(str).to_numpy()
                if "name" in gdf.columns
                else ["" for _ in range(len(gdf))]
            ),
        }
    )


def _read_cached(path):
    if not path.exists():
        return None
    try:
        return gpd.read_file(path)
    except Exception as exc:  # pragma: no cover - corrupt cache
        log.warning("Could not read cache %s (%s); refetching.", path.name, exc)
        return None


def _write_cache(gdf: gpd.GeoDataFrame, path, succeeded: bool = True):
    """
    Persist a layer.

    `succeeded` distinguishes "the query ran and this city genuinely has none
    of these" from "the query failed". Only the first is cached — caching a
    failure would turn one dropped connection into a permanently empty layer
    that never retries.
    """
    if not succeeded:
        log.info("Not caching %s — the query failed, so it will retry.", path.name)
        return
    try:
        if gdf is None or gdf.empty:
            # Valid empty GeoJSON: readable, and records a real negative result.
            path.write_text('{"type":"FeatureCollection","features":[]}')
        else:
            gdf.to_file(path, driver="GeoJSON")
    except Exception as exc:  # pragma: no cover
        log.warning("Could not write cache %s (%s).", path.name, exc)


def _empty_points() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"name": []}, geometry=[], crs="EPSG:4326")


# ------------------------------------------------------------------ network
def load_graph(city: str, refresh: bool = False):
    """Drivable road graph for `city`, cached to <data>/<city>/graph.graphml."""
    key = ("graph", city)
    if not refresh and key in _MEM:
        return _MEM[key]

    path = city_dir(city) / "graph.graphml"
    if path.exists() and not refresh:
        log.info("[%s] loading road network from cache", city)
        graph = ox.load_graphml(path)
    else:
        place = city_config(city)["place"]
        log.info("[%s] downloading road network from OpenStreetMap…", city)
        graph = ox.graph_from_place(place, network_type="drive")
        ox.save_graphml(graph, path)
        log.info(
            "[%s] cached %d nodes / %d edges",
            city,
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )

    _MEM[key] = graph
    return graph


def load_gdfs(city: str, refresh: bool = False):
    """(nodes, edges) GeoDataFrames for the city road graph."""
    key = ("gdfs", city)
    if not refresh and key in _MEM:
        return _MEM[key]
    nodes, edges = ox.graph_to_gdfs(load_graph(city, refresh=refresh))
    _MEM[key] = (nodes, edges)
    return nodes, edges


# ------------------------------------------------------------------ stations
def load_stations(city: str, refresh: bool = False) -> gpd.GeoDataFrame:
    """
    Existing EV charging stations tagged in OSM, as representative points.

    This is what OSM has *tagged* — not a complete inventory of the city's
    charging infrastructure. See config.DATA_SOURCE_NOTE.
    """
    key = ("stations", city)
    if not refresh and key in _MEM:
        return _MEM[key]

    path = city_dir(city) / "stations.geojson"
    cached = None if refresh else _read_cached(path)

    if cached is None:
        place = city_config(city)["place"]
        ok = True
        try:
            log.info("[%s] querying existing charging stations…", city)
            raw = ox.features_from_place(place, {"amenity": "charging_station"})
            pts = to_points(_sanitize(raw), city_config(city)["utm_epsg"])
        except Exception as exc:
            log.warning("[%s] charging-station query failed: %s", city, exc)
            pts, ok = _empty_points(), False
        _write_cache(pts, path, succeeded=ok)
    else:
        pts = cached

    if pts is None or len(pts) == 0:
        pts = _empty_points()

    log.info("[%s] %d tagged charging stations", city, len(pts))
    _MEM[key] = pts
    return pts


# ------------------------------------------------------------------ POIs
def load_poi(city: str, category: str, refresh: bool = False) -> gpd.GeoDataFrame:
    """
    One POI layer as representative points.

    Applies the filter_tag / exclude_tag split described in config, which is
    what keeps metro stations and mainline railway stations from collapsing
    into the same set.
    """
    if category not in POI_CATEGORIES:
        raise KeyError(f"Unknown POI category {category!r}")

    key = ("poi", city, category)
    if not refresh and key in _MEM:
        return _MEM[key]

    spec = POI_CATEGORIES[category]
    path = city_dir(city) / f"poi_{category}.geojson"
    cached = None if refresh else _read_cached(path)

    if cached is None:
        place = city_config(city)["place"]
        ok = True
        try:
            log.info("[%s] querying POI layer '%s'…", city, category)
            raw = ox.features_from_place(place, spec["tags"])

            col, val = spec.get("filter_tag", (None, None))
            if col and col in raw.columns:
                raw = raw[raw[col] == val]
            elif col:
                raw = raw.iloc[0:0]  # tag absent entirely -> no members

            col, val = spec.get("exclude_tag", (None, None))
            if col and col in raw.columns:
                raw = raw[raw[col] != val]

            pts = to_points(_sanitize(raw), city_config(city)["utm_epsg"])
        except Exception as exc:
            # Logged, never silent: a swallowed failure would quietly zero a
            # whole scoring dimension.
            log.warning("[%s] POI query '%s' failed: %s", city, category, exc)
            pts, ok = _empty_points(), False
        _write_cache(pts, path, succeeded=ok)
    else:
        pts = cached

    if pts is None or len(pts) == 0:
        pts = _empty_points()

    _MEM[key] = pts
    return pts


def load_all_pois(city: str, refresh: bool = False) -> dict:
    return {c: load_poi(city, c, refresh=refresh) for c in POI_CATEGORIES}


# ------------------------------------------------------------------ land use
def load_restricted(city: str, key_name: str, refresh: bool = False) -> gpd.GeoDataFrame:
    """
    Restricted land-use POLYGONS, kept as polygons — feasibility needs
    point-in-polygon containment, not proximity.
    """
    if key_name not in RESTRICTED_LANDUSE:
        raise KeyError(f"Unknown restricted category {key_name!r}")

    key = ("restricted", city, key_name)
    if not refresh and key in _MEM:
        return _MEM[key]

    spec = RESTRICTED_LANDUSE[key_name]
    path = city_dir(city) / f"restricted_{key_name}.geojson"
    cached = None if refresh else _read_cached(path)

    if cached is None:
        place = city_config(city)["place"]
        ok = True
        try:
            log.info("[%s] querying restricted land use '%s'…", city, key_name)
            raw = ox.features_from_place(place, spec["tags"])
            raw = raw[raw.geometry.notna()]
            polys = raw[raw.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
            polys = _sanitize(polys) if len(polys) else _empty_points()
        except Exception as exc:
            log.warning("[%s] land-use query '%s' failed: %s", city, key_name, exc)
            polys, ok = _empty_points(), False
        _write_cache(polys, path, succeeded=ok)
    else:
        polys = cached

    if polys is None or len(polys) == 0:
        polys = _empty_points()

    _MEM[key] = polys
    return polys


def load_all_restricted(city: str, refresh: bool = False) -> dict:
    return {k: load_restricted(city, k, refresh=refresh) for k in RESTRICTED_LANDUSE}


# ------------------------------------------------------------------ warm-up
def prefetch_city(city: str, refresh: bool = False) -> dict:
    """
    Download and cache everything one city needs, in one go.

    Run this once per city offline, commit the resulting data/<city>/ folder,
    and the dashboard never needs the network again.
    """
    summary = {"city": city}
    graph = load_graph(city, refresh=refresh)
    summary["nodes"] = graph.number_of_nodes()
    summary["edges"] = graph.number_of_edges()
    summary["stations"] = len(load_stations(city, refresh=refresh))
    summary["poi_layers"] = {
        c: len(g) for c, g in load_all_pois(city, refresh=refresh).items()
    }
    summary["restricted_layers"] = {
        k: len(g) for k, g in load_all_restricted(city, refresh=refresh).items()
    }
    return summary


def clear_memory_cache() -> None:
    """Drop in-process caches. On-disk caches are untouched."""
    _MEM.clear()

import city_area  # boundary fallback
