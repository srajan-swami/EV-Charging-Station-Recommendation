"""
ChargeSense — feature computation.

Turns candidate coordinates into a table of measured quantities: real
kilometre distances to every POI layer, distance to the nearest existing
charging station, the road class of the nearest road, and the degree of the
nearest intersection.

Two rules shape this module:

1. Raw kilometres are kept, not just derived scores. The popup has to show
   "Metro Station 0.4 km", and you cannot recover 0.4 km from 1/(1+d).
2. Nothing here applies a weight. Features are facts about a place; weights
   are a policy choice. Keeping them apart is what lets a slider recompute a
   score without recomputing a feature.

The result is cached per city, so this runs once and the dashboard reads it.
"""

from __future__ import annotations

import logging

import numpy as np
import osmnx as ox
import pandas as pd

from config import (
    POI_CATEGORIES,
    ROAD_CLASS_DEFAULT,
    ROAD_CLASS_WEIGHTS,
    city_dir,
)
from data_loader import load_all_pois, load_gdfs, load_graph, load_stations
from geo import build_tree, nearest_distance_km

log = logging.getLogger("chargesense.features")

FEATURE_FILE = "features.csv"


def _first_road_class(value) -> str:
    """OSM `highway` can be a list when a way carries several classes."""
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def road_context(city: str, lat, lon) -> pd.DataFrame:
    """
    Road class of the nearest road, and degree of the nearest intersection.

    Both come from the graph that is already loaded — no extra download. This
    replaces the old approach of querying every `highway` feature in the city,
    which pulled an enormous result set and then collapsed it to one number
    identical for every candidate.
    """
    graph = load_graph(city)
    _, edges = load_gdfs(city)

    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    # --- nearest intersection -> degree (real road connectivity)
    try:
        osmids = ox.distance.nearest_nodes(graph, X=lon, Y=lat)
        osmids = np.atleast_1d(osmids)
        degrees = np.array([graph.degree(int(n)) for n in osmids], dtype=float)
    except Exception as exc:
        log.warning("[%s] nearest_nodes failed (%s); degree defaults to 0", city, exc)
        osmids = np.full(len(lat), -1, dtype=np.int64)
        degrees = np.zeros(len(lat), dtype=float)

    # --- nearest road -> hierarchy class (accessibility proxy)
    classes = np.array([""] * len(lat), dtype=object)
    try:
        u, v, k = ox.distance.nearest_edges(graph, X=lon, Y=lat)
        for i, triple in enumerate(zip(np.atleast_1d(u), np.atleast_1d(v), np.atleast_1d(k))):
            try:
                classes[i] = _first_road_class(edges.loc[triple, "highway"])
            except Exception:
                classes[i] = ""
    except Exception as exc:
        log.warning("[%s] nearest_edges failed (%s); road class defaults", city, exc)

    weights = np.array(
        [ROAD_CLASS_WEIGHTS.get(c, ROAD_CLASS_DEFAULT) for c in classes], dtype=float
    )

    return pd.DataFrame(
        {
            "osmid": np.asarray(osmids, dtype=np.int64),
            "road_degree": degrees,
            "road_class": classes.astype(str),
            "road_class_weight": weights,
        }
    )


def poi_distances(city: str, lat, lon) -> pd.DataFrame:
    """Straight-line kilometres from each candidate to the nearest POI of each type."""
    pois = load_all_pois(city)
    out = {}
    for category in POI_CATEGORIES:
        layer = pois.get(category)
        if layer is None or len(layer) == 0:
            log.info("[%s] POI layer '%s' is empty for this city", city, category)
            out[f"km_to_{category}"] = np.full(len(lat), np.inf)
            continue
        tree = build_tree(layer.geometry.y.to_numpy(), layer.geometry.x.to_numpy())
        out[f"km_to_{category}"] = nearest_distance_km(tree, lat, lon)
    return pd.DataFrame(out)


def station_distance(city: str, lat, lon) -> np.ndarray:
    """
    Kilometres to the nearest existing charging station.

    This is the real coverage-gap measurement. It replaces the old
    `DistanceScore`, which measured distance to a K-Means cluster centre while
    the popup described it as separation from existing stations.
    """
    stations = load_stations(city)
    if len(stations) == 0:
        log.warning(
            "[%s] no tagged charging stations; coverage gap is undefined "
            "and will score neutrally",
            city,
        )
        return np.full(len(lat), np.inf)
    tree = build_tree(stations.geometry.y.to_numpy(), stations.geometry.x.to_numpy())
    return nearest_distance_km(tree, lat, lon)


def build_features(city: str, candidates: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
    """
    Full feature table for a city's candidates.

    Cached to data/<city>/features.csv keyed on candidate count, so a rerun
    with the same candidate set is instant.
    """
    path = city_dir(city) / FEATURE_FILE

    if use_cache and path.exists():
        cached = pd.read_csv(path)
        if len(cached) == len(candidates):
            log.info("[%s] loaded %d cached feature rows", city, len(cached))
            return cached
        log.info("[%s] candidate count changed; recomputing features", city)

    lat = candidates["latitude"].to_numpy(dtype=float)
    lon = candidates["longitude"].to_numpy(dtype=float)

    log.info("[%s] computing features for %d candidates…", city, len(lat))
    features = pd.DataFrame({"latitude": lat, "longitude": lon})
    features = pd.concat(
        [features, road_context(city, lat, lon), poi_distances(city, lat, lon)],
        axis=1,
    )
    features["km_to_station"] = station_distance(city, lat, lon)

    features.to_csv(path, index=False)
    log.info("[%s] wrote %s", city, path)
    return features
