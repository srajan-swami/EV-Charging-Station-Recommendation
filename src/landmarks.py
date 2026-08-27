"""
ChargeSense — nearby landmark listing for the recommendation popup.

Distances here are STRAIGHT-LINE (great-circle), and every surface that shows
them says so. Calling a straight-line distance a road distance would be wrong
and spec 19 is explicit about it.

`network_distance_km` is provided for the cases where a true road-network
distance is worth the cost — it is far more expensive than a haversine lookup,
so it is applied to the final shortlist only, never to every candidate.
"""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np

from config import (
    LANDMARK_CATEGORIES,
    LANDMARK_MAX_DISTANCE_KM,
    MAX_LANDMARKS_SHOWN,
    POI_CATEGORIES,
)

log = logging.getLogger("chargesense.landmarks")

DISTANCE_KIND = "straight-line"
NO_LANDMARKS = "No nearby landmark found"


def nearby(row, max_items: int = MAX_LANDMARKS_SHOWN, max_km: float = LANDMARK_MAX_DISTANCE_KM):
    """
    Landmarks within `max_km` of one scored candidate, nearest first.

    Returns a list of (label, km). Empty when nothing is in range — the caller
    is expected to render NO_LANDMARKS rather than an empty block.
    """
    found = []
    for category in LANDMARK_CATEGORIES:
        km = row.get(f"km_to_{category}", np.inf)
        try:
            km = float(km)
        except (TypeError, ValueError):
            continue
        if np.isfinite(km) and km <= max_km:
            found.append((POI_CATEGORIES[category]["label"], km))

    found.sort(key=lambda p: p[1])
    return found[:max_items]


def format_lines(row, **kwargs) -> list[str]:
    """Landmark lines ready for a popup, e.g. 'Metro Station — 0.4 km'."""
    items = nearby(row, **kwargs)
    if not items:
        return [NO_LANDMARKS]
    return [f"{label} — {km:.1f} km" for label, km in items]


def primary_landmark(row) -> str:
    """Single closest landmark, for the ranked table's Landmark column."""
    items = nearby(row, max_items=1)
    if not items:
        return NO_LANDMARKS
    label, km = items[0]
    return f"{label} {km:.1f} km"


def network_distance_km(graph, origin_osmid: int, dest_lat: float, dest_lon: float):
    """
    True road-network distance between an intersection and a point.

    Returns None when no route exists or the lookup fails, so the caller can
    fall back to the straight-line figure and label it accordingly.
    """
    try:
        import osmnx as ox

        dest = ox.distance.nearest_nodes(graph, X=dest_lon, Y=dest_lat)
        metres = nx.shortest_path_length(
            graph, int(origin_osmid), int(dest), weight="length"
        )
        return float(metres) / 1000.0
    except Exception as exc:
        log.debug("network distance unavailable: %s", exc)
        return None
