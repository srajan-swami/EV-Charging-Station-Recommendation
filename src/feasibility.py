"""
ChargeSense — site feasibility.

The distinction this module exists to make: being NEAR a hospital is good
(people dwell there, they charge), being INSIDE the hospital's grounds is
infeasible (you cannot build on land you do not own).

So feasibility is a containment test, not a distance test. A candidate is
penalised when it falls *within* a restricted land-use polygon, and is left
completely alone when it merely sits next to one. Proximity is handled by the
POI dimension, where it counts as a positive.

Buffers are applied in a projected CRS so the tolerance is real metres.
"""

from __future__ import annotations

import logging
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

from config import (
    FEASIBILITY_BANDS,
    FEASIBILITY_START_SCORE,
    RESTRICTED_LANDUSE,
    city_config,
)
import data_loader

log = logging.getLogger("chargesense.feasibility")

# Tolerance around a restricted polygon, in metres. A site a few metres from
# the edge of a water body or runway is in practice inside it.
EDGE_TOLERANCE_M = 15.0


def feasibility_band(score: float) -> str:
    for threshold, label in FEASIBILITY_BANDS:
        if score >= threshold:
            return label
    return FEASIBILITY_BANDS[-1][1]


def assess(city: str, candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Score every candidate for land-use feasibility.

    Returns a frame with:
        feasibility_score  0-100, starts at 100 and loses configured penalties
        feasibility_band   human label for that score
        conflicts          comma-separated conflict labels, empty if none
    """
    lat = candidates["latitude"].to_numpy(dtype=float)
    lon = candidates["longitude"].to_numpy(dtype=float)
    n = len(lat)

    scores = np.full(n, float(FEASIBILITY_START_SCORE))
    conflicts: list[list[str]] = [[] for _ in range(n)]

    points = gpd.GeoDataFrame(
        {"idx": np.arange(n)},
        geometry=gpd.points_from_xy(lon, lat),
        crs="EPSG:4326",
    )

    epsg = city_config(city)["utm_epsg"]
    # Resolved through the module rather than bound at import time, so the
    # data layer can be swapped for a fixture in tests.
    layers = data_loader.load_all_restricted(city)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        points_m = points.to_crs(epsg=epsg)

        for key, layer in layers.items():
            if layer is None or len(layer) == 0:
                continue

            spec = RESTRICTED_LANDUSE[key]
            try:
                polys = layer[
                    layer.geometry.notna()
                    & layer.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
                ]
                if polys.empty:
                    continue

                polys_m = polys.to_crs(epsg=epsg).copy()
                polys_m["geometry"] = polys_m.geometry.buffer(EDGE_TOLERANCE_M)
                polys_m = polys_m[["geometry"]]

                hit = gpd.sjoin(points_m, polys_m, how="inner", predicate="within")
                for i in hit["idx"].unique():
                    i = int(i)
                    scores[i] -= float(spec["penalty"])
                    conflicts[i].append(spec["label"])

            except Exception as exc:
                log.warning("[%s] feasibility layer '%s' failed: %s", city, key, exc)

    scores = np.clip(scores, 0.0, 100.0)

    return pd.DataFrame(
        {
            "feasibility_score": scores,
            "feasibility_band": [feasibility_band(s) for s in scores],
            "conflicts": [", ".join(c) for c in conflicts],
        }
    )


def conflict_summary(row) -> str:
    """One line for the popup's Site Feasibility section."""
    conflicts = str(row.get("conflicts", "") or "")
    if not conflicts:
        return "No major land-use conflict detected"
    return f"Conflicts with {conflicts}"
