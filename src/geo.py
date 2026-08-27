"""
ChargeSense — shared geometry helpers.

All distances in this project are great-circle (haversine) distances in
kilometres, computed on a BallTree. Nothing here uses Euclidean distance on
raw degrees, which is wrong away from the equator and inconsistent between
latitude and longitude.

These functions are pure: no network, no OSM, no global state. They are the
part of the system that can be unit-tested directly.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import BallTree

from config import EARTH_RADIUS_KM


def to_radians(lat, lon) -> np.ndarray:
    """(lat, lon) array-likes -> radian array shaped (n, 2) for BallTree."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    return np.radians(np.column_stack([lat, lon]))


def build_tree(lat, lon) -> BallTree | None:
    """Haversine BallTree over the given points, or None if there are none."""
    if lat is None or len(lat) == 0:
        return None
    return BallTree(to_radians(lat, lon), metric="haversine")


def nearest_distance_km(tree: BallTree | None, lat, lon, fill=np.inf) -> np.ndarray:
    """
    Distance in km from each query point to the nearest point in `tree`.

    Returns `fill` for every point when the tree is empty — an absent layer
    must not silently read as "distance zero".
    """
    n = len(np.atleast_1d(lat))
    if tree is None:
        return np.full(n, fill, dtype=float)
    dist, _ = tree.query(to_radians(lat, lon), k=1)
    return dist.ravel() * EARTH_RADIUS_KM


def nearest_index_and_km(tree: BallTree | None, lat, lon):
    """As above, but also returns the index of the nearest point."""
    n = len(np.atleast_1d(lat))
    if tree is None:
        return np.full(n, -1, dtype=int), np.full(n, np.inf, dtype=float)
    dist, idx = tree.query(to_radians(lat, lon), k=1)
    return idx.ravel().astype(int), dist.ravel() * EARTH_RADIUS_KM


def pairwise_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km between two single points."""
    p1, p2 = np.radians([lat1, lon1]), np.radians([lat2, lon2])
    dlat, dlon = p2[0] - p1[0], p2[1] - p1[1]
    a = np.sin(dlat / 2) ** 2 + np.cos(p1[0]) * np.cos(p2[0]) * np.sin(dlon / 2) ** 2
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))


def thin_by_spacing(lat, lon, min_km: float, order=None) -> np.ndarray:
    """
    Greedy spatial thinning: keep a point, drop everything within `min_km`
    of it, repeat.

    `order` controls which point survives each neighbourhood. Pass an array of
    indices sorted best-first and the *best* point in each neighbourhood wins.
    Pass nothing and it is whatever order the input happened to be in — which
    makes the survivor arbitrary rather than chosen.

    Returns the indices of the kept points, in visit order.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    n = len(lat)
    if n == 0:
        return np.array([], dtype=int)

    tree = BallTree(to_radians(lat, lon), metric="haversine")
    radius = float(min_km) / EARTH_RADIUS_KM
    visit = np.arange(n) if order is None else np.asarray(order, dtype=int)

    kept, blocked = [], np.zeros(n, dtype=bool)
    for i in visit:
        if blocked[i]:
            continue
        kept.append(int(i))
        for j in tree.query_radius(to_radians(lat[i], lon[i]).reshape(1, 2), r=radius)[0]:
            if j != i:
                blocked[j] = True

    return np.array(kept, dtype=int)


def select_spaced(lat, lon, scores, min_km: float, limit: int) -> np.ndarray:
    """
    Pick the highest-scoring points subject to a minimum separation.

    Highest score first, skip anything too close to an already-selected site,
    stop at `limit`. This is spec 28's algorithm.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if len(lat) == 0:
        return np.array([], dtype=int)

    order = np.argsort(-scores, kind="stable")
    chosen: list[int] = []

    for i in order:
        if len(chosen) >= limit:
            break
        ok = True
        for j in chosen:
            if pairwise_km(lat[i], lon[i], lat[j], lon[j]) < min_km:
                ok = False
                break
        if ok:
            chosen.append(int(i))

    return np.array(chosen, dtype=int)


def decay_score(distance_km, saturation_km: float) -> np.ndarray:
    """
    Proximity -> 0..1. Distance 0 scores 1.0, `saturation_km` scores 0.0,
    anything beyond stays 0. Infinite distance (absent layer) scores 0.

    Linear rather than 1/(1+d): the shape is easier to explain to a judge and
    it reaches a true zero instead of an asymptote.
    """
    d = np.asarray(distance_km, dtype=float)
    out = 1.0 - np.clip(d / float(saturation_km), 0.0, 1.0)
    return np.where(np.isfinite(d), out, 0.0)


def normalise(values, lo=None, hi=None) -> np.ndarray:
    """
    Min-max to 0..1. A constant input maps to 0.5 rather than exploding or
    silently collapsing to zero.
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return v
    lo = float(np.nanmin(v)) if lo is None else float(lo)
    hi = float(np.nanmax(v)) if hi is None else float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.full_like(v, 0.5)
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0)
