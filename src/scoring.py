"""
ChargeSense — multi-factor scoring.

Six dimensions, each reported 0-100, combined with configurable weights into
an overall 0-100 score.

    Demand              nearby activity that generates dwell time
    Traffic / Access    road hierarchy of the nearest road  (PROXY, not volume)
    POI / Landmarks     transport interchanges and public destinations
    Coverage Gap        real kilometres to the nearest existing charger
    Site Feasibility    land-use conflict, from feasibility.py
    Road Access         degree of the nearest intersection

This module is a pure function of (features, weights). It performs no I/O and
touches no network, which is what allows a dashboard slider to recompute a
full ranking in milliseconds from a cached feature table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    COVERAGE_GAP_SATURATION_KM,
    DEFAULT_WEIGHTS,
    DIMENSION_LABELS,
    POI_SATURATION_KM,
    ROAD_DEGREE_SATURATION,
)
from geo import decay_score

# Which POI layers feed which dimension, and how much within that dimension.
# These are relative weights inside a dimension; they are normalised, so they
# need not sum to anything in particular.
DEMAND_LAYERS = {
    "mall": 1.0,
    "office": 1.0,
    "commercial": 0.9,
    "restaurant": 0.6,
    "hotel": 0.5,
}

POI_LAYERS = {
    "metro": 1.0,
    "railway": 0.8,
    "bus_station": 0.6,
    "bus_stop": 0.5,
    "hospital": 0.7,
    "university": 0.6,
    "parking": 0.8,
    "fuel": 0.5,
}


class WeightError(ValueError):
    pass


def validate_weights(weights: dict) -> dict:
    """
    Check a weight mapping and return it normalised to sum to exactly 1.0.

    Raises rather than silently correcting when a key is unknown or a weight
    is negative — a typo in a weight name should not quietly zero a dimension.
    """
    if not weights:
        raise WeightError("No weights supplied.")

    unknown = set(weights) - set(DEFAULT_WEIGHTS)
    if unknown:
        raise WeightError(
            f"Unknown scoring dimension(s): {', '.join(sorted(unknown))}. "
            f"Valid dimensions: {', '.join(DEFAULT_WEIGHTS)}"
        )

    missing = set(DEFAULT_WEIGHTS) - set(weights)
    if missing:
        raise WeightError(
            f"Missing scoring dimension(s): {', '.join(sorted(missing))}"
        )

    if any(float(v) < 0 for v in weights.values()):
        raise WeightError("Weights cannot be negative.")

    total = float(sum(float(v) for v in weights.values()))
    if total <= 0:
        raise WeightError("Weights sum to zero — at least one must be positive.")

    return {k: float(v) / total for k, v in weights.items()}


def _blend(features: pd.DataFrame, layers: dict, saturation_km: float) -> np.ndarray:
    """
    Weighted blend of proximity scores across several POI layers.

    Layers absent for this city contribute nothing and are excluded from the
    denominator, so a city missing metro data is not penalised as though every
    metro station were infinitely far away.
    """
    n = len(features)
    total = np.zeros(n, dtype=float)
    weight_sum = 0.0

    for layer, w in layers.items():
        col = f"km_to_{layer}"
        if col not in features.columns:
            continue
        km = features[col].to_numpy(dtype=float)
        if not np.isfinite(km).any():
            continue  # layer empty for this city
        total += w * decay_score(km, saturation_km)
        weight_sum += w

    if weight_sum == 0:
        return np.full(n, 50.0)  # no data either way -> neutral, not zero
    return 100.0 * total / weight_sum


def demand_score(features: pd.DataFrame) -> np.ndarray:
    return _blend(features, DEMAND_LAYERS, POI_SATURATION_KM)


def poi_score(features: pd.DataFrame) -> np.ndarray:
    return _blend(features, POI_LAYERS, POI_SATURATION_KM)


def traffic_access_score(features: pd.DataFrame) -> np.ndarray:
    """
    Road-hierarchy accessibility proxy.

    OpenStreetMap carries no traffic volume, so this is deliberately named for
    what it measures. It must never be presented as measured traffic.
    """
    if "road_class_weight" not in features.columns:
        return np.full(len(features), 50.0)
    return 100.0 * np.clip(
        features["road_class_weight"].to_numpy(dtype=float), 0.0, 1.0
    )


def road_access_score(features: pd.DataFrame) -> np.ndarray:
    """Intersection degree — how many roads actually meet at the site."""
    if "road_degree" not in features.columns:
        return np.full(len(features), 50.0)
    deg = features["road_degree"].to_numpy(dtype=float)
    return 100.0 * np.clip(deg / float(ROAD_DEGREE_SATURATION), 0.0, 1.0)


def coverage_gap_score(features: pd.DataFrame) -> np.ndarray:
    """
    Distance to the nearest existing charger, saturating.

    Saturation matters: without it the highest-scoring site in any city is
    whichever field is furthest from everything, which is exactly the failure
    mode spec 26 warns about. Past the saturation distance, extra isolation
    earns nothing.

    When a city has no tagged stations the distance is infinite and undefined
    rather than excellent, so it scores neutrally.
    """
    if "km_to_station" not in features.columns:
        return np.full(len(features), 50.0)

    km = features["km_to_station"].to_numpy(dtype=float)
    if not np.isfinite(km).any():
        return np.full(len(features), 50.0)

    score = 100.0 * np.clip(km / float(COVERAGE_GAP_SATURATION_KM), 0.0, 1.0)
    return np.where(np.isfinite(km), score, 50.0)


def compute_dimensions(features: pd.DataFrame, feasibility: pd.DataFrame | None = None) -> pd.DataFrame:
    """All six dimension scores, 0-100. Independent of weights."""
    dims = pd.DataFrame(index=features.index)
    dims["demand"] = demand_score(features)
    dims["traffic_access"] = traffic_access_score(features)
    dims["poi"] = poi_score(features)
    dims["coverage_gap"] = coverage_gap_score(features)
    dims["road_access"] = road_access_score(features)

    if feasibility is not None and "feasibility_score" in feasibility.columns:
        dims["feasibility"] = feasibility["feasibility_score"].to_numpy(dtype=float)
    else:
        dims["feasibility"] = 100.0

    return dims.clip(0.0, 100.0)


def apply_weights(dimensions: pd.DataFrame, weights: dict | None = None) -> np.ndarray:
    """
    Combine dimension scores into one 0-100 overall score.

    Cheap by design — this is what a slider re-runs.
    """
    w = validate_weights(weights or DEFAULT_WEIGHTS)
    total = np.zeros(len(dimensions), dtype=float)
    for dim, weight in w.items():
        total += weight * dimensions[dim].to_numpy(dtype=float)
    return np.clip(total, 0.0, 100.0)


def score_candidates(
    features: pd.DataFrame,
    feasibility: pd.DataFrame | None = None,
    weights: dict | None = None,
) -> pd.DataFrame:
    """Dimension columns plus `overall_score`, aligned to `features`."""
    dims = compute_dimensions(features, feasibility)
    out = dims.copy()
    out["overall_score"] = apply_weights(dims, weights)
    return out


def breakdown_rows(row) -> list[tuple[str, float]]:
    """(label, value) pairs for the popup's score breakdown, strongest first."""
    pairs = [
        (label, float(row[key]))
        for key, label in DIMENSION_LABELS.items()
        if key in row.index
    ]
    return sorted(pairs, key=lambda p: -p[1])
