"""
ChargeSense — recommendation engine.

The engine is split into an expensive half and a cheap half, and the split is
the point:

    build_scored_table(city)      slow — downloads, features, feasibility.
                                  Runs once per city, then cached.

    select_recommendations(...)   fast — applies weights, filters, ranks.
                                  Runs every time a slider moves.

Because weights are only ever applied in the cheap half, changing a weight
never triggers a download. This is what makes the dashboard usable.

Both Folium and Streamlit call these same two functions. The scoring algorithm
exists in exactly one place.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import (
    DEFAULT_CITY,
    DEFAULT_N_CLUSTERS,
    DEFAULT_N_RECOMMENDATIONS,
    DEFAULT_WEIGHTS,
    FEASIBILITY_REJECT_BELOW,
    MIN_DISTANCE_FROM_STATION_KM,
    MIN_RECOMMENDATION_SPACING_KM,
    city_dir,
)

import candidate_locations
import feasibility as feasibility_mod
from clustering import cluster_candidates
from explainability import explain_frame
from feature_engineering import build_features
from geo import select_spaced
from landmarks import primary_landmark
from scoring import compute_dimensions, validate_weights

log = logging.getLogger("chargesense.recommendation")

SCORED_FILE = "scored_candidates.csv"


# ---------------------------------------------------------------- slow half
def build_scored_table(
    city: str = DEFAULT_CITY,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    use_cache: bool = True,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Everything about a city's candidates that does not depend on weights.

    Returns one row per candidate carrying its coordinates, cluster, raw
    feature distances, feasibility verdict and the six dimension scores.
    """
    path = city_dir(city) / SCORED_FILE
    if use_cache and not refresh and path.exists():
        table = pd.read_csv(path)
        log.info("[%s] loaded %d scored candidates from cache", city, len(table))
        return table

    candidates = candidate_locations.load(city, regenerate=refresh)
    log.info("[%s] %d candidates", city, len(candidates))

    features = build_features(city, candidates, use_cache=not refresh)
    feas = feasibility_mod.assess(city, features)
    dims = compute_dimensions(features, feas)

    labels, _ = cluster_candidates(features, n_clusters=n_clusters)

    table = pd.concat(
        [features.reset_index(drop=True), feas.reset_index(drop=True), dims.reset_index(drop=True)],
        axis=1,
    )
    table["cluster"] = labels

    table.to_csv(path, index=False)
    log.info("[%s] wrote %s", city, path)
    return table


# ---------------------------------------------------------------- fast half
def select_recommendations(
    table: pd.DataFrame,
    weights: dict | None = None,
    n_recommendations: int = DEFAULT_N_RECOMMENDATIONS,
    min_spacing_km: float = MIN_RECOMMENDATION_SPACING_KM,
    min_station_km: float = MIN_DISTANCE_FROM_STATION_KM,
    reject_below_feasibility: float = FEASIBILITY_REJECT_BELOW,
) -> pd.DataFrame:
    """
    Apply weights and pick the final ranked sites.

    Pipeline, in order:
        1. drop sites too close to an existing charger (we are siting the
           NEXT station, not a competitor to one already there)
        2. drop sites whose land-use conflict is disqualifying
        3. score with the supplied weights
        4. select highest-first with a minimum separation between picks
        5. rank and explain
    """
    from scoring import apply_weights

    w = validate_weights(weights or DEFAULT_WEIGHTS)
    df = table.copy()
    started = len(df)

    if "km_to_station" in df.columns:
        near = df["km_to_station"].to_numpy(dtype=float) < float(min_station_km)
        near = near & np.isfinite(df["km_to_station"].to_numpy(dtype=float))
        if near.any():
            log.info("[filter] %d sites dropped: within %.1f km of an existing charger",
                     int(near.sum()), min_station_km)
        df = df[~near]

    if "feasibility_score" in df.columns:
        unfit = df["feasibility_score"].to_numpy(dtype=float) < float(reject_below_feasibility)
        if unfit.any():
            log.info("[filter] %d sites dropped: land-use conflict", int(unfit.sum()))
        df = df[~unfit]

    df = df.reset_index(drop=True)
    if df.empty:
        log.warning("[filter] no candidates survived filtering")
        return _empty_result()

    df["overall_score"] = apply_weights(df, w)

    chosen = select_spaced(
        df["latitude"].to_numpy(dtype=float),
        df["longitude"].to_numpy(dtype=float),
        df["overall_score"].to_numpy(dtype=float),
        min_km=min_spacing_km,
        limit=int(n_recommendations),
    )

    out = df.iloc[chosen].reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["reason"] = explain_frame(out, w)
    out["nearest_landmark"] = out.apply(primary_landmark, axis=1)

    log.info(
        "[select] %d candidates -> %d after filtering -> %d ranked sites",
        started, len(df), len(out),
    )
    return out


def _empty_result() -> pd.DataFrame:
    cols = [
        "latitude", "longitude", "cluster", "overall_score", "rank", "reason",
        "feasibility_score", "feasibility_band", "conflicts",
        "demand", "traffic_access", "poi", "coverage_gap", "road_access",
        "km_to_station", "nearest_landmark",
    ]
    return pd.DataFrame(columns=cols)


# ---------------------------------------------------------------- full run
def recommend(
    city: str = DEFAULT_CITY,
    weights: dict | None = None,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    n_recommendations: int = DEFAULT_N_RECOMMENDATIONS,
    refresh: bool = False,
    with_addresses: bool = False,
) -> pd.DataFrame:
    """End-to-end for one city. Convenience wrapper over the two halves."""
    table = build_scored_table(city, n_clusters=n_clusters, refresh=refresh)
    result = select_recommendations(
        table, weights=weights, n_recommendations=n_recommendations
    )

    if with_addresses and not result.empty:
        from geocoding import resolve_addresses, short_address

        addresses = resolve_addresses(
            city, result["latitude"], result["longitude"], use_network=True
        )
        result["address"] = [short_address(a) for a in addresses]

    return result


def attach_cached_addresses(city: str, result: pd.DataFrame) -> pd.DataFrame:
    """Fill the address column from cache only — never hits the network."""
    from geocoding import resolve_addresses, short_address

    if result.empty:
        return result
    addresses = resolve_addresses(
        city, result["latitude"], result["longitude"], use_network=False
    )
    result = result.copy()
    result["address"] = [short_address(a) for a in addresses]
    return result


# ---------------------------------------------------------------- legacy API
def recommend_locations(csv_file, n_clusters: int = 3):
    """
    Backwards-compatible entry point.

    The original signature, kept working so existing scripts and teammates'
    code do not break. It clusters the given CSV and returns
    (clustered_dataframe, cluster_centers) exactly as before.

    New code should call `recommend()` instead — this path has no scoring,
    feasibility or explanation.
    """
    data = pd.read_csv(csv_file)
    labels, centers = cluster_candidates(data, n_clusters=n_clusters)
    data = data.copy()
    data["Cluster"] = labels
    return data, centers


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = recommend()
    print(df[["rank", "latitude", "longitude", "overall_score", "reason"]].head(10).to_string(index=False))
