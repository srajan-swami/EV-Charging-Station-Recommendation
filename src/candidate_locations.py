"""
ChargeSense — candidate site generation.

Every road intersection in the city is a possible site, because a charger has
to be reachable by road. That set is far too large to score directly, so it is
thinned to a workable number.

How the thinning picks a survivor matters. The obvious implementation keeps
whichever point it happens to encounter first and discards its neighbours,
which means that of the fifteen-odd intersections in a square kilometre, the
one that goes forward is chosen at random. `priority` fixes that: pass a
quality ordering and the best point in each neighbourhood wins instead.

Run as a script to regenerate a city's candidate file:

    python src/candidate_locations.py --city Chennai
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from config import DEFAULT_CITY, MIN_CANDIDATE_SPACING_KM, city_dir
from data_loader import load_gdfs
from geo import thin_by_spacing

log = logging.getLogger("chargesense.candidates")

CANDIDATE_FILE = "candidates.csv"
LEGACY_FILE = "sample_locations.csv"


def all_intersections(city: str) -> pd.DataFrame:
    """Every node in the drivable road graph, as latitude/longitude."""
    nodes, _ = load_gdfs(city)
    df = pd.DataFrame(
        {
            "latitude": nodes.geometry.y.to_numpy(),
            "longitude": nodes.geometry.x.to_numpy(),
            "osmid": nodes.index.to_numpy(),
        }
    )
    return df.drop_duplicates(subset=["latitude", "longitude"]).reset_index(drop=True)


def generate(
    city: str = DEFAULT_CITY,
    min_spacing_km: float = MIN_CANDIDATE_SPACING_KM,
    priority: np.ndarray | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Thin the intersection set to candidates at least `min_spacing_km` apart.

    `priority`: optional array of scores aligned to the full intersection set.
    Supplied, the highest-scoring point in each neighbourhood survives.
    Omitted, the survivor is whichever came first in the graph — arbitrary,
    and worth avoiding once a scoring pass is available.
    """
    raw = all_intersections(city)
    log.info("[%s] %d unique intersections", city, len(raw))

    order = None if priority is None else np.argsort(-np.asarray(priority, dtype=float))
    keep = thin_by_spacing(
        raw["latitude"].to_numpy(),
        raw["longitude"].to_numpy(),
        min_km=min_spacing_km,
        order=order,
    )

    out = (
        raw.iloc[keep]
        .sort_values(["latitude", "longitude"])
        .reset_index(drop=True)
    )
    log.info(
        "[%s] %d intersections -> %d candidates at %.1f km spacing",
        city, len(raw), len(out), min_spacing_km,
    )

    if save:
        path = city_dir(city) / CANDIDATE_FILE
        out.to_csv(path, index=False)
        log.info("[%s] wrote %s", city, path)

    return out


def load(city: str = DEFAULT_CITY, regenerate: bool = False) -> pd.DataFrame:
    """
    Candidates for a city, generating them if no file exists.

    Also reads the project's original `data/sample_locations.csv` when a
    per-city file has not been created yet, so existing work is not orphaned.
    """
    path = city_dir(city) / CANDIDATE_FILE
    if path.exists() and not regenerate:
        return pd.read_csv(path)

    legacy = city_dir(city).parent / LEGACY_FILE
    if legacy.exists() and not regenerate and city == DEFAULT_CITY:
        log.info("[%s] using legacy %s", city, legacy.name)
        df = pd.read_csv(legacy)
        if {"latitude", "longitude"}.issubset(df.columns):
            return df

    return generate(city)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Generate ChargeSense candidate sites.")
    parser.add_argument("--city", default=DEFAULT_CITY)
    parser.add_argument("--spacing", type=float, default=MIN_CANDIDATE_SPACING_KM)
    args = parser.parse_args()

    result = generate(args.city, min_spacing_km=args.spacing)
    print(result.head())
    print(f"\nTotal candidates: {len(result)}")
