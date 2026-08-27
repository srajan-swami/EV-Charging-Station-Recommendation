"""
ChargeSense — spatial clustering.

K-Means groups candidate sites into regions so that recommendations spread
across the city instead of piling into one district. It is a spatial
structuring step, not a demand model, and nothing in the scoring layer depends
on it — that separation is deliberate and is the honest answer to "does
K-Means predict demand?".

City-independent by construction: it takes coordinates, not a city name.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from config import DEFAULT_N_CLUSTERS, RANDOM_STATE


def cluster_candidates(
    candidates: pd.DataFrame,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    random_state: int = RANDOM_STATE,
):
    """
    Assign each candidate a cluster label.

    Returns (labels, centers). `n_clusters` is clamped to the number of
    available points so a small candidate set cannot raise.
    """
    coords = candidates[["latitude", "longitude"]].to_numpy(dtype=float)
    if len(coords) == 0:
        return np.array([], dtype=int), np.empty((0, 2))

    k = max(1, min(int(n_clusters), len(coords)))
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(coords)
    return labels.astype(int), model.cluster_centers_


def choose_k(candidates: pd.DataFrame, k_range=range(3, 13), random_state: int = RANDOM_STATE):
    """
    Silhouette sweep over `k_range`, returning (best_k, {k: score}).

    Used to justify the cluster count rather than asserting one. Optional —
    the pipeline runs fine with the configured default.
    """
    from sklearn.metrics import silhouette_score

    coords = candidates[["latitude", "longitude"]].to_numpy(dtype=float)
    scores = {}
    for k in k_range:
        if k >= len(coords):
            break
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(coords)
        try:
            scores[k] = float(silhouette_score(coords, labels))
        except ValueError:
            continue

    if not scores:
        return DEFAULT_N_CLUSTERS, {}
    return max(scores, key=scores.get), scores
