"""
ChargeSense — engine tests.

Covers the scenarios in spec 47. Everything here runs offline: no network, no
OpenStreetMap. The geometry, scoring, feasibility and explanation layers are
pure functions of their inputs, which is exactly why they can be tested.

    python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import geo  # noqa: E402
import scoring  # noqa: E402
from clustering import cluster_candidates  # noqa: E402
from explainability import explain  # noqa: E402
from landmarks import NO_LANDMARKS, nearby, primary_landmark  # noqa: E402


# ------------------------------------------------------------------ geometry
def test_haversine_matches_known_distance():
    # Chennai Central to Guindy, ~9.5 km apart
    km = geo.pairwise_km(13.0827, 80.2707, 13.0067, 80.2206)
    assert 9.0 < km < 11.0


def test_longitude_degree_is_shorter_than_latitude_degree():
    """The exact error Euclidean-on-degrees makes. Guards against regression."""
    lat_km = geo.pairwise_km(13.0, 80.0, 14.0, 80.0)
    lon_km = geo.pairwise_km(13.0, 80.0, 13.0, 81.0)
    assert lat_km > lon_km
    assert abs(lat_km - 111.2) < 1.0


def test_empty_tree_gives_infinite_distance_not_zero():
    """An absent POI layer must not read as 'a POI is right here'."""
    d = geo.nearest_distance_km(geo.build_tree([], []), [13.0], [80.0])
    assert np.isinf(d).all()


def test_thinning_enforces_minimum_spacing():
    rng = np.random.default_rng(0)
    lat = 13.0 + rng.random(400) * 0.3
    lon = 80.2 + rng.random(400) * 0.3
    keep = geo.thin_by_spacing(lat, lon, min_km=1.0)

    kept_lat, kept_lon = lat[keep], lon[keep]
    for i in range(len(keep)):
        for j in range(i + 1, len(keep)):
            assert geo.pairwise_km(kept_lat[i], kept_lon[i], kept_lat[j], kept_lon[j]) >= 0.999


def test_thinning_priority_keeps_the_best_point():
    """Without priority the survivor is arbitrary; with it, it is the best."""
    lat = np.array([13.000, 13.001, 13.002])  # all within ~200 m
    lon = np.array([80.000, 80.000, 80.000])
    quality = np.array([0.1, 0.9, 0.2])

    keep = geo.thin_by_spacing(lat, lon, min_km=1.0, order=np.argsort(-quality))
    assert len(keep) == 1
    assert keep[0] == 1


def test_selection_respects_spacing_and_limit():
    rng = np.random.default_rng(1)
    lat = 13.0 + rng.random(200) * 0.4
    lon = 80.2 + rng.random(200) * 0.4
    scores = rng.random(200) * 100

    chosen = geo.select_spaced(lat, lon, scores, min_km=2.0, limit=10)
    assert len(chosen) <= 10
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            a, b = chosen[i], chosen[j]
            assert geo.pairwise_km(lat[a], lon[a], lat[b], lon[b]) >= 1.999


def test_selection_is_score_ordered():
    lat = np.array([13.0, 13.5, 14.0])
    lon = np.array([80.0, 80.5, 81.0])
    chosen = geo.select_spaced(lat, lon, [10.0, 90.0, 50.0], min_km=1.0, limit=3)
    assert list(chosen) == [1, 2, 0]


def test_decay_reaches_true_zero_at_saturation():
    assert geo.decay_score([0.0], 3.0)[0] == pytest.approx(1.0)
    assert geo.decay_score([3.0], 3.0)[0] == pytest.approx(0.0)
    assert geo.decay_score([99.0], 3.0)[0] == pytest.approx(0.0)
    assert geo.decay_score([np.inf], 3.0)[0] == pytest.approx(0.0)


def test_normalise_handles_constant_input():
    out = geo.normalise([5.0, 5.0, 5.0])
    assert np.allclose(out, 0.5)


# ------------------------------------------------------------------ fixtures
def make_features(n=6, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {
        "latitude": 13.0 + rng.random(n) * 0.2,
        "longitude": 80.2 + rng.random(n) * 0.2,
        "road_degree": rng.integers(2, 7, n).astype(float),
        "road_class_weight": rng.random(n),
        "road_class": ["primary"] * n,
        "km_to_station": rng.random(n) * 10,
    }
    from config import POI_CATEGORIES

    for cat in POI_CATEGORIES:
        cols[f"km_to_{cat}"] = rng.random(n) * 4
    return pd.DataFrame(cols)


# ------------------------------------------------------------------ weights
def test_default_weights_sum_to_one():
    from config import DEFAULT_WEIGHTS

    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


def test_weights_are_normalised_not_rejected():
    out = scoring.validate_weights({k: 2.0 for k in scoring.DEFAULT_WEIGHTS})
    assert sum(out.values()) == pytest.approx(1.0)


def test_unknown_weight_key_raises_rather_than_silently_zeroing():
    bad = dict(scoring.DEFAULT_WEIGHTS)
    bad["demandd"] = 0.1
    with pytest.raises(scoring.WeightError):
        scoring.validate_weights(bad)


def test_negative_and_empty_weights_rejected():
    bad = dict(scoring.DEFAULT_WEIGHTS)
    bad["demand"] = -1.0
    with pytest.raises(scoring.WeightError):
        scoring.validate_weights(bad)
    with pytest.raises(scoring.WeightError):
        scoring.validate_weights({})


# ------------------------------------------------------------------ scoring
def test_all_dimensions_are_within_range():
    dims = scoring.compute_dimensions(make_features(20))
    assert set(dims.columns) == set(scoring.DEFAULT_WEIGHTS)
    assert dims.to_numpy().min() >= 0.0
    assert dims.to_numpy().max() <= 100.0


def test_overall_score_bounded():
    dims = scoring.compute_dimensions(make_features(20))
    total = scoring.apply_weights(dims)
    assert total.min() >= 0.0 and total.max() <= 100.0


def test_coverage_gap_rises_with_distance_from_stations():
    """Spec 47: a site near an existing station must score lower on coverage."""
    f = make_features(2)
    f.loc[0, "km_to_station"] = 0.3
    f.loc[1, "km_to_station"] = 7.5
    gap = scoring.coverage_gap_score(f)
    assert gap[0] < gap[1]


def test_coverage_gap_saturates_so_remote_sites_do_not_run_away():
    """Spec 26: isolation beyond saturation earns nothing extra."""
    f = make_features(2)
    f.loc[0, "km_to_station"] = 8.0
    f.loc[1, "km_to_station"] = 400.0
    gap = scoring.coverage_gap_score(f)
    assert gap[0] == pytest.approx(gap[1])


def test_missing_station_layer_scores_neutral_not_perfect():
    f = make_features(3)
    f["km_to_station"] = np.inf
    assert np.allclose(scoring.coverage_gap_score(f), 50.0)


def test_empty_poi_layer_is_excluded_not_counted_as_far():
    """A city with no metro data must not be punished as if metros were absent."""
    full = make_features(5, seed=3)
    stripped = full.copy()
    for cat in ("metro", "railway", "bus_station", "bus_stop"):
        stripped[f"km_to_{cat}"] = np.inf
    assert scoring.poi_score(stripped).min() > 0.0


def test_weight_change_alters_the_ranking():
    """Spec 47: recommendations must update when weights change."""
    f = make_features(30, seed=7)
    dims = scoring.compute_dimensions(f)

    demand_heavy = {k: 0.0 for k in scoring.DEFAULT_WEIGHTS}
    demand_heavy["demand"] = 1.0
    coverage_heavy = {k: 0.0 for k in scoring.DEFAULT_WEIGHTS}
    coverage_heavy["coverage_gap"] = 1.0

    a = scoring.apply_weights(dims, demand_heavy)
    b = scoring.apply_weights(dims, coverage_heavy)
    assert not np.allclose(a, b)
    assert list(np.argsort(-a)) != list(np.argsort(-b))


def test_scoring_is_reproducible():
    f = make_features(15, seed=11)
    a = scoring.score_candidates(f)["overall_score"].to_numpy()
    b = scoring.score_candidates(f)["overall_score"].to_numpy()
    assert np.array_equal(a, b)


def test_missing_feature_columns_do_not_crash():
    """Spec 47: handle missing values gracefully."""
    bare = pd.DataFrame({"latitude": [13.0], "longitude": [80.0]})
    dims = scoring.compute_dimensions(bare)
    assert len(dims) == 1
    assert np.isfinite(scoring.apply_weights(dims)).all()


# ------------------------------------------------------------------ feasibility
def test_feasibility_bands_are_ordered():
    from feasibility import feasibility_band

    assert feasibility_band(100) == "Excellent"
    assert feasibility_band(80) == "Good"
    assert feasibility_band(60) == "Moderate concerns"
    assert feasibility_band(0) == "Unsuitable"


def test_point_inside_polygon_is_a_conflict_but_nearby_is_not():
    """
    Spec 23, the central distinction: inside hospital grounds is infeasible,
    next door to the hospital is not.
    """
    import geopandas as gpd
    from shapely.geometry import Point, Polygon

    grounds = Polygon([(80.20, 13.00), (80.21, 13.00), (80.21, 13.01), (80.20, 13.01)])
    layer = gpd.GeoDataFrame({"name": ["General Hospital"]}, geometry=[grounds], crs="EPSG:4326")

    inside = gpd.GeoDataFrame(geometry=[Point(80.205, 13.005)], crs="EPSG:4326")
    beside = gpd.GeoDataFrame(geometry=[Point(80.215, 13.005)], crs="EPSG:4326")

    joined_inside = gpd.sjoin(inside, layer[["geometry"]], predicate="within", how="inner")
    joined_beside = gpd.sjoin(beside, layer[["geometry"]], predicate="within", how="inner")

    assert len(joined_inside) == 1
    assert len(joined_beside) == 0


def test_water_penalty_disqualifies():
    from config import FEASIBILITY_REJECT_BELOW, FEASIBILITY_START_SCORE, RESTRICTED_LANDUSE

    remaining = FEASIBILITY_START_SCORE - RESTRICTED_LANDUSE["water"]["penalty"]
    assert remaining < FEASIBILITY_REJECT_BELOW


# ------------------------------------------------------------------ landmarks
def test_landmarks_sorted_nearest_first():
    from config import POI_CATEGORIES

    row = make_features(1).iloc[0].copy()
    for cat in POI_CATEGORIES:  # clear the random fixture distances first
        row[f"km_to_{cat}"] = np.inf
    row["km_to_metro"] = 0.4
    row["km_to_mall"] = 1.8
    row["km_to_parking"] = 0.2
    items = nearby(row)
    assert items[0][1] <= items[1][1] <= items[2][1]
    assert items[0][0] == "Parking"


def test_no_landmark_in_range_reports_it():
    """Spec 47: 'No nearby landmark found'."""
    row = make_features(1).iloc[0].copy()
    from config import POI_CATEGORIES

    for cat in POI_CATEGORIES:
        row[f"km_to_{cat}"] = np.inf
    assert nearby(row) == []
    assert primary_landmark(row) == NO_LANDMARKS


# ------------------------------------------------------------------ explanations
def _scored_row(**overrides):
    f = make_features(1, seed=5)
    dims = scoring.compute_dimensions(f)
    row = pd.concat([f, dims], axis=1).iloc[0].copy()
    row["overall_score"] = 80.0
    row["conflicts"] = ""
    for k, v in overrides.items():
        row[k] = v
    return row


def test_explanation_is_grammatical():
    """
    The old generator produced 'Recommended because it has near an IT park'.
    Every clause must now read correctly after the shared stem.
    """
    broken = ["it has near", "it has close", "it has located", "it has well", "has near a"]
    for seed in range(40):
        f = make_features(1, seed=seed)
        dims = scoring.compute_dimensions(f)
        row = pd.concat([f, dims], axis=1).iloc[0].copy()
        row["overall_score"] = 70.0
        row["conflicts"] = ""
        text = explain(row).lower()
        for phrase in broken:
            assert phrase not in text, f"seed {seed}: {text}"


def test_explanation_quotes_the_real_coverage_distance():
    """
    The old text claimed 'well separated from existing charging stations'
    from a number measuring cluster centrality. Now it must quote the actual
    measured distance to a station.
    """
    row = _scored_row(km_to_station=6.4, coverage_gap=88.0)
    text = explain(row)
    assert "6.4 km" in text
    assert "charg" in text.lower()


def test_explanation_never_invents_a_landmark():
    from config import POI_CATEGORIES

    row = _scored_row()
    for cat in POI_CATEGORIES:
        row[f"km_to_{cat}"] = np.inf
    row["demand"] = 10.0
    row["poi"] = 10.0
    text = explain(row)
    for spec in POI_CATEGORIES.values():
        assert spec["label"].lower() not in text.lower()


def test_explanation_mentions_conflicts_when_present():
    row = _scored_row(conflicts="a park")
    assert "a park" in explain(row)


def test_explanation_is_bounded_in_length():
    for seed in range(25):
        f = make_features(1, seed=seed)
        dims = scoring.compute_dimensions(f)
        row = pd.concat([f, dims], axis=1).iloc[0].copy()
        row["overall_score"] = 75.0
        row["conflicts"] = ""
        assert len(explain(row)) < 320


# ------------------------------------------------------------------ clustering
def test_clustering_labels_every_candidate():
    f = make_features(40)
    labels, centers = cluster_candidates(f, n_clusters=5)
    assert len(labels) == 40
    assert len(centers) == 5
    assert set(labels) <= set(range(5))


def test_clustering_clamps_k_to_available_points():
    f = make_features(3)
    labels, centers = cluster_candidates(f, n_clusters=10)
    assert len(centers) == 3


def test_clustering_is_deterministic():
    f = make_features(30)
    a, _ = cluster_candidates(f, n_clusters=4)
    b, _ = cluster_candidates(f, n_clusters=4)
    assert np.array_equal(a, b)


def test_empty_candidate_set_does_not_crash():
    """Spec 47: empty external-data result must not crash the application."""
    empty = pd.DataFrame({"latitude": [], "longitude": []})
    labels, centers = cluster_candidates(empty, n_clusters=3)
    assert len(labels) == 0
    assert centers.shape[0] == 0


# ------------------------------------------------------------------ legacy API
def test_legacy_recommend_locations_still_works(tmp_path):
    """The original signature must keep working for teammates' code."""
    from recommendation import recommend_locations

    csv = tmp_path / "sample_locations.csv"
    make_features(20)[["latitude", "longitude"]].to_csv(csv, index=False)

    data, centers = recommend_locations(str(csv), n_clusters=3)
    assert "Cluster" in data.columns
    assert len(data) == 20
    assert centers.shape == (3, 2)
