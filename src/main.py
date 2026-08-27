"""
ChargeSense — command-line pipeline.

    python src/main.py                          # Chennai, defaults
    python src/main.py --city Mumbai
    python src/main.py --city Delhi --refresh   # ignore caches, re-download
    python src/main.py --geocode                # resolve addresses (slow, once)
    python src/main.py --prefetch-all           # cache all four cities

Outputs, per city, under outputs/<city>/:
    recommendations.csv    ranked sites with scores, reasons and feasibility
    scored_candidates.csv  every candidate with its dimension scores
    final_map.html         the interactive map
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from config import (
    CITIES,
    DEFAULT_CITY,
    DEFAULT_N_CLUSTERS,
    DEFAULT_N_RECOMMENDATIONS,
    DEFAULT_WEIGHTS,
    output_dir,
)

log = logging.getLogger("chargesense")

EXPORT_COLUMNS = [
    "rank", "latitude", "longitude", "address", "overall_score",
    "demand", "traffic_access", "poi", "coverage_gap", "feasibility", "road_access",
    "km_to_station", "nearest_landmark", "feasibility_band", "conflicts",
    "road_class", "road_degree", "cluster", "reason",
]


def run(city: str, n_clusters: int, n_recommendations: int,
        refresh: bool, geocode: bool, no_roads: bool) -> pd.DataFrame:
    from data_loader import load_gdfs, load_stations
    from recommendation import (
        attach_cached_addresses,
        build_scored_table,
        select_recommendations,
    )
    from visualization import build_map, save_map

    log.info("=" * 62)
    log.info("ChargeSense — %s", city)
    log.info("=" * 62)

    table = build_scored_table(city, n_clusters=n_clusters, refresh=refresh)
    result = select_recommendations(
        table, weights=DEFAULT_WEIGHTS, n_recommendations=n_recommendations
    )

    if result.empty:
        log.error("[%s] no recommendations produced — check the filters above.", city)
        return result

    if geocode:
        from geocoding import resolve_addresses, short_address

        addresses = resolve_addresses(
            city, result["latitude"], result["longitude"], use_network=True
        )
        result["address"] = [short_address(a) for a in addresses]
    else:
        result = attach_cached_addresses(city, result)

    out = output_dir(city)

    table.to_csv(out / "scored_candidates.csv", index=False)
    cols = [c for c in EXPORT_COLUMNS if c in result.columns]
    result[cols].to_csv(out / "recommendations.csv", index=False)

    stations = load_stations(city)
    edges = None
    if not no_roads:
        try:
            _, edges = load_gdfs(city)
        except Exception as exc:
            log.warning("[%s] road geometry unavailable for display: %s", city, exc)

    fmap = build_map(city, result, stations=stations, edges=edges, show_roads=not no_roads)
    save_map(fmap, out / "final_map.html")

    log.info("-" * 62)
    log.info("Candidates scored     : %d", len(table))
    log.info("Existing stations     : %d (OSM-tagged)", len(stations))
    log.info("Recommended sites     : %d", len(result))
    log.info("Mean score            : %.1f / 100", result["overall_score"].mean())
    log.info("Score range           : %.1f – %.1f",
             result["overall_score"].min(), result["overall_score"].max())
    log.info("Output directory      : %s", out)
    log.info("-" * 62)

    print("\nTop 10 recommended sites\n")
    show = ["rank", "latitude", "longitude", "overall_score", "feasibility_band"]
    print(result[[c for c in show if c in result.columns]].head(10).to_string(index=False))
    print(f"\nOpen: {out / 'final_map.html'}")

    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ChargeSense recommendation pipeline")
    parser.add_argument("--city", default=DEFAULT_CITY, choices=list(CITIES))
    parser.add_argument("--clusters", type=int, default=DEFAULT_N_CLUSTERS)
    parser.add_argument("--top", type=int, default=DEFAULT_N_RECOMMENDATIONS)
    parser.add_argument("--refresh", action="store_true", help="ignore caches and re-download")
    parser.add_argument("--geocode", action="store_true", help="resolve addresses (~1s per site)")
    parser.add_argument("--no-roads", action="store_true", help="skip road layer for a lighter map")
    parser.add_argument("--prefetch-all", action="store_true",
                        help="download and cache every supported city, then exit")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
    )

    if args.prefetch_all:
        from data_loader import prefetch_city

        for city in CITIES:
            log.info("\n--- prefetching %s ---", city)
            try:
                summary = prefetch_city(city, refresh=args.refresh)
                log.info("%s: %d nodes, %d edges, %d stations",
                         city, summary["nodes"], summary["edges"], summary["stations"])
            except Exception as exc:
                log.error("%s failed: %s", city, exc)
        return 0

    try:
        result = run(args.city, args.clusters, args.top,
                     args.refresh, args.geocode, args.no_roads)
    except Exception as exc:
        log.error("Pipeline failed: %s", exc, exc_info=not args.quiet)
        return 1

    return 0 if not result.empty else 1


if __name__ == "__main__":
    sys.exit(main())
