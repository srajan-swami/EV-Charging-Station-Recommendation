"""
ChargeSense — explanation generation.

Every recommendation gets a one-or-two line reason assembled from its own
computed features. Nothing here is generic filler: a clause only appears when
the number behind it actually cleared its threshold, and where a distance is
mentioned it is the measured distance for that specific site.

Grammar note. The clauses are written as noun phrases that all read correctly
after a single fixed stem ("Recommended for its ..."). That is deliberate: an
earlier version prefixed "Recommended because it has" onto a mixed bag of
adjectival and prepositional fragments and produced lines like "Recommended
because it has near an IT park". Designing the phrases to share one
grammatical shape removes that failure mode rather than patching it.
"""

from __future__ import annotations

import numpy as np

from config import DEFAULT_WEIGHTS, POI_CATEGORIES

STEM = "Recommended for its "
STEM_WEAK = "Included as a lower-ranked option, offering "

# A dimension has to clear this to be worth mentioning at all.
MENTION_THRESHOLD = 55.0
STRONG_THRESHOLD = 75.0
MAX_CLAUSES = 3


def _fmt_km(km: float) -> str:
    if km is None or not np.isfinite(km):
        return ""
    return f"{km:.1f} km" if km >= 0.1 else "under 0.1 km"


def _nearest_named_poi(row, categories) -> tuple[str, float] | None:
    """The closest POI among `categories`, as (label, km)."""
    best, best_km = None, np.inf
    for cat in categories:
        km = row.get(f"km_to_{cat}", np.inf)
        try:
            km = float(km)
        except (TypeError, ValueError):
            continue
        if np.isfinite(km) and km < best_km:
            best, best_km = cat, km
    if best is None:
        return None
    return POI_CATEGORIES[best]["label"], best_km


def _demand_clause(row, score) -> str | None:
    if score < MENTION_THRESHOLD:
        return None
    found = _nearest_named_poi(row, ["mall", "office", "commercial"])
    if found and found[1] <= 1.5:
        label, km = found
        qualifier = "strong" if score >= STRONG_THRESHOLD else "solid"
        return f"{qualifier} surrounding commercial activity ({label.lower()} {_fmt_km(km)} away)"
    return "strong surrounding commercial activity" if score >= STRONG_THRESHOLD else "steady local activity"


def _poi_clause(row, score) -> str | None:
    if score < MENTION_THRESHOLD:
        return None
    found = _nearest_named_poi(row, ["metro", "railway", "bus_station", "parking"])
    if found:
        label, km = found
        return f"proximity to a {label.lower()} ({_fmt_km(km)})"
    return "good access to nearby public destinations"


def _coverage_clause(row, score) -> str | None:
    if score < MENTION_THRESHOLD:
        return None
    km = row.get("km_to_station", np.inf)
    try:
        km = float(km)
    except (TypeError, ValueError):
        km = np.inf
    if np.isfinite(km):
        return f"a clear gap in existing charging coverage (nearest charger {_fmt_km(km)} away)"
    return "an area with no charging coverage currently mapped nearby"


def _traffic_clause(row, score) -> str | None:
    if score < MENTION_THRESHOLD:
        return None
    road = str(row.get("road_class", "") or "").replace("_", " ")
    if road and road not in ("nan", ""):
        return f"a position on a {road} road"
    return "good road accessibility"


def _road_access_clause(row, score) -> str | None:
    if score < MENTION_THRESHOLD:
        return None
    deg = row.get("road_degree", None)
    try:
        deg = int(float(deg))
    except (TypeError, ValueError):
        deg = None
    if deg and deg >= 4:
        return f"a well-connected junction ({deg} roads meeting)"
    return "a well-connected junction"


def _feasibility_clause(row, score) -> str | None:
    # Only worth saying when it is genuinely clean; "moderate concerns" is not
    # a selling point and belongs in the feasibility field instead.
    if score < 95.0:
        return None
    return "a site free of land-use conflicts"


CLAUSE_BUILDERS = {
    "demand": _demand_clause,
    "poi": _poi_clause,
    "coverage_gap": _coverage_clause,
    "traffic_access": _traffic_clause,
    "road_access": _road_access_clause,
    "feasibility": _feasibility_clause,
}


def _join(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def explain(row, weights: dict | None = None) -> str:
    """
    Build the reason for one scored candidate.

    `row` must carry the six dimension scores plus the raw feature columns, so
    the clause can quote the site's own measured distances.
    """
    w = weights or DEFAULT_WEIGHTS

    # Rank dimensions by how much they actually contributed to this score,
    # not by raw value — a 90 on a 10%-weighted dimension matters less than
    # a 70 on a 25%-weighted one.
    ranked = sorted(
        CLAUSE_BUILDERS,
        key=lambda d: -(float(row.get(d, 0.0)) * float(w.get(d, 0.0))),
    )

    clauses = []
    for dim in ranked:
        if len(clauses) >= MAX_CLAUSES:
            break
        score = float(row.get(dim, 0.0))
        clause = CLAUSE_BUILDERS[dim](row, score)
        if clause:
            clauses.append(clause)

    if not clauses:
        return (
            "Included as a balanced option: no single factor stands out, but "
            "the site has no disqualifying conflicts."
        )

    overall = float(row.get("overall_score", 0.0))
    stem = STEM if overall >= 50 else STEM_WEAK
    sentence = stem + _join(clauses) + "."

    conflicts = str(row.get("conflicts", "") or "")
    if conflicts:
        sentence += f" Note: partial land-use conflict with {conflicts}."

    return sentence


def explain_frame(df, weights: dict | None = None):
    """Vectorised wrapper — returns a Series of reasons aligned to `df`."""
    return df.apply(lambda r: explain(r, weights), axis=1)
