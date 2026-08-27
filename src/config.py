"""
ChargeSense — central configuration.

Every tunable value in the system lives here. Nothing else in the codebase
should hardcode a city name, a weight, a threshold or an OSM tag.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

for _d in (DATA_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def city_dir(city: str) -> Path:
    """Per-city data directory. City data is never mixed (see spec 36)."""
    d = DATA_DIR / city.lower().replace(" ", "_")
    d.mkdir(parents=True, exist_ok=True)
    return d


def output_dir(city: str) -> Path:
    d = OUTPUT_DIR / city.lower().replace(" ", "_")
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- cities
CITIES = {
    "Chennai": {
        "place": "Chennai, Tamil Nadu, India",
        "latitude": 13.0827,
        "longitude": 80.2707,
        "utm_epsg": 32644,
    },
    "Mumbai": {
        "place": "Mumbai, India",
        "latitude": 19.0760,
        "longitude": 72.8777,
        "utm_epsg": 32643,
    },
    "Delhi": {
        "place": "Delhi, India",
        "latitude": 28.6139,
        "longitude": 77.2090,
        "utm_epsg": 32643,
    },
    "Hyderabad": {
        "place": "Hyderabad, Telangana, India",
        "latitude": 17.3850,
        "longitude": 78.4867,
        "utm_epsg": 32644,
    },
}

DEFAULT_CITY = "Chennai"


def city_config(city: str) -> dict:
    if city not in CITIES:
        raise KeyError(
            f"Unknown city {city!r}. Known cities: {', '.join(CITIES)}"
        )
    return CITIES[city]


# ---------------------------------------------------------------- scoring
# Six dimensions. Weights must sum to 1.0 — validated by scoring.validate_weights.
DEFAULT_WEIGHTS = {
    "demand": 0.25,
    "traffic_access": 0.20,
    "poi": 0.15,
    "coverage_gap": 0.15,
    "feasibility": 0.15,
    "road_access": 0.10,
}

DIMENSION_LABELS = {
    "demand": "Demand",
    "traffic_access": "Traffic / Access",
    "poi": "POI / Landmarks",
    "coverage_gap": "Coverage Gap",
    "feasibility": "Site Feasibility",
    "road_access": "Road Access",
}

# Traffic is a PROXY. OpenStreetMap carries no traffic volume (spec 17).
TRAFFIC_IS_PROXY = True
TRAFFIC_PROXY_NOTE = (
    "Road-hierarchy accessibility proxy derived from OSM road classification. "
    "Not measured traffic volume."
)

# ---------------------------------------------------------------- POI layers
# `filter_tag` keeps only rows whose column == value; `exclude_tag` drops them.
# This is how metro and railway stations are kept genuinely distinct: OSMnx
# treats a multi-key tag dict as OR, not AND, so {"railway":"station",
# "station":"subway"} returns the whole railway set. We query once and split.
POI_CATEGORIES = {
    "mall": {
        "tags": {"shop": "mall"},
        "dimension": "demand",
        "label": "Shopping Mall",
    },
    "office": {
        "tags": {"office": True},
        "dimension": "demand",
        "label": "Office",
    },
    "commercial": {
        "tags": {"landuse": "commercial"},
        "dimension": "demand",
        "label": "Commercial Area",
    },
    "restaurant": {
        "tags": {"amenity": "restaurant"},
        "dimension": "demand",
        "label": "Restaurant",
    },
    "hotel": {
        "tags": {"tourism": "hotel"},
        "dimension": "demand",
        "label": "Hotel",
    },
    "metro": {
        "tags": {"railway": "station"},
        "filter_tag": ("station", "subway"),
        "dimension": "poi",
        "label": "Metro Station",
    },
    "railway": {
        "tags": {"railway": "station"},
        "exclude_tag": ("station", "subway"),
        "dimension": "poi",
        "label": "Railway Station",
    },
    "bus_stop": {
        "tags": {"highway": "bus_stop"},
        "dimension": "poi",
        "label": "Bus Stop",
    },
    "bus_station": {
        "tags": {"amenity": "bus_station"},
        "dimension": "poi",
        "label": "Bus Terminal",
    },
    "hospital": {
        "tags": {"amenity": "hospital"},
        "dimension": "poi",
        "label": "Hospital",
    },
    "university": {
        "tags": {"amenity": "university"},
        "dimension": "poi",
        "label": "University",
    },
    "parking": {
        "tags": {"amenity": "parking"},
        "dimension": "poi",
        "label": "Parking",
    },
    "fuel": {
        "tags": {"amenity": "fuel"},
        "dimension": "poi",
        "label": "Petrol Station",
    },
}

# Distance at which a POI stops contributing. Closer than this scores higher.
POI_SATURATION_KM = 3.0

# Landmarks surfaced in the popup (spec 19). Straight-line distance.
LANDMARK_CATEGORIES = [
    "metro",
    "railway",
    "bus_stop",
    "bus_station",
    "mall",
    "parking",
    "hospital",
    "fuel",
    "university",
]
MAX_LANDMARKS_SHOWN = 5
LANDMARK_MAX_DISTANCE_KM = 5.0

# ---------------------------------------------------------------- feasibility
# A candidate INSIDE one of these polygons has a physical land-use conflict.
# Being NEAR one is not penalised — proximity and conflict are different
# things (spec 23).
RESTRICTED_LANDUSE = {
    "hospital_grounds": {
        "tags": {"amenity": "hospital"},
        "label": "hospital grounds",
        "penalty": 70,
    },
    "school_grounds": {
        "tags": {"amenity": "school"},
        "label": "school grounds",
        "penalty": 70,
    },
    "park": {
        "tags": {"leisure": "park"},
        "label": "a park",
        "penalty": 50,
    },
    "water": {
        "tags": {"natural": "water"},
        "label": "a water body",
        "penalty": 100,
    },
    "railway_land": {
        "tags": {"landuse": "railway"},
        "label": "railway land",
        "penalty": 60,
    },
    "aerodrome": {
        "tags": {"aeroway": "aerodrome"},
        "label": "airport land",
        "penalty": 100,
    },
    "military": {
        "tags": {"landuse": "military"},
        "label": "restricted military land",
        "penalty": 100,
    },
}

FEASIBILITY_START_SCORE = 100.0
# Below this a candidate is dropped entirely rather than merely penalised.
FEASIBILITY_REJECT_BELOW = 40.0

FEASIBILITY_BANDS = [
    (90, "Excellent"),
    (75, "Good"),
    (50, "Moderate concerns"),
    (25, "Poor"),
    (0, "Unsuitable"),
]

# ---------------------------------------------------------------- geometry
EARTH_RADIUS_KM = 6371.0

MIN_CANDIDATE_SPACING_KM = 1.0
MIN_RECOMMENDATION_SPACING_KM = 2.0
MIN_DISTANCE_FROM_STATION_KM = 2.0

# Coverage-gap score saturates here: beyond this, further isolation earns
# nothing extra. Prevents recommending remote sites purely for being remote
# (spec 26).
COVERAGE_GAP_SATURATION_KM = 8.0

# Road hierarchy -> accessibility proxy weight (spec 17).
ROAD_CLASS_WEIGHTS = {
    "motorway": 1.00,
    "motorway_link": 0.90,
    "trunk": 0.90,
    "trunk_link": 0.82,
    "primary": 0.80,
    "primary_link": 0.72,
    "secondary": 0.65,
    "secondary_link": 0.58,
    "tertiary": 0.50,
    "tertiary_link": 0.45,
    "unclassified": 0.35,
    "residential": 0.30,
    "living_street": 0.22,
    "service": 0.20,
}
ROAD_CLASS_DEFAULT = 0.40

# Intersection degree at which road-access saturates.
ROAD_DEGREE_SATURATION = 6

# ---------------------------------------------------------------- clustering
DEFAULT_N_CLUSTERS = 5
DEFAULT_N_RECOMMENDATIONS = 50
RANDOM_STATE = 42

# ---------------------------------------------------------------- geocoding
GEOCODER_USER_AGENT = "ChargeSense/1.0 (hackathon prototype)"
GEOCODER_MIN_INTERVAL_SEC = 1.1  # Nominatim policy: max 1 request/second
GEOCODER_TIMEOUT_SEC = 10
ADDRESS_UNAVAILABLE = "Address unavailable"

# ---------------------------------------------------------------- provenance
DATA_SOURCE_NOTE = (
    "Recommendations are based on geospatial features currently mapped in "
    "OpenStreetMap for the selected city. OSM coverage is uneven and is not a "
    "complete inventory of existing charging infrastructure."
)
