"""ChargeSense -- boundary fallback.

graph_from_place and features_from_place both resolve a place name through
osmnx.geocoder.geocode_to_gdf, which raises if the result is a point rather
than a (Multi)Polygon. Several large Indian cities -- Mumbai among them --
geocode to a point under every name form, so both fail before a single road
is downloaded.

The fix wraps that one function: when it fails for a city we have a centre
for, return a circle of the configured radius around that centre instead.
That circle approximates the city area, not its administrative boundary, and
the log says so. A real boundary is still preferred -- the original function
is always tried first.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import osmnx as ox
from osmnx import geocoder as _geocoder
from shapely.geometry import Point

log = logging.getLogger("chargesense.area")

_original_geocode_to_gdf = _geocoder.geocode_to_gdf

DEFAULT_RADIUS_KM = 20.0

# Records which cities fell back, so the pipeline can report it honestly.
APPROXIMATED = {}


def _city_for(query):
    """Match a place query back to a configured city."""
    from config import CITIES

    text = str(query).lower()
    for name, cfg in CITIES.items():
        if name.lower() in text:
            return name, cfg
    return None, None


def _circle_gdf(cfg, radius_km):
    """Circle of radius_km around the city centre, as a one-row GeoDataFrame."""
    centre = gpd.GeoSeries(
        [Point(float(cfg["longitude"]), float(cfg["latitude"]))], crs="EPSG:4326"
    )
    circle = (
        centre.to_crs(epsg=int(cfg["utm_epsg"]))
        .buffer(float(radius_km) * 1000.0)
        .to_crs(epsg=4326)
    )
    return gpd.GeoDataFrame(
        {"name": ["approximate city area"]}, geometry=circle, crs="EPSG:4326"
    )


def geocode_to_gdf(query, *, which_result=None, by_osmid=False):
    """Drop-in replacement that falls back to a radius instead of raising."""
    try:
        return _original_geocode_to_gdf(
            query, which_result=which_result, by_osmid=by_osmid
        )
    except Exception as exc:
        name, cfg = _city_for(query)
        if cfg is None:
            raise  # not one of ours -- let the real error surface

        radius_km = float(cfg.get("radius_km") or DEFAULT_RADIUS_KM)
        APPROXIMATED[name] = radius_km
        log.warning(
            "[%s] OpenStreetMap has no usable boundary for %r (%s). "
            "Falling back to a %.0f km radius around the city centre -- "
            "this is an approximation of the city area, not its boundary.",
            name, query, type(exc).__name__, radius_km,
        )
        return _circle_gdf(cfg, radius_km)


def area_note(city):
    """What area was actually analysed, for the map and the pitch."""
    if city in APPROXIMATED:
        return f"{APPROXIMATED[city]:.0f} km radius around the city centre (approximate)"
    return "OpenStreetMap administrative boundary"


# Install on import. Both graph_from_place and features_from_place resolve
# through this one attribute, so patching it covers every call site.
_geocoder.geocode_to_gdf = geocode_to_gdf
ox.geocode_to_gdf = geocode_to_gdf
