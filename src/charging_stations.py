import osmnx as ox
import geopandas as gpd

city = "Chennai, Tamil Nadu, India"

# Retrieve EV charging stations from OpenStreetMap
tags = {
    "amenity": "charging_station"
}

charging_stations = ox.features_from_place(city, tags)

print(charging_stations.head())
print(f"\nTotal charging stations found: {len(charging_stations)}")