print("=== NEW MAIN.PY IS RUNNING ===")
import folium
import osmnx as ox

from recommendation import recommend_locations

# -----------------------------
# Load Road Network
# -----------------------------
city = "Chennai, Tamil Nadu, India"

print("Loading road network...")

graph = ox.graph_from_place(city, network_type="drive")
nodes, edges = ox.graph_to_gdfs(graph)

# -----------------------------
# Load Existing Charging Stations
# -----------------------------
print("Loading charging stations...")

tags = {"amenity": "charging_station"}
stations = ox.features_from_place(city, tags)

# -----------------------------
# Run Recommendation Model
# -----------------------------
print("Running clustering...")

clustered_data, centers = recommend_locations(
    "data/sample_locations.csv"
)

# -----------------------------
# Create Map
# -----------------------------
m = folium.Map(
    location=[13.0827, 80.2707],
    zoom_start=11
)

# Roads
folium.GeoJson(edges.to_json()).add_to(m)

# Existing Charging Stations
for _, row in stations.iterrows():
    point = row.geometry

    folium.Marker(
        [point.y, point.x],
        popup=row.get("name", "Charging Station"),
        icon=folium.Icon(color="green", icon="bolt", prefix="fa")
    ).add_to(m)

# Recommended Locations (Cluster Centers)
for center in centers:
    folium.Marker(
        [center[0], center[1]],
        popup="Recommended Location",
        icon=folium.Icon(color="red", icon="star")
    ).add_to(m)

# Save map
m.save("outputs/final_map.html")

print("Project completed successfully!")