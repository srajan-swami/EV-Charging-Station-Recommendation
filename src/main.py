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

# Save ranked recommendations for visualization
clustered_data.to_csv(
    "outputs/recommendations.csv",
    index=False
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

NUM_RECOMMENDATIONS = 75
# Top Recommended Locations
for _, row in clustered_data.head(NUM_RECOMMENDATIONS).iterrows():
    score = row["RecommendationScore"]

    color = "red"

    popup = folium.Popup(
        f"""
        <b>⚡ EV Charging Recommendation</b><br><br>
        <b>Rank:</b> {int(row['Rank'])}<br>
        <b>Recommendation Score:</b> {score:.2f}<br><br>
        <b>Why Recommended?</b><br>
        {row['Recommendation']}
        """,
        max_width=320,
    )

    folium.Marker(
        [row["latitude"], row["longitude"]],
        popup=popup,
        icon=folium.Icon(color="red", icon="bolt", prefix="fa")
    ).add_to(m)

# Save map
m.save("outputs/final_map.html")

print(f"Total Candidate Locations : {len(clustered_data)}")
print(f"Top Recommendations Shown : {min(NUM_RECOMMENDATIONS, len(clustered_data))}")
print("Recommendations saved to outputs/recommendations.csv")
print("Interactive map saved to outputs/final_map.html")
print("Project completed successfully!")