import os
import osmnx as ox
import folium
import pandas as pd

city = "Chennai, Tamil Nadu, India"

print("Loading road network...")
graph = ox.graph_from_place(city, network_type="drive")
nodes, edges = ox.graph_to_gdfs(graph)

print("Loading existing charging stations...")
tags = {"amenity": "charging_station"}
stations = ox.features_from_place(city, tags)

m = folium.Map(location=[13.0827, 80.2707], zoom_start=11)

# Road network
folium.GeoJson(edges.to_json(), name="Road Network").add_to(m)

# Existing charging stations
for _, row in stations.iterrows():
    point = row.geometry
    if point is None:
        continue
    folium.Marker(
        location=[point.y, point.x],
        popup=f"Existing Station\n{row.get('name', 'Unnamed Station')}",
        icon=folium.Icon(color="red", icon="bolt", prefix="fa"),
    ).add_to(m)

# Recommended locations
csv_path = "outputs/recommendations.csv"
if os.path.exists(csv_path):
    recommendations = pd.read_csv(csv_path)

    NUM_RECOMMENDATIONS = 75

    if "RecommendationScore" in recommendations.columns:
        recommendations = recommendations.sort_values(
            "RecommendationScore", ascending=False
        ).head(NUM_RECOMMENDATIONS)
    else:
        recommendations = recommendations.head(NUM_RECOMMENDATIONS)

    for i, row in recommendations.iterrows():
        score = row.get("RecommendationScore", "N/A")

        if score != "N/A":
            color = "red"
        else:
            color = "blue"

        popup = (
            f"<b>Recommended EV Location</b><br>"
            f"Rank: {i+1}<br>"
            f"Score: {score}"
        )

        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=popup,
            icon=folium.Icon(color="red", icon="bolt", prefix="fa"),
        ).add_to(m)

folium.LayerControl().add_to(m)

output_file = "outputs/final_map.html"
m.save(output_file)

print(f"Map saved successfully: {output_file}")