import osmnx as ox
import folium

city = "Chennai, Tamil Nadu, India"

# Load road network
graph = ox.graph_from_place(city, network_type="drive")
nodes, edges = ox.graph_to_gdfs(graph)

# Load charging stations
tags = {"amenity": "charging_station"}
stations = ox.features_from_place(city, tags)

# Create map
m = folium.Map(location=[13.0827, 80.2707], zoom_start=11)

# Add roads
folium.GeoJson(
    edges.to_json(),
    name="Road Network"
).add_to(m)

# Add charging station markers
for idx, row in stations.iterrows():
    point = row.geometry

    folium.Marker(
        location=[point.y, point.x],
        popup=row.get("name", "EV Charging Station"),
        icon=folium.Icon(color="green", icon="bolt", prefix="fa")
    ).add_to(m)

# Save map
m.save("outputs/chennai_ev_map.html")

print("Map saved successfully!")