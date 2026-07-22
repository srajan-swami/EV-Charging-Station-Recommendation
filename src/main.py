import osmnx as ox
import geopandas as gpd
import folium

city = "Chennai, Tamil Nadu, India"
graph = ox.graph_from_place(
    city,
    network_type="drive"
)
print(graph)

nodes, edges = ox.graph_to_gdfs(graph)
print(nodes.head())
print(edges.head())

# Create a map centered on Chennai
m = folium.Map(
    location=[13.0827, 80.2707],
    zoom_start=11
)

folium.GeoJson(
    edges.to_json(),
    name="Road Network"
).add_to(m)

# Save the map to an HTML file
m.save("outputs/chennai_map.html")
print("Map saved successfully!")