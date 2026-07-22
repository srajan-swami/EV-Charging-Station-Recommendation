import osmnx as ox
import pandas as pd

city = "Chennai, Tamil Nadu, India"

print("Downloading road network...")

graph = ox.graph_from_place(city, network_type="drive")

# Get all road intersections (nodes)
nodes, edges = ox.graph_to_gdfs(graph)

# Create latitude and longitude columns
candidate_locations = pd.DataFrame({
    "latitude": nodes.geometry.y,
    "longitude": nodes.geometry.x
})

print(candidate_locations.head())

print(f"\nTotal candidate locations: {len(candidate_locations)}")

# Save CSV for clustering
candidate_locations.to_csv(
    "data/sample_locations.csv",
    index=False
)

print("\nSaved to data/sample_locations.csv")