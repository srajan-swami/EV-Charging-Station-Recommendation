import osmnx as ox
import pandas as pd
from sklearn.neighbors import BallTree
import numpy as np

city = "Chennai, Tamil Nadu, India"

print("Downloading road network...")
graph = ox.graph_from_place(city, network_type="drive")

# Get road intersections
nodes, edges = ox.graph_to_gdfs(graph)

candidate_locations = pd.DataFrame({
    "latitude": nodes.geometry.y,
    "longitude": nodes.geometry.x,
})

print(f"Initial candidate locations: {len(candidate_locations)}")

# Remove exact duplicate coordinates
candidate_locations = candidate_locations.drop_duplicates(
    subset=["latitude", "longitude"]
).reset_index(drop=True)

# Remove locations that are too close together (minimum 1 km)
coords_deg = candidate_locations[["latitude", "longitude"]].to_numpy()
coords_rad = np.radians(coords_deg)

tree = BallTree(coords_rad, metric="haversine")

radius = 1.0 / 6371.0  # 1 km in radians

keep = []
visited = set()

for i in range(len(coords_rad)):
    if i in visited:
        continue

    keep.append(i)
    neighbors = tree.query_radius([coords_rad[i]], r=radius)[0]

    for n in neighbors:
        if n != i:
            visited.add(int(n))

candidate_locations = candidate_locations.iloc[keep].reset_index(drop=True)

candidate_locations = candidate_locations.sort_values(
    by=["latitude", "longitude"]
).reset_index(drop=True)

candidate_locations.to_csv(
    "data/sample_locations.csv",
    index=False,
)

print("Saved to data/sample_locations.csv")
print(candidate_locations.head())