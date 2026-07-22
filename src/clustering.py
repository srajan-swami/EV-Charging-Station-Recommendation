import pandas as pd
from sklearn.cluster import KMeans

# Read CSV
data = pd.read_csv("data/sample_locations.csv")

# Select Latitude and Longitude
X = data[["latitude", "longitude"]]

# Create K-Means Model
model = KMeans(
    n_clusters=3,
    random_state=42
)

# Train the model
model.fit(X)

# Add cluster number to each point
data["Cluster"] = model.labels_

print(data)

print("\nCluster Centers:")
print(model.cluster_centers_)