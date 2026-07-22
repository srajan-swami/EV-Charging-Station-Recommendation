import pandas as pd
from sklearn.cluster import KMeans


def recommend_locations(csv_file, n_clusters=3):
    # Read CSV
    data = pd.read_csv(csv_file)

    # Select Latitude and Longitude
    X = data[["latitude", "longitude"]]

    # Create K-Means Model
    model = KMeans(n_clusters=n_clusters, random_state=42)

    # Train the model
    model.fit(X)

    # Add cluster labels
    data["Cluster"] = model.labels_

    return data, model.cluster_centers_


# This runs only when clustering.py is executed directly
if __name__ == "__main__":
    data, centers = recommend_locations("data/sample_locations.csv")

    print(data)

    print("\nCluster Centers:")
    print(centers)