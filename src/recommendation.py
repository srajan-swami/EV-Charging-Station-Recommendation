import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import BallTree
import osmnx as ox
import numpy as np


def recommend_locations(csv_file, n_clusters=5):
    # Read CSV
    data = pd.read_csv(csv_file)
    # Load existing EV charging stations from OpenStreetMap
    city = "Chennai, Tamil Nadu, India"
    stations = ox.features_from_place(
        city,
        {"amenity": "charging_station"}
    )

    # Candidate coordinates
    X = data[["latitude", "longitude"]]

    # Cluster candidate locations
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    data["Cluster"] = model.fit_predict(X)

    # -------- Recommendation Features using OpenStreetMap --------
    feature_queries = {
        "MallScore": {"shop": "mall"},
        "HospitalScore": {"amenity": "hospital"},
        "BusTerminalScore": {"amenity": "bus_station"},
        "RailwayScore": {"railway": "station"},
        "MetroScore": {"railway": "station", "station": "subway"},
        "ITParkScore": {
            "office": True,
            "building": "office"
        },
        "CommercialScore": {
            "landuse": "commercial"
        },
        "CompanyScore": {
            "office": "company"
        },
        "TrafficScore": {
            "highway": True
        },
    }

    candidate_coords = np.radians(
        data[["latitude", "longitude"]].to_numpy()
    )

    for score_name, tags in feature_queries.items():
        try:
            pois = ox.features_from_place(city, tags)

            # Convert polygons to centroids using a projected CRS to avoid
            # geographic CRS centroid warnings.
            if not pois.empty:
                pois = pois.copy()
                projected = pois.to_crs(epsg=32644)  # UTM zone covering Chennai
                projected["geometry"] = projected.geometry.centroid
                pois = projected.to_crs(epsg=4326)

            poi_coords = np.radians(
                np.array([[g.y, g.x] for g in pois.geometry])
            )

            tree = BallTree(poi_coords, metric="haversine")
            dist, _ = tree.query(candidate_coords, k=1)

            dist_km = dist.flatten() * 6371
            score = 1 / (1 + dist_km)

            if score_name == "TrafficScore":
                road_weights = {
                    "motorway": 1.0,
                    "trunk": 0.9,
                    "primary": 0.8,
                    "secondary": 0.65,
                    "tertiary": 0.5,
                    "residential": 0.3,
                    "service": 0.2,
                }

                if "highway" in pois.columns:
                    weights = pois["highway"].map(road_weights).fillna(0.4)
                    score = score * weights.mean()

            data[score_name] = score

        except Exception:
            data[score_name] = 0

    cluster_sizes = data["Cluster"].value_counts()
    data["RoadConnectivity"] = data["Cluster"].map(cluster_sizes)
    data["RoadConnectivity"] = (
        data["RoadConnectivity"] /
        data["RoadConnectivity"].max()
    )

    centers = model.cluster_centers_
    distances = []

    for _, row in data.iterrows():
        center = centers[int(row["Cluster"])]
        d = ((row["latitude"] - center[0]) ** 2 +
             (row["longitude"] - center[1]) ** 2) ** 0.5
        distances.append(d)

    scaler = MinMaxScaler()
    data["DistanceScore"] = scaler.fit_transform(pd.DataFrame(distances))
    data["DistanceScore"] = 1 - data["DistanceScore"]

    data["RecommendationScore"] = (
        0.18 * data["DistanceScore"] +
        0.10 * data["RoadConnectivity"] +
        0.12 * data["MallScore"] +
        0.10 * data["HospitalScore"] +
        0.10 * data["MetroScore"] +
        0.08 * data["RailwayScore"] +
        0.05 * data["BusTerminalScore"] +
        0.10 * data["ITParkScore"] +
        0.05 * data["CommercialScore"] +
        0.05 * data["CompanyScore"] +
        0.07 * data["TrafficScore"]
    )

    data["RecommendationScore"] = scaler.fit_transform(
        data[["RecommendationScore"]]
    )

    # Remove duplicate coordinates
    data = data.drop_duplicates(subset=["latitude", "longitude"])
    # Remove candidate locations that are too close to existing charging stations
    if not stations.empty:
        station_coords = np.radians(
            stations.geometry.apply(lambda g: [g.y, g.x]).tolist()
        )

        station_tree = BallTree(
            station_coords,
            metric="haversine"
        )

        candidate_coords = np.radians(
            data[["latitude", "longitude"]].to_numpy()
        )

        min_distance_km = 2.0
        radius = min_distance_km / 6371.0

        keep_rows = []

        for i, coord in enumerate(candidate_coords):
            nearby = station_tree.query_radius([coord], r=radius)[0]
            if len(nearby) == 0:
                keep_rows.append(i)

        data = data.iloc[keep_rows].reset_index(drop=True)

    # Remove candidate locations that are too close to each other (minimum 1 km)
    coords = np.radians(data[["latitude", "longitude"]].to_numpy())
    tree = BallTree(coords, metric="haversine")

    keep_indices = []
    visited = set()

    radius = 1.0 / 6371.0  # 1 km in radians

    for i, point in enumerate(coords):
        if i in visited:
            continue

        keep_indices.append(i)
        neighbors = tree.query_radius([point], r=radius)[0]

        for n in neighbors:
            if n != i:
                visited.add(int(n))

    data = data.iloc[keep_indices].reset_index(drop=True)

    # Select recommendations while enforcing minimum spacing
    data = data.sort_values(
        "RecommendationScore", ascending=False
    ).reset_index(drop=True)

    selected = []
    min_distance_km = 2.0

    for _, row in data.iterrows():
        keep = True

        for chosen in selected:
            d = ((row["latitude"] - chosen["latitude"]) ** 2 +
                 (row["longitude"] - chosen["longitude"]) ** 2) ** 0.5 * 111

            if d < min_distance_km:
                keep = False
                break

        if keep:
            selected.append(row)

        if len(selected) >= 50:
            break

    data = pd.DataFrame(selected).reset_index(drop=True)

    def build_reason(row):
        reasons = []

        if row["RoadConnectivity"] >= 0.70:
            reasons.append("excellent road connectivity")

        if row["TrafficScore"] >= 0.60:
            reasons.append("high traffic area")

        if row["ITParkScore"] >= 0.60:
            reasons.append("near an IT park")

        if row["MetroScore"] >= 0.60:
            reasons.append("close to a metro station")

        if row["RailwayScore"] >= 0.60:
            reasons.append("near a railway station")

        if row["MallScore"] >= 0.60:
            reasons.append("close to a shopping mall")

        if row["HospitalScore"] >= 0.60:
            reasons.append("good access to hospitals")

        if row["CommercialScore"] >= 0.60:
            reasons.append("located in a commercial area")

        if row["CompanyScore"] >= 0.60:
            reasons.append("near corporate offices")

        if row["DistanceScore"] >= 0.60:
            reasons.append("well separated from existing charging stations")

        if not reasons:
            return "Balanced location with good overall accessibility and infrastructure."

        sentence = ", ".join(reasons[:-1])
        if len(reasons) > 1:
            sentence += " and " + reasons[-1]
        else:
            sentence = reasons[0]

        return "Recommended because it has " + sentence + "."

    data["Recommendation"] = data.apply(build_reason, axis=1)
    data["Rank"] = data.index + 1

    return data, centers

