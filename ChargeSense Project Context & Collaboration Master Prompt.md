# MASTER PROJECT CONTEXT — CHARGESENSE

You are joining an ongoing college/hackathon project called **ChargeSense — AI-Driven Site Intelligence for EV Charging Infrastructure**.

Your job is to understand the complete project context below before suggesting or changing anything. Treat the information below as the current source of truth.

---

## 1. PROJECT OVERVIEW

### Project Name
**ChargeSense**

### One-line idea
> “Instead of asking where charging stations already exist, we ask: where should the next charging station be?”

### Problem
The growth of electric vehicles requires accessible and strategically distributed charging infrastructure. However, deciding where to install new charging stations is difficult because charging stations are unevenly distributed and road accessibility and spatial demand vary across a city.

### Proposed Solution
We are building a **Geospatial AI-based EV Charging Station Location Recommendation System**.

The current system:
1. Extracts a city road network from OpenStreetMap.
2. Retrieves existing EV charging stations.
3. Creates candidate locations.
4. Applies K-Means clustering to candidate coordinates.
5. Uses cluster centers as current recommendation points.
6. Displays existing and recommended charging locations on an interactive Folium map.

The current demonstration city is **Chennai, Tamil Nadu, India**.

---

# 2. CURRENT PROJECT STATUS

We currently have a **working end-to-end prototype**.

The following parts have already been implemented and tested:

✅ Python project setup  
✅ Virtual environment  
✅ GitHub repository  
✅ OSMnx road-network extraction  
✅ GeoPandas conversion of road-network data  
✅ OpenStreetMap EV charging-station retrieval  
✅ Folium interactive map  
✅ Candidate-location / sample-coordinate data  
✅ K-Means clustering using Scikit-learn  
✅ Recommendation function returning cluster centers  
✅ Integration of road network + charging stations + clustering  
✅ Final interactive map generation  

The integrated program successfully creates:

```text
outputs/final_map.html
```

The final map displays:

- Road network
- Existing EV charging stations
- Recommended locations generated from K-Means cluster centers

---

# 3. IMPORTANT: WHAT HAS ACTUALLY BEEN IMPLEMENTED

Do NOT assume that every idea in the presentation is already implemented.

The presentation discusses possible future/advanced inputs such as:

- Traffic flow
- EV registration density
- POI density
- Charger utilization
- Grid capacity
- Land-use constraints
- Advanced location scoring
- Streamlit dashboard

These are either proposed enhancements or future scope and are NOT all part of the current working implementation.

The current prototype primarily uses:

- OpenStreetMap road data
- Existing charging-station data
- Candidate latitude/longitude points
- K-Means clustering
- Folium visualization

Do not claim advanced traffic/grid/EV-utilization analysis is already implemented unless the team adds and verifies it.

---

# 4. TEAM WORKFLOW

The project is maintained on GitHub.

Repository:

```text
https://github.com/srajanswami/EV-Charging-Station-Recommendation
```

The team is collaborating through GitHub.

Typical workflow:

```bash
git pull origin main
```

before beginning work.

After making changes:

```bash
git add .
git commit -m "Describe the change"
git push origin main
```

IMPORTANT:
- Do not overwrite another person's work.
- Pull before starting.
- Work on separate modules/files whenever possible.
- Keep the `main` branch working.
- Before modifying an existing file, inspect its current contents.
- Prefer creating a new module when the functionality is independent.
- Tell the team what files you changed.

---

# 5. CURRENT PROJECT STRUCTURE

The current VS Code project is approximately:

```text
EV-Charging-Station-Recommendation/
│
├── data/
│   └── sample_locations.csv
│
├── outputs/
│   ├── chennai_map.html
│   ├── chennai_ev_map.html
│   └── final_map.html
│
├── src/
│   ├── candidate_locations.py
│   ├── charging_stations.py
│   ├── clustering.py
│   ├── main.py
│   ├── recommendation.py
│   └── visualize_stations.py
│
├── venv/
├── README.md
└── requirements.txt
```

There may also be Python cache files such as `__pycache__`.

---

# 6. TECHNOLOGY STACK

### Core
- Python

### Geospatial / Data
- OpenStreetMap
- OSMnx 2.1.1
- GeoPandas
- Pandas
- Shapely
- NetworkX
- PyProj

### Machine Learning
- Scikit-learn
- K-Means clustering
- NumPy

### Visualization
- Folium
- HTML-based interactive maps

### Possible future
- Streamlit dashboard

Do not say Streamlit is implemented unless it has actually been added and tested.

---

# 7. ROAD NETWORK IMPLEMENTATION

We use OSMnx to download the Chennai drivable road network.

Current logic:

```python
import osmnx as ox

city = "Chennai, Tamil Nadu, India"

graph = ox.graph_from_place(
    city,
    network_type="drive"
)
```

The downloaded network produced approximately:

```text
68,658 nodes
174,013 edges
```

This means the graph contains road-network nodes/intersections and road segments.

Then:

```python
nodes, edges = ox.graph_to_gdfs(graph)
```

converts the graph into GeoDataFrames.

The `nodes` GeoDataFrame contains point geometries.

The `edges` GeoDataFrame contains road geometries.

Important:
We initially tried `ox.plot_graph_folium()`, but OSMnx 2.1.1 does not provide that function.

Therefore, we use:

```python
folium.GeoJson(
    edges.to_json(),
    name="Road Network"
).add_to(m)
```

to display roads.

---

# 8. EV CHARGING STATION RETRIEVAL

Existing charging stations are retrieved directly from OpenStreetMap.

Current code concept:

```python
tags = {
    "amenity": "charging_station"
}

stations = ox.features_from_place(
    city,
    tags
)
```

For Chennai, this query returned:

```text
4 explicitly tagged charging stations
```

Important:
This should be described as “4 charging stations explicitly tagged in the queried OpenStreetMap data”, NOT as “Chennai has only 4 charging stations.”

The retrieved records contain geometry and other OpenStreetMap attributes.

---

# 9. CURRENT CHARGING-STATION VISUALIZATION

Each charging station is plotted as a marker:

```python
for _, row in stations.iterrows():
    point = row.geometry

    folium.Marker(
        [point.y, point.x],
        popup=row.get("name", "Charging Station"),
        icon=folium.Icon(
            color="green",
            icon="bolt",
            prefix="fa"
        )
    ).add_to(m)
```

Therefore:

🟢 Green marker = existing charging station

---

# 10. CANDIDATE LOCATIONS

The project uses candidate geographic points with latitude and longitude.

A CSV exists at:

```text
data/sample_locations.csv
```

The expected structure is approximately:

```text
latitude,longitude
13.xxxx,80.xxxx
...
```

The K-Means implementation currently operates on these latitude and longitude columns.

The candidate-location generation module also exists:

```text
src/candidate_locations.py
```

Do not create a duplicate candidate dataset unless necessary.

First inspect the existing CSV and current script.

---

# 11. CURRENT K-MEANS IMPLEMENTATION

The current clustering logic is based on:

```python
import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_csv("data/sample_locations.csv")

X = data[["latitude", "longitude"]]

model = KMeans(
    n_clusters=3,
    random_state=42
)

model.fit(X)

data["Cluster"] = model.labels_

print(data)

print("\nCluster Centers:")
print(model.cluster_centers_)
```

So currently:

```text
Latitude + Longitude
        ↓
K-Means
        ↓
Cluster labels
        ↓
Cluster centers
```

The current default number of clusters is:

```text
3
```

---

# 12. RECOMMENDATION MODULE

There is a file:

```text
src/recommendation.py
```

The reusable function currently looks like:

```python
import pandas as pd
from sklearn.cluster import KMeans


def recommend_locations(csv_file, n_clusters=3):
    data = pd.read_csv(csv_file)

    X = data[["latitude", "longitude"]]

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42
    )

    model.fit(X)

    data["Cluster"] = model.labels_

    return data, model.cluster_centers_
```

This function currently returns:

```text
clustered_data
cluster_centers
```

IMPORTANT:
Despite its filename, this function currently performs clustering and returns cluster centers. It is NOT yet a sophisticated location-scoring engine.

A major next improvement is to make the recommendation logic more realistic.

---

# 13. CURRENT MAIN PIPELINE

The current integrated `main.py` loads:

```text
OSM road network
        ↓
Existing charging stations
        ↓
sample_locations.csv
        ↓
recommend_locations()
        ↓
K-Means cluster centers
        ↓
Folium map
```

The current integrated `main.py` is conceptually:

```python
import folium
import osmnx as ox

from recommendation import recommend_locations

city = "Chennai, Tamil Nadu, India"

print("Loading road network...")

graph = ox.graph_from_place(
    city,
    network_type="drive"
)

nodes, edges = ox.graph_to_gdfs(graph)

print("Loading charging stations...")

tags = {"amenity": "charging_station"}

stations = ox.features_from_place(
    city,
    tags
)

print("Running clustering...")

clustered_data, centers = recommend_locations(
    "data/sample_locations.csv"
)

m = folium.Map(
    location=[13.0827, 80.2707],
    zoom_start=11
)

folium.GeoJson(
    edges.to_json()
).add_to(m)

for _, row in stations.iterrows():
    point = row.geometry

    folium.Marker(
        [point.y, point.x],
        popup=row.get("name", "Charging Station"),
        icon=folium.Icon(
            color="green",
            icon="bolt",
            prefix="fa"
        )
    ).add_to(m)

for center in centers:
    folium.Marker(
        [center[0], center[1]],
        popup="Recommended Location",
        icon=folium.Icon(
            color="red",
            icon="star"
        )
    ).add_to(m)

m.save("outputs/final_map.html")

print("Project completed successfully!")
```

This integrated version has been successfully run.

---

# 14. CURRENT FINAL MAP

The generated file is:

```text
outputs/final_map.html
```

It opens in a web browser.

The current visualization contains:

🛣️ Road network

🟢 Existing charging stations

🔴 Recommended locations

The current recommended markers correspond to K-Means cluster centers.

IMPORTANT:
A K-Means centroid is a mathematical point. It is not necessarily located on an actual road or feasible plot of land.

Therefore, a key next improvement is:

```text
K-Means cluster center
        ↓
Find nearest real road/candidate node
        ↓
Use that point as practical recommendation
```

---

# 15. CURRENT PROJECT LIMITATIONS

The current model is a prototype.

It currently does NOT yet robustly model:

- Population density
- EV registration density
- Actual traffic volume
- Charger utilization
- Parking availability
- Shopping malls
- Offices
- Petrol pumps
- Electricity-grid capacity
- Land availability
- Construction cost
- Revenue/profitability
- Real charging demand
- Road-network travel-time demand

Therefore, recommendations should currently be described as:

> “K-Means-based spatial candidate recommendations”

rather than claiming they are economically optimal locations.

---

# 16. CURRENT PRESENTATION

The team has a final PPT for **EDGE NOVA’26**.

Project title:

**ChargeSense**

Subtitle:

**AI-Driven Site Intelligence for EV Charging Infrastructure**

Tagline:

> “Instead of asking where charging stations already exist, we ask: where should the next charging station be?”

Team:
- Srajan Swami
- Shashwat
- Sidharth S Nair

The PPT currently contains these main sections:

1. Title
2. Problem Statement
3. Abstract
4. Motivation
5. Technical Stack
6. Completion Status

The current Problem Statement focuses on uneven charging infrastructure, road accessibility and uneven spatial demand.

The current Motivation emphasizes moving from reactive mapping to proactive infrastructure planning.

The Technical Stack slide mentions:

- Python
- OpenStreetMap
- OSMnx
- GeoPandas
- Pandas
- Scikit-learn
- NumPy
- Folium
- Streamlit

IMPORTANT:
Streamlit should only be presented as implemented if it is actually completed. Otherwise treat it as future scope.

The Completion Status currently describes:

Completed:
- OSM road and charging-station data
- Geospatial processing
- K-Means clustering
- Candidate locations
- Interactive map

In progress / future:
- Location scoring and ranking
- Recommendation pipeline improvements
- Final dashboard
- Testing / deployment

---

# 17. HOW THE PROJECT SHOULD BE EXPLAINED

Use this simple architecture when explaining:

```text
OpenStreetMap
      ↓
OSMnx
      ↓
Road Network + Existing EV Stations
      ↓
GeoPandas / Pandas
      ↓
Candidate Locations
      ↓
K-Means Clustering
      ↓
Cluster Centers
      ↓
Recommendation Layer
      ↓
Folium Interactive Map
```

---

# 18. TEAM RESPONSIBILITIES SO FAR

### Srajan / Project Lead
Main work:
- Project setup
- Python environment
- OSMnx road-network extraction
- GeoPandas processing
- EV charging station retrieval
- Folium visualization
- Main integration
- GitHub coordination
- Debugging and testing
- Understanding the full technical pipeline
- Presentation responsibility for Motivation, Technical Stack and Completion Status

### Other team members
The other team members have worked on:
- K-Means clustering
- Recommendation module
- Project documentation / presentation work

When adding new work, avoid overwriting the modules another teammate owns.

---

# 19. WHAT THE NEW COLLABORATOR SHOULD DO

You are the newly added collaborator.

DO NOT immediately rewrite the project.

First:

### Step 1
Clone or pull the GitHub repository.

### Step 2
Inspect:

```text
src/
data/
outputs/
requirements.txt
```

### Step 3
Read:

```text
src/main.py
src/recommendation.py
src/clustering.py
src/charging_stations.py
src/candidate_locations.py
```

### Step 4
Understand the current working pipeline.

### Step 5
Run:

```bash
python src/main.py
```

and verify that:

```text
outputs/final_map.html
```

is generated.

### Step 6
Only after confirming the existing system works should you propose improvements.

---

# 20. HIGH-VALUE NEXT TASKS

The project should now focus on making the recommendation system more realistic.

Priority 1:
### Practical recommendation points
Instead of plotting raw K-Means centroids, select the nearest actual candidate/road node to each centroid.

Priority 2:
### Recommendation scoring
Create a score using available spatial features.

Possible factors:

- Distance from existing charging station
- Road accessibility
- Spatial demand proxy
- Cluster density

Priority 3:
### Remove duplicate / poor recommendations
Ensure recommended sites are reasonably separated and do not overlap existing charging stations.

Priority 4:
### Save outputs
Produce:

```text
outputs/clustered_locations.csv
outputs/recommended_locations.csv
outputs/final_map.html
```

Priority 5:
### Improve visualization
Add:
- map legend
- clear marker labels
- cluster layers
- recommendation ranking

Priority 6:
### Optional future scope
Only after the core system is stable:
- Traffic
- Population
- POIs
- EV demand
- Grid proximity
- Streamlit dashboard

---

# 21. IMPORTANT DEVELOPMENT RULE

Because the team has a short deadline:

Do NOT over-engineer the project.

Do not replace the entire architecture unless necessary.

Prefer small modular improvements.

Before modifying a file, explain:

1. What is wrong?
2. Why does it need changing?
3. What file will change?
4. What will the new output be?

Keep the existing working prototype intact.

---

# 22. HOW THE AI SHOULD HELP YOU

Act as a senior software engineer and project mentor.

When I ask you to modify something:

- First inspect the existing architecture described above.
- Do not assume features are implemented when they are only planned.
- Give code that is compatible with our current package versions.
- Prefer OSMnx 2.x-compatible code.
- Keep modules reusable.
- Avoid duplicate code.
- Explain integration points clearly.
- When changing a file, give the complete replacement file unless only a small edit is needed.
- Tell me exactly where to save the file.
- Tell me the exact command to run.
- Tell me what output I should expect.
- Help debug errors using the full traceback.
- Keep Git collaboration in mind.
- Never delete working features without a reason.

---

# 23. FINAL GOAL

The final system should become:

```text
User
  ↓
Choose / analyze city
  ↓
Retrieve geospatial data
  ↓
Analyze roads + existing charging infrastructure
  ↓
Generate candidate locations
  ↓
K-Means spatial clustering
  ↓
Recommendation / scoring
  ↓
Rank best sites
  ↓
Interactive map
```

The key message of the project is:

> **ChargeSense converts geospatial data into actionable recommendations for where the next EV charging station should be installed.**

Start by understanding the existing codebase and getting the current prototype running. Do NOT immediately build new features until you have confirmed the current pipeline works.