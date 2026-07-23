

# 🚗 AI-Based EV Charging Station Recommendation System

## 📌 Project Overview

This project recommends the best locations for installing new Electric Vehicle (EV) charging stations using Artificial Intelligence (AI) and Geospatial Analytics.

The system combines OpenStreetMap (OSM) data, clustering algorithms, and multiple location-based factors to identify high-potential charging station locations.

---

## 🎯 Objectives

- Identify suitable locations for new EV charging stations.
- Reduce competition with existing charging stations.
- Improve accessibility for EV users.
- Support data-driven urban planning.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- GeoPandas
- OSMnx
- Folium
- BallTree (Nearest Neighbor Search)
- OpenStreetMap (OSM)

---

## 🧠 AI & GIS Workflow

1. Load candidate locations.
2. Download road network and existing charging stations from OpenStreetMap.
3. Perform K-Means clustering.
4. Calculate feature scores such as:
   - Distance Score
   - Road Connectivity Score
   - Mall Score
   - Hospital Score
   - Metro Score
   - Railway Score
   - Bus Station Score
   - IT Park Score
   - Commercial Area Score
   - Company Score
   - Traffic Score
5. Compute the final Recommendation Score.
6. Remove duplicate and nearby recommendations.
7. Export recommendations to CSV.
8. Visualize results on an interactive Folium map.

---

## 📂 Project Structure

```
EV-Charging-Station-Recommendation/
│── data/
│── outputs/
│── src/
│   ├── main.py
│   ├── recommendation.py
│   ├── candidate_locations.py
│   └── visualize_stations.py
│── README.md
```

---

## ▶️ How to Run

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 src/main.py
```

---

## 📊 Output

Running the project generates:

- `outputs/recommendations.csv`
- `outputs/final_map.html`

---

## 📸 Project Screenshots

### 🗺️ Interactive Recommendation Map

![Interactive Recommendation Map](screenshots/final_map.png)

### 📄 Recommendation Results (CSV)

![Recommendation Results](screenshots/recommendations_csv.png)

---

## ✨ Current Features

- AI-based recommendation engine
- K-Means clustering
- Multi-factor scoring
- OpenStreetMap integration
- Traffic-aware recommendations
- Duplicate removal
- Interactive map visualization

---

## 🏗️ System Workflow

```
OpenStreetMap Data
        │
        ▼
Candidate Location Generation
        │
        ▼
Feature Extraction
(Hospitals, Malls, Metro,
Roads, Companies, Traffic)
        │
        ▼
AI Recommendation Engine
        │
        ▼
Recommendation Score
        │
        ▼
Duplicate Removal
        │
        ▼
CSV Output + Interactive Map
```

---

## 📈 Expected Output

Each recommended location includes:

- Rank
- Latitude
- Longitude
- Recommendation Score

---

## 🚀 Future Improvements

- Power infrastructure analysis
- Parking availability scoring
- Population density analysis
- Demand heatmap
- Evaluation metrics dashboard

---

## 👨‍💻 Author

**Koushal Edupulapati**

B.Tech Artificial Intelligence & Machine Learning
SRM Institute of Science and Technology