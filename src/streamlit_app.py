"""
ChargeSense — Streamlit dashboard.

    streamlit run src/streamlit_app.py

Then open http://localhost:8501 in a browser.

This is a thin shell. It imports the same engine the CLI uses and never
reimplements scoring — spec 32. The expensive half (`build_scored_table`) is
wrapped in Streamlit's cache and keyed on city, so moving a weight slider
recomputes only the cheap half and returns in milliseconds without touching
the network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import (  # noqa: E402
    CITIES,
    DATA_SOURCE_NOTE,
    DEFAULT_CITY,
    DEFAULT_N_CLUSTERS,
    DEFAULT_WEIGHTS,
    DIMENSION_LABELS,
    TRAFFIC_PROXY_NOTE,
)
from recommendation import (  # noqa: E402
    attach_cached_addresses,
    build_scored_table,
    select_recommendations,
)

st.set_page_config(page_title="ChargeSense", page_icon="⚡", layout="wide")


# ------------------------------------------------------------------ cached
@st.cache_data(show_spinner="Loading city data — first run downloads from OpenStreetMap…")
def load_table(city: str, n_clusters: int) -> pd.DataFrame:
    """Expensive half. Cached per (city, n_clusters); never re-run by a slider."""
    return build_scored_table(city, n_clusters=n_clusters)


@st.cache_data(show_spinner=False)
def load_stations_count(city: str) -> int:
    from data_loader import load_stations

    return len(load_stations(city))


# ------------------------------------------------------------------ sidebar
st.sidebar.title("ChargeSense")
st.sidebar.caption("AI-Driven EV Charging Site Intelligence")

city = st.sidebar.selectbox("Select City", list(CITIES), index=list(CITIES).index(DEFAULT_CITY))

n_clusters = st.sidebar.slider("Number of clusters", 2, 15, DEFAULT_N_CLUSTERS)
n_recommendations = st.sidebar.slider("Number of recommendations", 5, 100, 25, step=5)

st.sidebar.markdown("---")
st.sidebar.subheader("Scoring weights")
st.sidebar.caption("Relative importance. Values are normalised, so they need not total 100.")

weights = {}
for key, label in DIMENSION_LABELS.items():
    weights[key] = st.sidebar.slider(
        label, 0.0, 1.0, float(DEFAULT_WEIGHTS[key]), step=0.05, key=f"w_{key}"
    )

if sum(weights.values()) <= 0:
    st.sidebar.error("At least one weight must be above zero.")
    st.stop()

if st.sidebar.button("Reset weights to defaults"):
    for key in DIMENSION_LABELS:
        st.session_state[f"w_{key}"] = float(DEFAULT_WEIGHTS[key])
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(f"**Data source:** OpenStreetMap  \n{DATA_SOURCE_NOTE}")


# ------------------------------------------------------------------ main
st.title("ChargeSense")
st.caption("Instead of asking where charging stations already exist, we ask: where should the next one be?")

try:
    table = load_table(city, n_clusters)
except Exception as exc:
    st.error(f"Could not load data for {city}: {exc}")
    st.info(
        "First run for a city downloads a large road network from OpenStreetMap. "
        "Run `python src/main.py --city " + city + "` once from a terminal to build "
        "the cache, then reload this page."
    )
    st.stop()

if table.empty:
    st.warning(f"No candidate sites available for {city}.")
    st.stop()

result = select_recommendations(table, weights=weights, n_recommendations=n_recommendations)
result = attach_cached_addresses(city, result)

if result.empty:
    st.warning(
        "No sites passed the filters. Every candidate was either within "
        "2 km of an existing charger or had a disqualifying land-use conflict."
    )
    st.stop()

# --- metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Candidate locations", f"{len(table):,}")
c2.metric("Existing stations", f"{load_stations_count(city):,}", help="OSM-tagged, not a complete inventory")
c3.metric("Recommended sites", f"{len(result):,}")
c4.metric("Average score", f"{result['overall_score'].mean():.1f} / 100")

tab_map, tab_table, tab_detail = st.tabs(["Map", "Ranked recommendations", "Site detail"])

# --- map
with tab_map:
    try:
        import streamlit.components.v1 as components
        from data_loader import load_gdfs, load_stations
        from visualization import build_map

        show_roads = st.checkbox("Show road network", value=False,
                                 help="Slower to render on large cities")
        edges = None
        if show_roads:
            _, edges = load_gdfs(city)

        fmap = build_map(city, result, stations=load_stations(city),
                         edges=edges, show_roads=show_roads)
        components.html(fmap.get_root().render(), height=620, scrolling=False)
    except Exception as exc:
        st.error(f"Map could not be rendered: {exc}")

# --- ranked table
with tab_table:
    display = pd.DataFrame({
        "Rank": result["rank"],
        "Location": result["address"] if "address" in result.columns else "",
        "Latitude": result["latitude"].round(6),
        "Longitude": result["longitude"].round(6),
        "Score": result["overall_score"].round(1),
        "Nearest Station": result["km_to_station"].map(
            lambda v: f"{v:.1f} km" if pd.notna(v) and v != float("inf") else "—"
        ),
        "Landmark": result["nearest_landmark"],
        "Feasibility": result["feasibility_band"],
    })
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.download_button(
        "Download recommendations as CSV",
        result.to_csv(index=False).encode("utf-8"),
        file_name=f"chargesense_{city.lower()}_recommendations.csv",
        mime="text/csv",
    )

# --- detail
with tab_detail:
    labels = [
        f"#{int(r['rank'])} — {float(r['overall_score']):.1f}/100"
        for _, r in result.iterrows()
    ]
    picked = st.selectbox("Select a recommended site", labels)
    row = result.iloc[labels.index(picked)]

    left, right = st.columns([1, 1])

    with left:
        st.subheader(f"Site #{int(row['rank'])}")
        st.write("**Address**")
        st.write(row.get("address", "Address unavailable"))
        st.write("**Coordinates**")
        st.code(f"{row['latitude']:.6f}, {row['longitude']:.6f}")
        st.metric("Overall score", f"{row['overall_score']:.1f} / 100")

        st.write("**Site feasibility**")
        conflicts = str(row.get("conflicts", "") or "")
        if conflicts:
            st.warning(f"Conflicts with {conflicts}")
        else:
            st.success(f"No major land-use conflict — {row.get('feasibility_band', '')}")

        st.write("**Why recommended**")
        st.info(row.get("reason", ""))

    with right:
        st.write("**Score breakdown**")
        breakdown = pd.DataFrame(
            {
                "Dimension": [DIMENSION_LABELS[k] for k in DIMENSION_LABELS if k in row.index],
                "Score": [round(float(row[k]), 1) for k in DIMENSION_LABELS if k in row.index],
            }
        )
        st.bar_chart(breakdown.set_index("Dimension"), height=260)

        st.write("**Nearby landmarks** (straight-line distance)")
        from landmarks import format_lines

        for line in format_lines(row):
            st.write(f"- {line}")

st.caption(f"⚠️ {TRAFFIC_PROXY_NOTE}")
