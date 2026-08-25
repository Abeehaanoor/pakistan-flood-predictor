import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(page_title="Pakistan Flood Risk Predictor", page_icon="🌊", layout="centered")

st.title("Pakistan Flood Risk Predictor")
st.write(
    "Estimates **seasonal flood susceptibility** for a location in Pakistan, based on a model "
    "trained on real 2022 Sentinel-1 satellite flood data. This tool answers *'given this "
    "location's terrain and expected monsoon rainfall, how much flood risk does it carry?'* "
    "— it is **not** a real-time, day-of-storm alarm. See the note at the bottom for why."
)

# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
@st.cache_resource
def load_pipeline():
    return joblib.load("flood_risk_pipeline.joblib")

pipeline = load_pipeline()

# ---------------------------------------------------------
# City presets (nearest-grid-cell approximations)
# Replace/expand these with real lookups from your training
# dataset for better accuracy - see note in the code comment below.
# ---------------------------------------------------------
CITY_PRESETS = {
    "Karachi (Sindh)":        {"lat": 24.8607, "lon": 67.0011, "elevation_m": 8,    "dist_river_km": 5,  "province": "Sindh"},
    "Hyderabad (Sindh)":      {"lat": 25.3960, "lon": 68.3578, "elevation_m": 28,   "dist_river_km": 2,  "province": "Sindh"},
    "Sukkur (Sindh)":         {"lat": 27.7052, "lon": 68.8574, "elevation_m": 65,   "dist_river_km": 1,  "province": "Sindh"},
    "Lahore (Punjab)":        {"lat": 31.5497, "lon": 74.3436, "elevation_m": 217,  "dist_river_km": 12, "province": "Punjab"},
    "Multan (Punjab)":        {"lat": 30.1575, "lon": 71.5249, "elevation_m": 122,  "dist_river_km": 8,  "province": "Punjab"},
    "Islamabad":              {"lat": 33.6844, "lon": 73.0479, "elevation_m": 540,  "dist_river_km": 20, "province": "Islamabad"},
    "Peshawar (KPK)":         {"lat": 34.0151, "lon": 71.5805, "elevation_m": 359,  "dist_river_km": 8,  "province": "Khyber-Pakhtunkhwa"},
    "Quetta (Balochistan)":   {"lat": 30.1798, "lon": 66.9750, "elevation_m": 1680, "dist_river_km": 30, "province": "Balochistan"},
    "Custom location":        None,
}

# NOTE for accuracy: instead of hand-typed presets, consider looking up the
# nearest row in your training CSV by lat/lon distance and pulling its real
# elevation_m / distance_to_river_km. That reuses data you already trust
# rather than approximating it by hand.

st.subheader("Location")
city = st.selectbox("Select your city (or choose Custom to enter your own location)", list(CITY_PRESETS.keys()))
preset = CITY_PRESETS[city]

if preset is None:
    col1, col2 = st.columns(2)
    with col1:
        latitude = st.number_input("Latitude", value=30.0, format="%.4f")
        elevation_m = st.number_input("Elevation (m)", value=100.0, min_value=0.0)
    with col2:
        longitude = st.number_input("Longitude", value=70.0, format="%.4f")
        dist_river_km = st.number_input("Distance to nearest river (km)", value=10.0, min_value=0.0)
    province = st.selectbox(
        "Province",
        ["Sindh", "Punjab", "Khyber-Pakhtunkhwa", "Balochistan", "Islamabad",
         "Gilgit-Baltistan", "AzadKashmir", "FederallyAdministeredTribalAr", "Unknown"],
    )
else:
    latitude = preset["lat"]
    longitude = preset["lon"]
    elevation_m = preset["elevation_m"]
    dist_river_km = preset["dist_river_km"]
    province = preset["province"]
    st.caption(
        f"Using approximate values for {city}: elevation {elevation_m} m, "
        f"~{dist_river_km} km from nearest river."
    )

# ---------------------------------------------------------
# Rainfall inputs
# ---------------------------------------------------------
st.subheader("Rainfall")

rainfall_mm = st.slider(
    "Expected / total monsoon-season rainfall (mm)",
    min_value=0, max_value=1500, value=400, step=10,
    help="This is the strongest driver of the model's prediction — it reflects seasonal "
         "rainfall totals, similar to what was observed during the 2022 monsoon.",
)

with st.expander("Optional: recent short-term rainfall (context only, minor effect on prediction)"):
    st.caption(
        "These figures had low predictive weight in the trained model, since the 2022 dataset "
        "captures a single flood event and doesn't contain enough short-term rainfall variation "
        "to learn a strong day-to-day relationship. They're shown for context, not as the main driver."
    )
    rain_24h = st.number_input("Rainfall last 24h (mm)", value=0.0, min_value=0.0)
    rain_3day = st.number_input("Rainfall last 3 days (mm)", value=0.0, min_value=0.0)
    rain_7day = st.number_input("Rainfall last 7 days (mm)", value=0.0, min_value=0.0)

col3, col4 = st.columns(2)
with col3:
    temperature_c = st.number_input("Temperature (°C)", value=30.0)
with col4:
    humidity_pct = st.number_input("Humidity (%)", value=55.0, min_value=0.0, max_value=100.0)

# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------
if st.button("Predict Flood Risk", type="primary"):
    input_df = pd.DataFrame([{
        "latitude": latitude,
        "longitude": longitude,
        "elevation_m": elevation_m,
        "rainfall_mm": rainfall_mm,
        "distance_to_river_km": dist_river_km,
        "rain_24h": rain_24h,
        "rain_3day": rain_3day,
        "rain_7day": rain_7day,
        "temperature_c": temperature_c,
        "humidity_pct": humidity_pct,
        "province": province,
    }])

    prediction = pipeline.predict(input_df)[0]
    prediction = max(0.0, float(prediction))  # guard against tiny negative outputs

    st.subheader("Predicted Flood Extent")
    st.metric("Estimated area flooded", f"{prediction:.1f}%")

    # -------------------------------------------------
    # Alarm — thresholds set from the real distribution of percent_flooded
    # across all 10,641 training locations:
    #   50th percentile = 0.192%   80th percentile = 1.942%   95th percentile = 18.018%
    # The data is heavily skewed — most locations saw little to no flooding in 2022,
    # so "high risk" is calibrated against the worst 5% of observed locations, not
    # round numbers like 30%.
    # -------------------------------------------------
    if prediction >= 18.0:
        st.error("🚨 HIGH FLOOD RISK — This location's predicted flood extent is in the top "
                  "5% of locations observed during the 2022 floods. Monitor official flood "
                  "advisories closely.")
    elif prediction >= 1.9:
        st.warning("⚠️ MODERATE FLOOD RISK — Predicted flood extent is above the typical "
                    "range seen across most locations in 2022. Keep an eye on monsoon "
                    "forecasts and river levels.")
    elif prediction >= 0.2:
        st.info("ℹ️ LOW-MODERATE FLOOD RISK — Slightly above the median location in the 2022 "
                "dataset, but well below severely affected areas.")
    else:
        st.success("✅ LOW FLOOD RISK — Predicted flood extent is at or below the median "
                    "location in the 2022 dataset.")

    st.caption(
        "Model: XGBoost, trained on 10,641 locations from the 2022 Pakistan floods. "
        "Evaluated with a spatial train/test split (R² = 0.743, MAE = 2.11). "
        "Based on a single flood event — best interpreted as **seasonal flood susceptibility** "
        "given expected monsoon conditions, not a real-time forecast or day-of-storm alarm."
    )

st.divider()
with st.expander("Why isn't this a real-time flood alarm?"):
    st.write(
        "Early versions of this app treated 24h/3-day/7-day rainfall as inputs to a live-style "
        "'flood alarm.' Checking the model's feature importances showed those short-term rainfall "
        "features carry very little weight — the model relies mainly on **total seasonal rainfall**, "
        "**elevation**, and **province**, which reflect the terrain and monsoon patterns behind the "
        "2022 flood event it was trained on. Since that event is the only flood the model has seen, "
        "there isn't enough short-term rainfall variation in the data to reliably learn a "
        "'today's storm → today's flood' relationship. Rather than present a misleading real-time "
        "alarm, this app is scoped to what the model can actually support: seasonal susceptibility."
    )
