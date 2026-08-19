import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Pakistan Flood Risk Predictor", page_icon="🌊", layout="centered")

@st.cache_resource
def load_pipeline():
    return joblib.load("flood_risk_pipeline.joblib")

pipeline = load_pipeline()

st.title("Pakistan Flood Risk Predictor")
st.write("Enter conditions for a location to predict flood risk, using a model trained on real 2022 satellite flood data.")

provinces = ["Sindh", "Balochistan", "Punjab", "Khyber-Pakhtunkhwa",
             "AzadKashmir", "Gilgit-Baltistan", "Islamabad",
             "FederallyAdministeredTribalAr", "Unknown"]

col1, col2 = st.columns(2)
with col1:
    province = st.selectbox("Province", provinces, index=provinces.index("Sindh"))
    latitude = st.number_input("Latitude", value=28.09, format="%.4f")
    longitude = st.number_input("Longitude", value=67.76, format="%.4f")
    elevation_m = st.number_input("Elevation (m)", value=51.0)
    rainfall_mm = st.number_input("Total monsoon rainfall (mm)", value=800.0)
    distance_to_river_km = st.number_input("Distance to nearest river (km)", value=80.0)

with col2:
    rain_24h = st.number_input("Rainfall last 24h (mm)", value=5.0)
    rain_3day = st.number_input("Rainfall last 3 days (mm)", value=15.0)
    rain_7day = st.number_input("Rainfall last 7 days (mm)", value=40.0)
    temperature_c = st.number_input("Temperature (°C)", value=30.0)
    humidity_pct = st.number_input("Humidity (%)", value=55.0)

if st.button("Predict Flood Risk", type="primary"):
    input_df = pd.DataFrame([{
        "latitude": latitude,
        "longitude": longitude,
        "elevation_m": elevation_m,
        "rainfall_mm": rainfall_mm,
        "distance_to_river_km": distance_to_river_km,
        "rain_24h": rain_24h,
        "rain_3day": rain_3day,
        "rain_7day": rain_7day,
        "temperature_c": temperature_c,
        "humidity_pct": humidity_pct,
        "province": province
    }])

    prediction = pipeline.predict(input_df)[0]
    prediction = max(0, min(100, prediction))

    st.subheader("Predicted Flood Extent")
    st.metric(label="", value=f"{prediction:.1f}%")

    if prediction < 15:
        st.success("Low risk")
    elif prediction < 40:
        st.warning("Moderate risk")
    else:
        st.error("Severe risk")

st.caption("Model: XGBoost, trained on 10,641 locations from the 2022 Pakistan floods. "
           "Evaluated with a spatial train/test split (R² = 0.743). "
           "Based on a single flood event — best interpreted as flood susceptibility under monsoon-like conditions, not a general forecast.")
