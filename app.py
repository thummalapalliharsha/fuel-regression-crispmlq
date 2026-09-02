# =============================================================================
#  PHASE 5 (Deployment) -- production web app for the fuel model.
#  Run from VS Code terminal:   streamlit run app.py
#  It loads the model artifact saved by fuel_regression_crispmlq.py.
# =============================================================================
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Flight Fuel Predictor", page_icon="airplane")

# Load the trained pipeline once (cached across reruns).
@st.cache_resource
def load_model():
    return joblib.load("fuel_model.joblib")

model = load_model()

st.title("Flight Fuel-Burn Predictor")
st.caption("Linear regression, served the CRISP-ML(Q) way. Enter a flight to estimate fuel.")

# --- Inputs (the same raw columns the model was trained on) ------------------
col1, col2 = st.columns(2)
with col1:
    distance_km = st.number_input("Flight distance (km)", 300, 5000, 2200, step=50)
    payload_tonnes = st.number_input("Payload (tonnes)", 0.0, 25.0, 14.0, step=0.5)
    aircraft_type = st.selectbox("Aircraft type", ["A320", "B737"])
with col2:
    headwind_kts = st.number_input("Headwind (kts, negative = tailwind)", -50, 60, 12)
    cruise_altitude_ft = st.number_input("Cruise altitude (ft)", 28000, 42000, 35000, step=500)

# --- Predict -----------------------------------------------------------------
if st.button("Predict fuel"):
    flight = pd.DataFrame([{
        "distance_km": distance_km,
        "payload_tonnes": payload_tonnes,
        "headwind_kts": headwind_kts,
        "cruise_altitude_ft": cruise_altitude_ft,
        "aircraft_type": aircraft_type,
    }])
    fuel = float(model.predict(flight)[0])
    st.metric("Predicted fuel burn", f"{fuel:,.0f} kg")
    st.info(f"Suggested uplift with 5% contingency: {fuel * 1.05:,.0f} kg")
    st.dataframe(flight)
