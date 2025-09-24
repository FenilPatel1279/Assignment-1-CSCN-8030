import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from src.model import train_model, evaluate_model, show_sample_predictions
from sklearn.metrics import ConfusionMatrixDisplay

# --------------------------
# Streamlit Config
# --------------------------
st.set_page_config(page_title="Ecoflare DSS - Real-Time", layout="wide")
st.markdown("<h1 style='text-align: center; color: red;'>🔥 Ecoflare DSS Dashboard</h1>", unsafe_allow_html=True)
st.caption("Simulated real-time wildfire detection using AI + sensor feeds")

# --------------------------
# Live Data Generator
# --------------------------
def generate_live_data(n=1):
    """Simulate live sensor readings for wildfire features."""
    temperature = np.random.normal(32, 5, n)
    humidity = np.random.normal(40, 10, n)
    smoke_level = np.random.normal(50, 15, n)
    satellite_heat = np.random.normal(70, 20, n)

    fire_risk = (temperature > 34) & (humidity < 35) & (smoke_level > 55) & (satellite_heat > 80)
    fire_risk = fire_risk.astype(int)

    return pd.DataFrame({
        "Temperature": temperature,
        "Humidity": humidity,
        "SmokeLevel": smoke_level,
        "SatelliteHeat": satellite_heat,
        "Wildfire": fire_risk
    })

# Session buffer
if "live_data" not in st.session_state:
    st.session_state.live_data = pd.DataFrame(columns=["Temperature","Humidity","SmokeLevel","SatelliteHeat","Wildfire"])

# --------------------------
# Sidebar Controls
# --------------------------
st.sidebar.header("⚙️ Controls")
update_interval = st.sidebar.slider("Update interval (seconds)", 1, 5, 2)
max_points = st.sidebar.slider("Max data points", 50, 300, 100)

# --------------------------
# Live Data Section
# --------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📡 Latest Sensor Feed")
    placeholder_table = st.empty()

with col2:
    st.subheader("📊 Live Scatterplot")
    placeholder_chart1 = st.empty()

st.subheader("📈 Trends Over Time")
placeholder_chart2 = st.empty()

# --------------------------
# Real-Time Simulation
# --------------------------
for _ in range(15):  # simulate 15 updates
    new_data = generate_live_data()
    st.session_state.live_data = pd.concat([st.session_state.live_data, new_data], ignore_index=True)

    # Limit buffer
    if len(st.session_state.live_data) > max_points:
        st.session_state.live_data = st.session_state.live_data.tail(max_points)

    # Update table (latest 5 rows)
    placeholder_table.dataframe(st.session_state.live_data.tail(5).style.format(precision=2))

    # Chart 1: Scatterplot
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    sns.scatterplot(
        data=st.session_state.live_data,
        x="Temperature", y="SmokeLevel",
        hue="Wildfire", palette="coolwarm", ax=ax1
    )
    ax1.set_title("Temp vs Smoke (Live Data)", fontsize=10)
    placeholder_chart1.pyplot(fig1)

    # Chart 2: Line plot of averages
    rolling = st.session_state.live_data.rolling(window=5).mean()
    fig2, ax2 = plt.subplots(figsize=(7, 3))
    ax2.plot(rolling["Temperature"], label="Temperature (°C)")
    ax2.plot(rolling["Humidity"], label="Humidity (%)")
    ax2.plot(rolling["SmokeLevel"], label="SmokeLevel")
    ax2.plot(rolling["SatelliteHeat"], label="SatelliteHeat")
    ax2.legend(fontsize=8)
    ax2.set_title("Sensor Averages Over Time", fontsize=10)
    placeholder_chart2.pyplot(fig2)

    time.sleep(update_interval)

# --------------------------
# AI Model Section
# --------------------------
st.subheader("🤖 AI Model - Wildfire Prediction")

if len(st.session_state.live_data) > 30:
    model, X_test, y_test, y_pred, y_prob = train_model(st.session_state.live_data)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Records", len(st.session_state.live_data))
    col_b.metric("Accuracy", f"{(y_pred == y_test).mean()*100:.1f}%")
    col_c.metric("Wildfires Detected", int(sum(y_pred)))

    # Confusion Matrix
    fig_cm, ax_cm = plt.subplots(figsize=(3, 3))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues", ax=ax_cm)
    st.pyplot(fig_cm)

    # Predictions
    st.write("🔮 Sample Predictions")
    results = show_sample_predictions(X_test, y_test, y_pred, y_prob, n=5)
    st.dataframe(results.style.format(precision=2))
else:
    st.warning("Waiting for enough live data (min 30 points) before training model...")
