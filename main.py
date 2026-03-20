import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page config
st.set_page_config(page_title="FactoryGuard AI", layout="wide")

# Load model
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

# Custom CSS (Luxury look)
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1, h2, h3 {
    color: #00f5d4;
}
.stButton>button {
    background-color: #00f5d4;
    color: black;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🏭 FactoryGuard AI")
st.caption("Real-Time Predictive Maintenance System")

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Input Sensor Data")
    
    temperature = st.slider("Temperature", 250, 400, 300)
    vibration = st.slider("Vibration", 0, 100, 50)
    pressure = st.slider("Pressure", 50, 200, 100)
    
    temp_roll_mean = st.slider("Temp Rolling Mean", 250, 400, 300)
    vib_roll_std = st.slider("Vibration Std", 0, 50, 10)
    temp_lag1 = st.slider("Previous Temperature", 250, 400, 300)
    vib_lag1 = st.slider("Previous Vibration", 0, 100, 50)

    predict_btn = st.button("🚀 Predict Failure")

with col2:
    st.subheader("📊 Live Status Dashboard")

    if predict_btn:

        data_dict = {
            "temperature": temperature,
            "vibration": vibration,
            "pressure": pressure,
            "temp_roll_mean": temp_roll_mean,
            "vib_roll_std": vib_roll_std,
            "temp_lag1": temp_lag1,
            "vib_lag1": vib_lag1
        }

        values = [data_dict[col] for col in columns]
        arr = np.array(values).reshape(1, -1)

        prob = model.predict_proba(arr)[0][1]
        prediction = int(prob > 0.3)

        # ---------------------------
        # BIG STATUS CARD
        # ---------------------------
        if prediction == 1:
            st.error("⚠️ HIGH RISK OF FAILURE")
        else:
            st.success("✅ MACHINE IS SAFE")

        # ---------------------------
        # PROGRESS BAR (Gauge feel)
        # ---------------------------
        st.write("### 🔍 Failure Probability")
        st.progress(float(prob))

        st.metric(label="Failure Probability", value=f"{prob:.2f}")

        # ---------------------------
        # CHART
        # ---------------------------
        chart_data = pd.DataFrame({
            "Sensor": ["Temperature", "Vibration", "Pressure"],
            "Value": [temperature, vibration, pressure]
        })

        st.write("### 📈 Sensor Overview")
        st.bar_chart(chart_data.set_index("Sensor"))

        # ---------------------------
        # EXTRA INSIGHT
        # ---------------------------
        if prob > 0.7:
            st.warning("🚨 Immediate maintenance required!")
        elif prob > 0.4:
            st.info("⚡ Monitor machine closely.")
        else:
            st.success("🟢 System operating normally.")