# 🏭 FactoryGuard AI – Predictive Maintenance System

## 📌 Project Overview
FactoryGuard AI is an end-to-end machine learning project designed to predict machine failures using time-series sensor data.  
The system analyzes parameters like temperature, vibration, and pressure to identify potential failures before they occur.

This helps industries reduce downtime, improve efficiency, and optimize maintenance schedules.

---

## 🎯 Objectives
- Predict machine failures in advance
- Reduce unexpected breakdowns
- Provide real-time predictions through API and UI
- Improve maintenance decision-making

---

## 📊 Dataset
- Synthetic time-series dataset generated using Python
- Features include:
  - Temperature
  - Vibration
  - Pressure
  - Rolling Mean (trend)
  - Rolling Standard Deviation (variation)
  - Lag Features (previous values)

---

## ⚙️ Tech Stack
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn
- **Model:** Random Forest Classifier
- **Backend:** FastAPI
- **Frontend/UI:** Streamlit
- **Model Saving:** Joblib

---

## 🧠 Machine Learning Approach

### 🔹 Feature Engineering
- Rolling Mean → Captures trends
- Rolling Std → Detects fluctuations
- Lag Features → Captures past behavior

### 🔹 Model
- Random Forest (baseline model)

### 🔹 Handling Imbalance
- Used class weights to improve failure detection

### 🔹 Threshold Tuning
- Adjusted threshold to 0.3 to improve recall

---

## 📊 Model Performance
- Accuracy: ~83%
- Recall (Failure): ~95% ✅
- Precision (Failure): ~32%

> The model prioritizes high recall to avoid missing failures.

---

## 🚀 Project Workflow
