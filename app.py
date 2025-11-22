# app.py
import streamlit as st
import requests
import json
import os

# API URL (works locally and on Streamlit Cloud when using the proxy)
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")  # change if needed

# Load performance & feature importance (you'll create these files)
with open("model_performance.json") as f:
    perf = json.load(f)

st.title("🚀 My Machine Learning Model Predictor")

st.sidebar.header("Model Performance")
st.sidebar.metric("Test Accuracy", f"{perf['test_accuracy']:.3f}")
st.sidebar.metric("Test ROC-AUC", f"{perf.get('test_auc', 'N/A'):.3f}")

st.sidebar.header("Top 5 Important Features")
for feat, imp in perf["top_features"]:
    st.sidebar.write(f"**{feat}**: `{imp:.4f}`")

st.header("Make a Prediction")
st.write("Enter values for each feature:")

# Load feature names
with open("feature_names.json") as f:
    feature_names = json.load(f)

inputs = []
cols = st.columns(3)
for i, feat in enumerate(feature_names):
    with cols[i % 3]:
        val = st.number_input(feat, value=0.0, format="%.4f", key=feat)
        inputs.append(val)

if st.button("Predict", type="primary"):
    with st.spinner("Predicting..."):
        response = requests.post(API_URL, json={"features": inputs})
        if response.status_code == 200:
            result = response.json()
            pred = result["prediction"]
            st.success(f"**Prediction: {pred}**")
            if "predicted_probability" in result:
                prob = result["predicted_probability"]
                st.metric("Confidence", f"{prob:.3f}")
            if "probabilities" in result:
                st.write("Class probabilities:", result["probabilities"])
        else:
            st.error(f"API Error: {response.text}")