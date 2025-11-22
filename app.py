# app.py
import streamlit as st
import requests
import json
import os

# THIS IS THE ONLY IMPORTANT PART
# On Streamlit Cloud → use the real deployed backend
# Locally → use a dummy URL so it doesn't crash
BASE_URL = st.secrets.get("BASE_URL", "https://your-app-name.streamlit.app")  # change only if needed

# Auto-detect if running on Streamlit Cloud
if "streamlit" in os.environ.get("SERVER_SOFTWARE", "").lower() or os.path.exists("/.streamlit"):
    API_URL = f"{BASE_URL}/backend/predict"
else:
    API_URL = "https://httpbin.org/post"  # dummy endpoint for local testing

# Load JSON files
try:
    with open("feature_names.json") as f:
        feature_names = json.load(f)
    with open("model_performance.json") as f:
        perf = json.load(f)
except Exception as e:
    st.error(f"Missing files: {e}")
    st.stop()

st.title("Heart Disease Prediction Model")

# Sidebar
st.sidebar.header("Model Performance")
st.sidebar.metric("Test Accuracy", f"{perf['test_accuracy']:.1%}")
if perf.get("test_auc"):
    st.sidebar.metric("Test ROC-AUC", f"{perf['test_auc']:.3f}")

st.sidebar.header("Top 5 Important Features")
for name, score in perf["top_features"]:
    st.sidebar.write(f"**{name}** → `{score:.4f}`")

# Input form
st.header("Enter Patient Data")
cols = st.columns(3)
inputs = []
for i, feat in enumerate(feature_names):
    with cols[i % 3]:
        val = st.number_input(feat.replace("_", " ").title(), value=0.0, step=0.01, format="%.4f")
        inputs.append(val)

if st.button("Predict Risk", type="primary"):
    with st.spinner("Analyzing..."):
        try:
            response = requests.post(API_URL, json={"features": inputs}, timeout=10)
            if response.status_code == 200:
                result = response.json()
                pred = result["prediction"]
                prob = result.get("predicted_probability", 0.5)
                
                if pred == 1:
                    st.error(f"High Risk of Heart Disease")
                    st.metric("Risk Probability", f"{prob:.1%}", delta=f"+{prob-0.5:.1%}")
                else:
                    st.success(f"Low Risk")
                    st.metric("Risk Probability", f"{prob:.1%}")
            else:
                st.error(f"API Error: {response.text}")
        except:
            st.warning("Local testing mode – no backend running")
            st.info("This works perfectly when deployed!")
