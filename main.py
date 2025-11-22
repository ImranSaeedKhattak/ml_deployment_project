# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Load model once when the app starts
model_path = "model.joblib"
model = joblib.load(model_path)

app = FastAPI(title="My ML Model API")

# Get feature names (assuming you saved them or can infer)
# You can also hardcode them if you know the order
try:
    import json
    with open("../feature_names.json") as f:
        FEATURE_NAMES = json.load(f)
except:
    FEATURE_NAMES = model.feature_names_in_ if hasattr(model, "feature_names_in_") else None

class PredictionRequest(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        X = np.array(request.features).reshape(1, -1)
        
        # Optional: check number of features
        if FEATURE_NAMES and len(request.features) != len(FEATURE_NAMES):
            return {"error": f"Expected {len(FEATURE_NAMES)} features, got {len(request.features)}"}
        
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0].tolist() if hasattr(model, "predict_proba") else None
        
        response = {
            "prediction": int(prediction) if prediction.dtype == np.int64 else float(prediction),
        }
        if probability:
            response["probabilities"] = {str(i): prob for i, prob in enumerate(probability)}
            response["predicted_probability"] = float(max(probability))
        
        return response
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    return {"message": "Model API is running!"}