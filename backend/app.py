import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import traceback
import numpy as np

# ------------------------------------------------------
# LOAD MODEL BUNDLE (YOUR TRAINED FILE)
# ------------------------------------------------------
bundle = joblib.load("model_pipeline_full.joblib")

# model stored as nested dict → extract real model
model_info = bundle["model"]          # this is a dict
model = model_info["model"]           # REAL model object
scaler = model_info["scaler"]         # REAL scaler object
feature_names = model_info["features"]  # list of feature names used during training

# For your model features are EXACTLY these 6:
INPUT_COLS = ["Age", "BMI", "GenHlth", "HighBP", "HighChol", "Smoker"]

# ------------------------------------------------------
# FASTAPI + CORS
# ------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------
# INPUT MODEL
# ------------------------------------------------------
class Input6(BaseModel):
    Age: float
    BMI: float
    GenHlth: float
    HighBP: float
    HighChol: float
    Smoker: float

# ------------------------------------------------------
# ROUTES
# ------------------------------------------------------
@app.get("/")
def home():
    return {"message": "Diabetes Prediction API running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict_json")
def predict_json(inp: Input6):
    try:
        # Convert to dataframe row
        row = pd.DataFrame([{
            "Age": inp.Age,
            "BMI": inp.BMI,
            "GenHlth": inp.GenHlth,
            "HighBP": inp.HighBP,
            "HighChol": inp.HighChol,
            "Smoker": inp.Smoker
        }], columns=INPUT_COLS)

        # Scale
        X_scaled = scaler.transform(row.values)

        # Predict
        pred = int(model.predict(X_scaled)[0])
        label_map = {0: "Non-Diabetic", 1: "Pre-Diabetic", 2: "Diabetic"}

        return {
            "prediction_code": pred,
            "prediction_label": label_map[pred]
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
