import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import traceback

# Load simple model (6-feature model)
bundle = joblib.load("model_simple.joblib")

model = bundle["model"]
scaler = bundle["scaler"]
FEATURES = bundle["features"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Input6(BaseModel):
    Age: float
    BMI: float
    GenHlth: float
    HighBP: float
    HighChol: float
    Smoker: float

@app.get("/")
def home():
    return {"message": "Simple Diabetes Prediction API Running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict_json")
def predict_json(inp: Input6):
    try:
        row = pd.DataFrame([{
            "Age": inp.Age,
            "BMI": inp.BMI,
            "GenHlth": inp.GenHlth,
            "HighBP": inp.HighBP,
            "HighChol": inp.HighChol,
            "Smoker": inp.Smoker
        }], columns=FEATURES)

        X_scaled = scaler.transform(row)

        pred = int(model.predict(X_scaled)[0])
        label_map = {0: "Non-Diabetic", 1: "Pre-Diabetic", 2: "Diabetic"}

        return {"prediction_code": pred, "prediction_label": label_map[pred]}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
