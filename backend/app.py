import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import traceback
import os

# -----------------------------
# LOAD SIMPLE MODEL
# -----------------------------
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

# -----------------------------
# SERVE FRONTEND STATIC FILES
# -----------------------------
frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(frontend_path, "index.html"))

# -----------------------------
# INPUT MODEL
# -----------------------------
class Input6(BaseModel):
    Age: float
    BMI: float
    GenHlth: float
    HighBP: float
    HighChol: float
    Smoker: float

# -----------------------------
# PREDICT ROUTE
# -----------------------------
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
        label = {0: "Non-Diabetic", 1: "Pre-Diabetic", 2: "Diabetic"}[pred]

        return {"prediction_code": pred, "prediction_label": label}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
