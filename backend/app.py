import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import traceback

# ---------------------------------
# Load the trained bundle
# ---------------------------------
bundle = joblib.load("model_pipeline_full.joblib")
model = bundle["model"]
pre = bundle["preprocessor"]

raw_columns        = pre["raw_columns"]
onehot_columns     = pre["onehot_columns"]
top20              = pre["top20"]
poly               = pre["poly_obj"]
selector_columns   = pre["selector_columns"]
selector_support   = pre["selector_support"]
final_features     = pre["final_features"]
scaler             = pre["scaler"]
raw_defaults       = pre["raw_defaults"]

# ---------------------------------
# FastAPI app + CORS
# ---------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # production me specific domain daal dena
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------
# Input model
# ---------------------------------
class Input6(BaseModel):
    Age: float
    BMI: float
    GenHlth: float
    HighBP: float
    HighChol: float
    Smoker: float
    extra: dict = None   # optional raw extra columns

# ---------------------------------
# Helper to safely merge raw row
# ---------------------------------
def build_feature_row_from_input(inp: Input6):
    # start with defaults
    raw = {col: raw_defaults.get(col, 0) for col in raw_columns}

    # override mandatory fields
    raw["Age"] = inp.Age
    raw["BMI"] = inp.BMI
    raw["GenHlth"] = inp.GenHlth
    raw["HighBP"] = inp.HighBP
    raw["HighChol"] = inp.HighChol
    raw["Smoker"] = inp.Smoker

    # extra features (only scalar allowed)
    if inp.extra:
        for k, v in inp.extra.items():
            if k in raw:
                if isinstance(v, (int, float, str, bool)):
                    raw[k] = v
                else:
                    # ignore non-scalar extras
                    continue

    return raw

# ---------------------------------
# Routes
# ---------------------------------
@app.get("/")
def home():
    return {"message": "Diabetes Prediction API is running!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict_json")
def predict_json(inp: Input6):
    try:
        # Build raw feature row
        raw = build_feature_row_from_input(inp)
        df_raw = pd.DataFrame([raw], columns=raw_columns)

        # One-hot encode
        X_encoded = pd.get_dummies(df_raw, drop_first=True)

        # Add missing OHE columns
        for c in onehot_columns:
            if c not in X_encoded.columns:
                X_encoded[c] = 0

        # Correct order
        X_encoded = X_encoded[onehot_columns]

        # Select top20 OHE features
        X_top20 = X_encoded.reindex(columns=top20, fill_value=0)

        # Polynomial transform
        X_poly = poly.transform(X_top20.values)
        X_poly_df = pd.DataFrame(X_poly, columns=selector_columns)

        # Feature selection
        if len(selector_support) != len(X_poly_df.columns):
            X_final = X_poly_df.reindex(columns=final_features, fill_value=0)
        else:
            X_final = X_poly_df.loc[:, selector_support]
            X_final = X_final.reindex(columns=final_features, fill_value=0)

        # Scale
        X_scaled = scaler.transform(X_final.values)

        # Predict
        pred = model.predict(X_scaled)[0]
        label_map = {
            0: "Non-Diabetic",
            1: "Pre-Diabetic",
            2: "Diabetic"
        }

        return {
            "prediction_code": int(pred),
            "prediction_label": label_map.get(int(pred), str(pred))
        }

    except Exception as e:
        # Print detailed traceback to Render logs
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal error during prediction: {str(e)}"
        )
