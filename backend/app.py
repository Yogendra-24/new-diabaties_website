from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib, os, numpy as np, pandas as pd
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Diabetes Predictor")

# Allow CORS for all origins (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "model_pipeline_full.joblib"
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Place model_pipeline_full.joblib in backend/")

bundle = joblib.load(MODEL_PATH)
# Expect bundle to contain 'model' and 'preprocessor'
if "model" in bundle:
    model = bundle["model"]
else:
    raise RuntimeError("Bundle must contain 'model' key.")

if "preprocessor" not in bundle:
    raise RuntimeError("Bundle must contain 'preprocessor' key with preprocessing artifacts.")

pre = bundle["preprocessor"]

# Preprocessor pieces (names may vary slightly depending on training script)
raw_columns = pre.get("raw_columns", [])
onehot_columns = pre.get("onehot_columns", [])
top20 = pre.get("top20", [])
poly = pre.get("poly_obj", pre.get("poly", None))
selector_columns = pre.get("selector_columns", [])
selector_support = np.array(pre.get("selector_support", pre.get("selector_support_mask", [])))
final_features = pre.get("final_features", [])
scaler = pre.get("scaler", None)
raw_defaults = pre.get("raw_defaults", {})

if poly is None or scaler is None or len(final_features) == 0:
    raise RuntimeError("Preprocessor bundle missing required pieces (poly, scaler, final_features).")

# Input model: 6 fields
class Input6(BaseModel):
    Age: float
    BMI: float
    GenHlth: int
    HighBP: int
    HighChol: int
    Smoker: int
    extra: Optional[dict] = None

@app.get("/features")
def features():
    return {"expected_inputs": ["Age","BMI","GenHlth","HighBP","HighChol","Smoker"]}

def build_feature_row_from_input(inp: Input6):
    # start with defaults for raw columns
    raw = {col: raw_defaults.get(col, 0) for col in raw_columns}
    # override with provided values
    raw["Age"] = inp.Age
    raw["BMI"] = inp.BMI
    raw["GenHlth"] = inp.GenHlth
    raw["HighBP"] = inp.HighBP
    raw["HighChol"] = inp.HighChol
    raw["Smoker"] = inp.Smoker
    if inp.extra:
        for k,v in inp.extra.items():
            if k in raw:
                raw[k] = v
    return raw

@app.post("/predict_json")
def predict_json(inp: Input6):
    # Build raw row
    raw = build_feature_row_from_input(inp)
    # DataFrame single row
    df_raw = pd.DataFrame([raw], columns=raw_columns)
    # One-hot encode to match training encoded columns
    X_encoded = pd.get_dummies(df_raw, drop_first=True)
    # Ensure all training onehot columns present
    for c in onehot_columns:
        if c not in X_encoded.columns:
            X_encoded[c] = 0
    # Reorder columns to match training order
    X_encoded = X_encoded[onehot_columns]
    # Build top20 features matrix (these columns were chosen from encoded space)
    # If any top20 column missing, fill with 0
    X_top20 = X_encoded.reindex(columns=top20, fill_value=0)
    # Apply polynomial transform (poly was fitted during training and saved)
    X_poly = poly.transform(X_top20.values)
    # Create DataFrame with selector_columns
    X_poly_df = pd.DataFrame(X_poly, columns=selector_columns)
    # Apply selector_support mask to pick final features
    # selector_support is boolean mask corresponding to selector_columns
    if len(selector_support) != len(X_poly_df.columns):
        # fallback: try to select by matching final_features
        X_final = X_poly_df.reindex(columns=final_features, fill_value=0)
    else:
        X_final = X_poly_df.loc[:, selector_support]
        # ensure order matches final_features
        X_final = X_final.reindex(columns=final_features, fill_value=0)
    # Scale
    X_scaled = scaler.transform(X_final.values)
    # Predict
    pred = model.predict(X_scaled)[0]
    label_map = {0: "Non-Diabetic", 1: "Pre-Diabetic", 2: "Diabetic"}
    return {"prediction_code": int(pred), "prediction_label": label_map.get(int(pred), str(pred))}

@app.get("/health")
def health():
    return {"status": "ok"}