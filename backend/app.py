import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import traceback
import numpy as np

# ---------------------------
# Load the trained bundle
# ---------------------------
BUNDLE_PATH = "model_pipeline_full.joblib"

try:
    top_bundle = joblib.load(BUNDLE_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load {BUNDLE_PATH}: {e}")

# top_bundle should be a dict with keys like 'model' and 'preprocessor'
if not isinstance(top_bundle, dict):
    raise RuntimeError("Loaded bundle is not a dict; expected {'model':..., 'preprocessor':...}")

# extract raw entries
top_model_entry = top_bundle.get("model", None)
pre = top_bundle.get("preprocessor", {}) or {}

# If model entry itself is a dict (nested), unpack sensibly
model = None
fallback_scaler = None
fallback_features = None

if isinstance(top_model_entry, dict):
    # common shapes: {'model': <estimator>, 'scaler': <scaler>, 'features': [...]}
    model = top_model_entry.get("model", None)
    fallback_scaler = top_model_entry.get("scaler", None)
    # try different keys for feature names
    fallback_features = top_model_entry.get("features", None) or top_model_entry.get("final_features", None)
else:
    model = top_model_entry

# Now read preprocessor pieces (prefer preprocessor dict; fallback to nested model entry)
raw_columns      = pre.get("raw_columns", None)
onehot_columns   = pre.get("onehot_columns", None)
top20            = pre.get("top20", None) or pre.get("top_20_raw", None)
poly             = pre.get("poly_obj", None) or pre.get("poly", None)
selector_columns = pre.get("selector_columns", None) or pre.get("selector_mask_cols", None)
selector_support = pre.get("selector_support", None) or pre.get("selector_support_mask", None)
final_features   = pre.get("final_features", None)
scaler           = pre.get("scaler", None)
raw_defaults     = pre.get("raw_defaults", None)

# If some pieces are missing in preprocessor, try to populate from nested model entry
if scaler is None and fallback_scaler is not None:
    scaler = fallback_scaler

if final_features is None and fallback_features is not None:
    final_features = fallback_features

# Basic validation
missing = []
for name, val in [
    ("model", model),
    ("poly", poly),
    ("scaler", scaler),
    ("final_features", final_features)
]:
    if val is None:
        missing.append(name)

# raw_columns and onehot_columns are nice-to-have for full reconstruction; if missing we'll try to infer later
if missing:
    # don't crash immediately; keep going but warn (server logs)
    print("Warning: some preprocessor pieces missing:", missing)

# ---------------------------
# FastAPI app + CORS
# ---------------------------
app = FastAPI(title="Diabetes Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for production, lock this down to your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Input model
# ---------------------------
class Input6(BaseModel):
    Age: float
    BMI: float
    GenHlth: float
    HighBP: float
    HighChol: float
    Smoker: float
    extra: dict = None   # optional raw extra columns

# ---------------------------
# Helper to safely merge raw row
# ---------------------------
def build_feature_row_from_input(inp: Input6):
    # Build raw defaults if available, else create an empty placeholder for common known raw names
    if raw_defaults:
        raw = {col: raw_defaults.get(col, 0) for col in raw_columns} if raw_columns else dict(raw_defaults)
    else:
        # fallback: try to build a minimal raw dict if raw_columns missing
        if raw_columns:
            raw = {col: 0 for col in raw_columns}
        else:
            # create a minimal raw dict with the 6 fields and some reasonable keys
            raw = {
                "Age": 0,
                "BMI": 0,
                "GenHlth": 3,
                "HighBP": 0,
                "HighChol": 0,
                "Smoker": 0
            }

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
            if isinstance(v, (int, float, str, bool)):
                raw[k] = v
            else:
                # ignore non-scalar extras
                continue

    return raw

# ---------------------------
# Routes
# ---------------------------
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

        # If we have full raw_columns and onehot_columns, follow the original pipeline
        if raw_columns and onehot_columns and poly is not None and selector_columns and final_features and scaler is not None:
            df_raw = pd.DataFrame([raw], columns=raw_columns)
            X_encoded = pd.get_dummies(df_raw, drop_first=True)
            # ensure training onehot columns exist
            for c in onehot_columns:
                if c not in X_encoded.columns:
                    X_encoded[c] = 0
            X_encoded = X_encoded[onehot_columns]

            # top20 (these are columns chosen from encoded space)
            X_top20 = X_encoded.reindex(columns=top20, fill_value=0)

            # polynomial transform
            X_poly = poly.transform(X_top20.values)
            X_poly_df = pd.DataFrame(X_poly, columns=selector_columns)

            # feature selection (selector_support expected boolean mask)
            if selector_support is None or len(selector_support) != len(X_poly_df.columns):
                X_final = X_poly_df.reindex(columns=final_features, fill_value=0)
            else:
                X_final = X_poly_df.loc[:, selector_support]
                X_final = X_final.reindex(columns=final_features, fill_value=0)

            X_scaled = scaler.transform(X_final.values)

        else:
            # Fallback: if we don't have the full preprocessor saved, try a minimal path:
            # Use only the 6 inputs in a fixed order (Age,BMI,GenHlth,HighBP,HighChol,Smoker)
            # This requires the model was trained with a matching fallback — may be less accurate.
            # Build numpy row: order Age,BMI,GenHlth,HighBP,HighChol,Smoker
            fallback_row = np.array([[raw.get("Age",0), raw.get("BMI",0), raw.get("GenHlth",3),
                                      raw.get("HighBP",0), raw.get("HighChol",0), raw.get("Smoker",0)]])
            # If scaler available, scale; else pass raw
            if scaler is not None:
                X_scaled = scaler.transform(fallback_row)
            else:
                X_scaled = fallback_row

        # Ensure model is callable: if model variable is a dict (nested) try to extract
        global model
        if isinstance(model, dict):
            inner = model
            if "model" in inner:
                model = inner["model"]
            else:
                raise HTTPException(status_code=500, detail="Loaded model entry is a dict but contains no 'model' key.")

        # final check
        if not hasattr(model, "predict"):
            raise HTTPException(status_code=500, detail="Loaded model object is invalid (no predict method).")

        pred = model.predict(X_scaled)[0]
        label_map = {0: "Non-Diabetic", 1: "Pre-Diabetic", 2: "Diabetic"}

        return {"prediction_code": int(pred), "prediction_label": label_map.get(int(pred), str(pred))}

    except HTTPException:
        # re-raise to let FastAPI send proper response
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error during prediction: {str(e)}")
