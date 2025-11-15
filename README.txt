
Diabetes Website - FINAL deployable package
==========================================

Contents:
- backend/
  - app.py            -> FastAPI backend (loads model_pipeline_full.joblib)
  - model_pipeline_full.joblib  -> Your trained model bundle (must be present)
  - requirements.txt  -> pip requirements for Render
- frontend/
  - index.html        -> Simple UI for 6 inputs (Age, BMI, GenHlth, HighBP, HighChol, Smoker)

How to run locally:
-------------------
1. cd backend
2. python3 -m venv venv
3. source venv/bin/activate   (Windows: venv\Scripts\activate)
4. pip install -r requirements.txt
5. uvicorn app:app --reload --host 0.0.0.0 --port 8000
6. Open http://localhost:8000 (if serving frontend via same server, or open frontend/index.html and change fetch URL to http://localhost:8000/predict_json)

Deploy to Render (quick):
-------------------------
1. Push repository to GitHub with this folder structure.
2. Create a new Web Service on Render, connect to the repo.
3. Set Start Command: uvicorn app:app --host 0.0.0.0 --port $PORT
4. Ensure requirements.txt present (it is).
5. Deploy. Note: model_pipeline_full.joblib is included in backend/ in this package.

Notes:
- If model_pipeline_full.joblib is larger than GitHub's normal limits, use Git LFS for the file.
- The backend expects the preprocessor keys saved in the bundle: 'raw_columns', 'onehot_columns', 'top20', 'poly_obj', 'selector_columns', 'selector_support', 'final_features', 'scaler', 'raw_defaults'.
