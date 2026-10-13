"""Inference helper used by the dashboard (never retrains)."""
import json
import joblib
import pandas as pd
from .config import MODELS_DIR, ALL_FEATURES, HG_PER_TONNE

def load_best():
    pipe = joblib.load(MODELS_DIR / "best_pipeline.pkl")
    meta = json.loads((MODELS_DIR / "metadata.json").read_text())
    return pipe, meta

def predict_yield(inputs: dict):
    pipe, meta = load_best()
    row = {k: inputs[k] for k in ALL_FEATURES}  # KeyError on missing -> handled by caller
    df = pd.DataFrame([row])
    hg = float(pipe.predict(df)[0])
    return {"hg_per_ha": hg, "tonnes_per_ha": hg / HG_PER_TONNE, "model": meta["best_model"]}
