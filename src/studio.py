"""ML Studio helpers: lazy per-model loading + honest on-demand retraining.

Retraining uses the IDENTICAL 80/20 split (seed 42) as train.py, so results are
comparable. Every retrain function fits preprocessing on TRAIN only.
Results are cached per parameter value, so revisiting a setting is instant.
"""
import joblib
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

from .config import DATA_PATH, MODELS_DIR, TARGET, SEED, TEST_SIZE, NUM_FEATURES
from .data_loader import load_data
from .preprocessing import build_pipeline
from .evaluate import regression_metrics


def get_split():
    df = load_data(DATA_PATH)
    X = df.drop(columns=[TARGET]); y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)


@st.cache_resource(show_spinner="Loading trained model...")
def load_pipeline(name: str):
    safe = {"Linear Regression": "linear", "Polynomial Regression": "poly",
            "Decision Tree": "tree", "Random Forest": "forest", "KNN": "knn",
            "Gradient Boosting": "gb"}[name]
    return joblib.load(MODELS_DIR / "pipelines" / f"{safe}.pkl")


@st.cache_resource(show_spinner="Retraining Polynomial model (degree changed)...")
def retrain_poly(degree: int):
    Xtr, Xte, ytr, yte = get_split()
    pipe = build_pipeline(LinearRegression(), use_poly=True if degree == 2 else False)
    if degree != 2:  # rebuild numeric expansion with the requested degree
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from .config import NUM_FEATURES, CAT_FEATURES
        pre = ColumnTransformer([
            ("num", Pipeline([("sc", StandardScaler()),
                              ("pfx", PolynomialFeatures(degree=degree, include_bias=False))]), NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ])
        pipe = Pipeline([("pre", pre), ("reg", LinearRegression())])
    pipe.fit(Xtr, ytr)
    m = regression_metrics(yte, pipe.predict(Xte))
    return {"degree": degree, "metrics": m}


@st.cache_resource(show_spinner="Retraining KNN (neighbours changed)...")
def retrain_knn(k: int):
    Xtr, Xte, ytr, yte = get_split()
    pipe = build_pipeline(KNeighborsRegressor(n_neighbors=k), use_poly=False)
    pipe.fit(Xtr, ytr)
    return {"k": k, "metrics": regression_metrics(yte, pipe.predict(Xte))}


@st.cache_resource(show_spinner="Retraining Decision Tree (depth changed)...")
def retrain_tree(depth: int):
    Xtr, Xte, ytr, yte = get_split()
    pipe = build_pipeline(DecisionTreeRegressor(max_depth=depth, random_state=SEED), use_poly=False)
    pipe.fit(Xtr, ytr)
    return {"depth": depth, "metrics": regression_metrics(yte, pipe.predict(Xte))}


@st.cache_resource(show_spinner="Training Random Forest with new tree count (takes ~30s)...")
def retrain_forest(n_trees: int):
    Xtr, Xte, ytr, yte = get_split()
    pipe = build_pipeline(RandomForestRegressor(n_estimators=n_trees, random_state=SEED, n_jobs=-1),
                          use_poly=False)
    pipe.fit(Xtr, ytr)
    return {"n_estimators": n_trees, "metrics": regression_metrics(yte, pipe.predict(Xte))}


@st.cache_resource(show_spinner="Training Gradient Boosting with new settings (takes ~30-60s)...")
def retrain_gb(n_est: int, lr: float):
    Xtr, Xte, ytr, yte = get_split()
    pipe = build_pipeline(GradientBoostingRegressor(n_estimators=n_est, learning_rate=lr,
                                                    random_state=SEED), use_poly=False)
    pipe.fit(Xtr, ytr)
    return {"n_estimators": n_est, "lr": lr, "metrics": regression_metrics(yte, pipe.predict(Xte))}


def predict_with(pipe, inputs: dict) -> float:
    return float(pipe.predict(pd.DataFrame([inputs]))[0])


REQUIRED_COLUMNS = ["area", "year", "average_rain_fall_mm_per_year", "pesticides_tonnes",
                    "avg_temp", "item", "value_hg_ha"]


def validate_upload(df: pd.DataFrame) -> dict:
    """Check an uploaded CSV against the FAO schema. Never crashes on odd files."""
    errors, warnings, info = [], [], {}
    try:
        cols = [str(c).strip() for c in df.columns]
        df = df.copy()
        df.columns = cols
    except Exception as e:
        return {"ok": False, "errors": [f"Unreadable header: {e}"], "warnings": [], "info": {}}
    missing = [c for c in REQUIRED_COLUMNS if c not in cols]
    if missing:
        errors.append("Missing columns: " + ", ".join(missing))
    extra = [c for c in cols if c not in REQUIRED_COLUMNS]
    if extra:
        warnings.append("Ignored extra columns: " + ", ".join(extra))
    for c in ["year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp", "value_hg_ha"]:
        if c in df.columns:
            bad = pd.to_numeric(df[c], errors="coerce").isnull().sum()
            if bad > 0:
                (errors if bad / max(len(df), 1) > 0.05 else warnings).append(
                    f"{c}: {int(bad)} non-numeric values ({'rejected' if bad / max(len(df), 1) > 0.05 else 'tolerated'})")
    n = len(df)
    info = {"rows": n, "columns": len(cols),
            "missing_cells": int(df.isnull().sum().sum()) if n else 0,
            "duplicates": int(df.duplicated().sum()) if n else 0}
    if n < 50:
        errors.append(f"Only {n} rows — need ≥ 50 to train anything meaningful.")
    elif n < 1000:
        warnings.append(f"Small file ({n} rows): models will train but generalise worse than on 25,932 FAO rows.")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "info": info, "frame": df}


def transform_preview(pipe, df: pd.DataFrame, n: int = 5) -> dict:
    """Show what the fitted preprocessing actually produces (no refit, no leakage)."""
    pre = pipe.named_steps["pre"]
    sample = df.drop(columns=["value_hg_ha"], errors="ignore").head(n)
    arr = pre.transform(sample)
    try:
        names = list(pre.get_feature_names_out())
    except Exception:
        names = [f"f{i}" for i in range(arr.shape[1])]
    num_n = len(NUM_FEATURES)
    return {"in_rows": int(arr.shape[0]), "out_features": int(arr.shape[1]),
            "scaled_numeric": num_n, "one_hot": int(arr.shape[1]) - num_n,
            "feature_names_head": names[:12]}
