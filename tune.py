"""
Hyperparameter tuning (previously NOT done — all six models used fixed settings).
GridSearchCV with small honest grids (kept light so a college laptop finishes).
Run:  py tune.py
Out:  models/tuning.json (best params, CV R², held-out test scores, vs-default verdict)
"""
import json
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.config import DATA_PATH, MODELS_DIR, TARGET, SEED, TEST_SIZE
from src.data_loader import load_data
from src.preprocessing import build_pipeline
from src.evaluate import regression_metrics

GRIDS = {
    "Decision Tree": (DecisionTreeRegressor(random_state=SEED), False,
                      {"reg__max_depth": [None, 8, 12], "reg__min_samples_split": [2, 5]}, 3, 1),
    "KNN": (KNeighborsRegressor(), False,
            {"reg__n_neighbors": [3, 5, 9, 15, 25]}, 3, 1),
    "Random Forest": (RandomForestRegressor(random_state=SEED, n_jobs=-1), False,
                      {"reg__n_estimators": [50, 100], "reg__max_depth": [None, 15]}, 2, 1),
    "Gradient Boosting": (GradientBoostingRegressor(random_state=SEED), False,
                          {"reg__n_estimators": [50, 100], "reg__learning_rate": [0.05, 0.1]}, 2, 1),
}


def main():
    df = load_data(DATA_PATH)
    X = df.drop(columns=[TARGET]); y = df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    base = pd.read_csv(MODELS_DIR / "metrics.csv").set_index("Model")["R2_test"].to_dict()
    out = {}
    for name, (reg, poly, grid, cv, jobs) in GRIDS.items():
        pipe = build_pipeline(reg, poly)
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="r2", n_jobs=jobs, refit=True)
        gs.fit(Xtr, ytr)  # preprocessing inside pipeline fits on train folds only
        m = regression_metrics(yte, gs.predict(Xte))
        out[name] = {"best_params": {k.replace("reg__", ""): v for k, v in gs.best_params_.items()},
                     "cv_R2": float(gs.best_score_), "test_R2": m["R2"], "test_RMSE": m["RMSE"],
                     "test_MAE": m["MAE"], "default_R2": float(base.get(name, float("nan"))),
                     "beats_default": bool(m["R2"] > base.get(name, -9))}
        print(f"{name}: best={out[name]['best_params']} CV={m['R2']:.4f}... "
              f"test_R2={m['R2']:.4f} (default {base.get(name, float('nan')):.4f})", flush=True)
    out["_note"] = ("Linear/Polynomial Regression have no meaningful hyperparameters to tune "
                    "(solver settings only) — honestly skipped. Shipped predictor keeps default "
                    "settings; tuned scores are reported for comparison.")
    (MODELS_DIR / "tuning.json").write_text(json.dumps(out, indent=2))
    print("Saved models/tuning.json")


if __name__ == "__main__":
    main()
