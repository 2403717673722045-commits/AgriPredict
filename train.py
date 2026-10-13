"""
Train all six regressors fairly (common split, preprocessing fit on train only).
Run:  py train.py
Out:  models/best_pipeline.pkl, models/pipelines/*.pkl, models/metadata.json,
      models/metrics.csv, models/test_predictions.csv, models/degree_sweep.json,
      outputs/figures/*.png
"""
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.config import (DATA_PATH, MODELS_DIR, FIG_DIR, TARGET, SEED, TEST_SIZE,
                        NUM_FEATURES, CAT_FEATURES, DATASET_SOURCE, UNAVAILABLE)
from src.data_loader import load_data
from src.preprocessing import build_pipeline, build_preprocessor
from src.models import get_models
from src.evaluate import regression_metrics
from src.eda import generate_eda_figures

SAFE_NAME = {"Linear Regression": "linear", "Polynomial Regression": "poly",
             "Decision Tree": "tree", "Random Forest": "forest", "KNN": "knn",
             "Gradient Boosting": "gb"}

def avp_fig(y_true, y_pred, title, path):
    plt.figure(figsize=(6, 4.5))
    plt.scatter(y_true, y_pred, alpha=0.25, s=8)
    mn, mx = float(np.min(y_true)), float(np.max(y_true))
    plt.plot([mn, mx], [mn, mx], "r--", lw=1.5)
    plt.title(title); plt.xlabel("Actual (hg/ha)"); plt.ylabel("Predicted (hg/ha)")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()

def main(data_path=DATA_PATH, models_dir=MODELS_DIR, fig_dir=FIG_DIR):
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "pipelines").mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(data_path)
    print(f"Rows: {df.shape[0]}, Cols: {df.shape[1]} | Target: {TARGET}")
    print(f"Missing: {int(df.isnull().sum().sum())} | Duplicates: {int(df.duplicated().sum())}")

    print("EDA figures...")
    print(f"EDA figures: {len(generate_eda_figures(df, fig_dir))}")

    X = df.drop(columns=[TARGET]); y = df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    print(f"Train {Xtr.shape} | Test {Xte.shape} (seed {SEED})")

    rows, fitted, test_preds = [], {}, {"y_true": np.asarray(yte)}
    for name, (reg, use_poly, cv, _note) in get_models().items():
        pipe = build_pipeline(reg, use_poly)
        pipe.fit(Xtr, ytr)  # preprocessor fits on TRAIN only -> no leakage
        pred = pipe.predict(Xte)
        m = regression_metrics(yte, pred)
        cv_r2 = float(np.mean(cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="r2")))
        rows.append({"Model": name, "MAE": m["MAE"], "MSE": m["MSE"],
                     "RMSE": m["RMSE"], "R2_test": m["R2"], "R2_cv_mean": cv_r2,
                     "cv_folds": cv, "poly": use_poly})
        fitted[name] = (pipe, pred)
        test_preds[SAFE_NAME[name]] = np.asarray(pred)
        joblib.dump(pipe, models_dir / "pipelines" / f"{SAFE_NAME[name]}.pkl")
        avp_fig(yte, pred, f"Actual vs predicted — {name}", fig_dir / f"avp_{SAFE_NAME[name]}.png")
        print(f"{name}: R2_test={m['R2']:.4f} R2_cv={cv_r2:.4f} RMSE={m['RMSE']:.1f} MAE={m['MAE']:.1f}")

    comp = pd.DataFrame(rows).sort_values("R2_test", ascending=False)
    comp.to_csv(models_dir / "metrics.csv", index=False)
    pd.DataFrame(test_preds).to_csv(models_dir / "test_predictions.csv", index=False)
    print("\n" + comp.to_string(index=False))

    best_name = comp.iloc[0]["Model"]
    best_pipe, best_pred = fitted[best_name]
    joblib.dump(best_pipe, models_dir / "best_pipeline.pkl")

    meta = {
        "best_model": best_name, "target": TARGET, "target_unit": "hg/ha (divide by 10000 for tonnes/ha)",
        "numeric_features": NUM_FEATURES, "categorical_features": CAT_FEATURES,
        "areas": sorted(df["area"].unique().tolist()), "items": sorted(df["item"].unique().tolist()),
        "year_min": int(df["year"].min()), "year_max": int(df["year"].max()),
        "n_rows": int(df.shape[0]), "seed": SEED, "test_size": TEST_SIZE,
        "dataset_source": DATASET_SOURCE, "unavailable_features": UNAVAILABLE,
        "safe_names": SAFE_NAME,
        "comparison": comp.to_dict(orient="records"),
        "model_notes": {k: v[3] for k, v in get_models().items()},
    }
    (models_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Best: {best_name} -> models/best_pipeline.pkl + models/pipelines/ (6)")

    # --- shared figures ---
    plt.figure(figsize=(9, 5))
    c = comp.sort_values("R2_test")
    plt.barh(c["Model"], c["R2_test"]); plt.title("Model comparison — R² on test set (higher = better)")
    plt.xlabel("R²"); plt.tight_layout(); plt.savefig(fig_dir / "10_model_r2.png"); plt.close()

    avp_fig(yte, best_pred, f"Actual vs predicted — {best_name}", fig_dir / "11_actual_vs_pred.png")

    plt.figure(figsize=(8, 4))
    plt.scatter(best_pred, np.asarray(yte) - np.asarray(best_pred), alpha=0.3, s=8)
    plt.axhline(0, color="r", ls="--"); plt.title(f"Residuals — {best_name} (spread around 0 = good)")
    plt.xlabel("Predicted (hg/ha)"); plt.ylabel("Residual"); plt.tight_layout()
    plt.savefig(fig_dir / "12_residuals.png"); plt.close()

    reg = best_pipe.named_steps["reg"]
    if hasattr(reg, "feature_importances_"):
        ohe = best_pipe.named_steps["pre"].named_transformers_["cat"]
        names = NUM_FEATURES + ohe.get_feature_names_out(CAT_FEATURES).tolist()
        imp = pd.Series(reg.feature_importances_, index=names).sort_values(ascending=False).head(20)
        plt.figure(figsize=(8, 6)); imp.sort_values().plot(kind="barh")
        plt.title(f"Top 20 importances — {best_name}"); plt.tight_layout()
        plt.savefig(fig_dir / "13_feature_importance.png"); plt.close()
        imp.to_csv(models_dir / "feature_importance.csv")
        print("Top importances:\n", imp.head(10).to_string())

    # --- ML Studio extras ---
    # 1. Readable display tree (depth 3, clearly labelled simplified version)
    dt_show = DecisionTreeRegressor(max_depth=3, random_state=SEED)
    dt_show.fit(build_preprocessor(False).fit_transform(Xtr), ytr)
    plt.figure(figsize=(16, 8))
    plot_tree(dt_show, filled=True, rounded=True, max_depth=3, fontsize=7)
    plt.title("Decision Tree — simplified display version (max_depth=3). Real model is unrestricted.")
    plt.tight_layout(); plt.savefig(fig_dir / "14_tree_display.png", dpi=110); plt.close()

    # 2. Polynomial degree sweep (numeric-only expansion, cheap honest fits)
    sweep = {}
    for d in (2, 3, 4):
        pre = ColumnTransformer([
            ("num", Pipeline([("sc", StandardScaler()),
                              ("pfx", PolynomialFeatures(degree=d, include_bias=False))]), NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ])
        p = Pipeline([("pre", pre), ("reg", LinearRegression())])
        p.fit(Xtr, ytr)
        r2 = float(p.score(Xte, yte))
        sweep[str(d)] = {"R2_test": r2, "n_poly_terms": int(14 if d == 2 else (34 if d == 3 else 69))}
        print(f"Poly degree {d}: R2_test={r2:.4f}")
    (models_dir / "degree_sweep.json").write_text(json.dumps(sweep, indent=2))
    print("Studio extras saved: per-model AvP, tree display, degree sweep.")

    # 3. Temporal generalization check: train on 1990-2005, test on UNSEEN years 2006-2013
    past = df[df["year"] <= 2005]; future = df[df["year"] > 2005]
    Xpa, ypa = past.drop(columns=[TARGET]), past[TARGET]
    Xfu, yfu = future.drop(columns=[TARGET]), future[TARGET]
    temporal = {}
    for tname, treg in [("Random Forest", RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)),
                        ("Linear Regression", LinearRegression())]:
        tp = build_pipeline(treg, use_poly=False)
        tp.fit(Xpa, ypa)
        temporal[tname] = {"train_years": "1990-2005", "test_years": "2006-2013 (unseen)",
                           "n_test": int(len(yfu)), "R2_unseen_years": float(tp.score(Xfu, yfu))}
        print(f"Temporal {tname}: R2={temporal[tname]['R2_unseen_years']:.4f}")
    meta["temporal_check"] = temporal
    (models_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("Temporal check saved.")

if __name__ == "__main__":
    main()
