"""Repair missing/corrupt per-model pipelines using the exact train.py procedure.

Small models retrain in seconds (GB ~1 min); untouched files are never rewritten.
Run:  py repair_models.py
"""
import joblib
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, MODELS_DIR, TARGET, SEED, TEST_SIZE
from src.data_loader import load_data
from src.preprocessing import build_pipeline
from src.models import get_models

SAFE = {"Linear Regression": "linear", "Polynomial Regression": "poly",
        "Decision Tree": "tree", "Random Forest": "forest", "KNN": "knn",
        "Gradient Boosting": "gb"}


def healthy(path) -> bool:
    try:
        joblib.load(path)
        return True
    except Exception:
        return False


def main():
    df = load_data(DATA_PATH)
    X = df.drop(columns=[TARGET]); y = df[TARGET]
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    models = get_models()
    for name, safe in SAFE.items():
        p = MODELS_DIR / "pipelines" / f"{safe}.pkl"
        if p.exists() and healthy(p):
            print(f"OK, kept: {p.name}")
            continue
        print(f"Rebuilding: {p.name} ...", flush=True)
        reg, poly, _, _ = models[name]
        pipe = build_pipeline(reg, poly)
        pipe.fit(Xtr, ytr)
        joblib.dump(pipe, p)
        print(f"Rebuilt + verified: {p.name} ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
