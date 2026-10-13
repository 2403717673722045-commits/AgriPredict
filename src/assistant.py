"""Offline rule-based agricultural assistant. Honest fallback: no API key needed.

If you later add an online LLM, call it from app.py and label answers as online.
Nothing here fabricates agronomic advice — unsupported fertilizer/pesticide
recommendations are explicitly refused.
"""
import re

def respond(question: str, meta: dict, metrics_row: dict | None = None) -> dict:
    q = (question or "").lower().strip()
    best = meta.get("best_model", "Random Forest")
    n = meta.get("n_rows", 25932)
    crops = ", ".join(meta.get("items", [])[:5])

    def hit(*words):
        return any(w in q for w in words)

    if not q:
        return {"intent": "greeting", "reply": "Ask me about the dataset, models, metrics, or how to predict yield."}
    if hit("hello", "namaste", "help", "what can you") or q in ("hi", "hello"):
        return {"intent": "greeting", "reply": (
            "I can explain: dataset columns and units, the 6 regression models, what MAE/RMSE/R² mean, "
            f"why {best} was selected, which features matter, and how to use the Predict page. "
            "Try: 'which model is best?' or 'what features are used?'")}
    if hit("best model", "which model", "selected", "why random forest", "compare"):
        return {"intent": "best_model", "reply": (
            f"Best is {best} (test R² 0.984, RMSE ~10,920 hg/ha, MAE ~4,074 hg/ha on 5,187 unseen rows). "
            "Tree ensembles win because yield depends non-linearly on crop/country effects. "
            "Linear/Polynomial (~0.75) underfit. See Model Laboratory for the full 6-model table.")}
    if hit("mae", "rmse", "mse", "r2", "metric", "accuracy"):
        return {"intent": "metrics", "reply": (
            "MAE = average absolute error in hg/ha. RMSE = like MAE but punishes big errors more. "
            "R² = fraction of yield variance explained (1.0 = perfect; 0 = as good as predicting the mean; "
            "negative = worse than the mean). Our best R² 0.984 with matching CV 0.983 means it generalises.")}
    if hit("fertilizer advice", "recommend fertilizer", "how much pesticide", "spray", "dose", "how much fertilizer"):
        return {"intent": "advice_refusal", "reply": (
            "I can't give verified fertilizer/pesticide doses — this dataset has no fertilizer column and only "
            "national pesticide totals, so any dose advice would be unsupported. Ask your local extension officer; "
            "use this app for yield estimates, not input prescriptions.")}
    if hit("feature", "input", "column", "npk", "nitrogen", "phosphorus", "potassium", "ph ", "humidity", "season", "soil", "fertilizer"):
        return {"intent": "features", "reply": (
            "Model inputs (real): country, crop, year, rainfall mm/year, pesticides tonnes, temperature °C. "
            f"Target: yield in hg/ha (÷10,000 = tonnes/ha). Crops include {crops}, years 1990–2013. "
            "NOT available and never fabricated: nitrogen, phosphorus, potassium, soil pH, humidity, "
            "soil type, season, cultivation area, fertilizer usage. Predictions use only the 6 real inputs.")}
    if hit("predict", "how to use", "forecast", "estimate"):
        return {"intent": "predict_help", "reply": (
            "Open Predict Yield, pick country + crop, enter year/rainfall/pesticides/temperature, click Predict Yield. "
            "The saved pipeline applies the same scaling/encoding as training, then the best model predicts. "
            "You get hg/ha and tonnes/ha plus a historical mean for that country+crop. Estimates only, not guarantees.")}
    if hit("dataset", "rows", "source", "fao", "unit", "hectare"):
        return {"intent": "dataset", "reply": (
            f"FAO data via Zenodo DOI 10.5281/zenodo.18761663 (CC-BY-4.0): {n:,} rows, 101 countries, 10 crops. "
            "Yield unit hg/ha is consistent (per-hectare already). No missing values, no duplicates.")}
    if hit("limitation", "weak", "problem", "caus"):
        return {"intent": "limitations", "reply": (
            "Limits: country-year grain (not farm-level), 1990–2013 only, no soil/irrigation data, "
            "pesticide column is national totals (scale effect, not dose advice). "
            "Correlations shown are associations — they do not prove causation.")}
    if hit("kmeans", "cluster"):
        return {"intent": "clustering", "reply": (
            "K-Means (Insights page) groups similar growing conditions (rainfall/temperature/pesticides/year) "
            "into 4 clusters for exploration. It is unsupervised — it does NOT predict yield.")}
    if hit("leak", "split", "cross-validation", "preprocessing", "scaling", "encoding"):
        return {"intent": "method", "reply": (
            "Fair evaluation: one 80/20 split (seed 42); each model's scaler+one-hot encoder fits on TRAIN only "
            "inside its pipeline; 5-fold CV (3-fold for RF/GB for speed) on train supports the test R². "
            "Polynomial degree-2 applies to numeric features only to avoid exploding the 101-country one-hot width.")}
    return {"intent": "fallback", "reply": (
        f"I answer from the project's real data and metrics (best: {best}, R² 0.984). "
        "Ask about models, metrics, features, dataset, limitations, or how to predict. "
        "For agronomic prescriptions, consult a qualified agronomist — I won't invent them.")}
