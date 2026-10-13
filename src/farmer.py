"""Farmer-facing helper: extract growing conditions from plain English and answer
from the project's real data. Offline rule-based (no API key). Never invents advice:
fertilizer/pesticide doses are refused; crop suggestions come from observed records only.
"""
from __future__ import annotations
import re
from .assistant import respond as generic_respond

NUM = {"rain": r"(\d+(?:\.\d+)?)\s*mm",
       "temp": r"(-?\d+(?:\.\d+)?)\s*(?:°\s*c|celsius|degrees?)",
       "pest": r"(\d+(?:\.\d+)?)\s*(?:tonnes?|tons?)(?![a-z])",
       "year": r"\b((?:19|20)\d{2})\b"}


def extract_features(text: str, areas: list, crops: list) -> dict:
    t = (text or "").lower()
    out: dict = {}
    for a in sorted(areas, key=len, reverse=True):
        if a.lower() in t:
            out["area"] = a
            break
    for c in sorted(crops, key=len, reverse=True):
        if c.lower() in t:
            out["item"] = c
            break
    for k, pat in NUM.items():
        m = re.search(pat, t)
        if m:
            v = float(m.group(1))
            if k == "year" and not (1990 <= int(v) <= 2035):
                continue
            out[{"rain": "average_rain_fall_mm_per_year", "temp": "avg_temp",
                 "pest": "pesticides_tonnes", "year": "year"}[k]] = v
    return out


def _similar(df, rain=None, temp=None):
    f = df
    if rain is not None:
        f = f[(f["average_rain_fall_mm_per_year"] >= rain - 250) & (f["average_rain_fall_mm_per_year"] <= rain + 250)]
    if temp is not None:
        f = f[(f["avg_temp"] >= temp - 3) & (f["avg_temp"] <= temp + 3)]
    return f


def answer(text: str, df, meta: dict) -> tuple:
    """Returns (intent, reply, extracted_dict)."""
    t = (text or "").lower().strip()
    ext = extract_features(text, meta.get("areas", []), meta.get("items", []))
    if not t:
        return ("greeting", "Tell me your crop, rainfall, temperature — e.g. “wheat with 1200mm rain at 24 degrees”.", ext)

    def hit(*ws):
        return any(w in t for w in ws)

    if hit("hello", "vanakkam", "help", "who are you", "what can you"):
        return ("greeting", ("I'm your offline farm guide. Describe your field — crop, rainfall in mm, temperature — "
                             "and I'll find similar past harvests, or tap a suggestion into the predictor. "
                             "I never invent fertilizer or pesticide doses."), ext)
    if t == "hi":
        return ("greeting", ("Hello! Tell me your crop, rainfall in mm and temperature — "
                             "e.g. “wheat with 1200mm rain at 24 degrees”."), ext)
    if hit("fertiliz", "how much pesticide", "spray", "dose", "how much urea", "pesticide advice"):
        return ("advice_refusal", ("I can't prescribe fertilizer or pesticide doses — this project has no fertilizer "
                                    "data and only country-level pesticide totals, so any dose would be made up. "
                                    "Please ask your local agriculture extension officer for doses; I can still estimate "
                                    "yields from weather and crop."), ext)
    if hit("which crop", "what crop", "what should i grow", "suggest", "suitable", "grow in"):
        rain = ext.get("average_rain_fall_mm_per_year")
        temp = ext.get("avg_temp")
        f = _similar(df, rain, temp)
        scope = "similar rainfall/temperature" if (rain is not None or temp is not None) else "the whole dataset"
        if len(f) < 20:
            f = df
            scope = "the whole dataset (too few similar records)"
        top = f.groupby("item")["value_hg_ha"].mean().sort_values(ascending=False).head(3)
        names = ", ".join(f"{c} (~{v:,.0f} hg/ha)" for c, v in top.items())
        return ("crop_suggest", (f"From {len(f):,} records with {scope}, the highest historical means are: {names}. "
                                 "That is observed history, not a guarantee — soils and seasons still decide."),
                {**ext, "suggested": list(top.index)})
    _have37 = [k for k in ("item", "average_rain_fall_mm_per_year", "avg_temp") if k in ext]
    if hit("rain", "temperature", "weather", "climate") and len(_have37) < 2:
        return ("weather_help", ("This model reads annual rainfall (mm/year) and mean temperature (°C). "
                                 "Give me numbers — e.g. “1400mm rain, 26 degrees” — and I'll find matching harvests "
                                 "or feed them to the predictor."), ext)
    if hit("best model", "accurate", "r2", "rmse", "mae", "metric", "tuned", "hyperparameter"):
        b = meta.get("best_model", "Random Forest")
        return ("metrics", (f"Best is {b} (test R² 0.984). MAE ≈ 4,074 hg/ha average miss; RMSE ≈ 10,920 punishes big "
                            "misses harder. Hyperparameters were tuned with GridSearchCV — see the Model Lab tuning table."), ext)
    if hit("predict", "yield", "estimate", "forecast"):
        have = [k for k in ("item", "average_rain_fall_mm_per_year", "avg_temp") if k in ext]
        if len(have) >= 2:
            return ("predict_ready", ("I caught enough to predict — tap the button below to carry these into the "
                                      "Yield Prediction form (you can adjust anything there)."), ext)
        return ("predict_help", ("To predict I need at least a crop plus rainfall and temperature — e.g. "
                                 "“maize, 1200mm rain, 24 degrees”. Add numbers and try again."), ext)
    if hit("fertilizer data", "npk", "nitrogen", "soil ph", "humidity", "season data"):
        return ("limits", ("Honestly missing here: nitrogen, phosphorus, potassium, soil pH, humidity, soil type, "
                           "season and fertilizer use were never collected in this FAO dataset — I work with country, "
                           "crop, year, rainfall, pesticides and temperature only."), ext)
    g = generic_respond(text, meta)
    if isinstance(g, dict):
        intent, reply = g.get("intent", "fallback"), g.get("reply") or "Ask me about crops, rainfall, models or predictions."
    else:
        intent, reply = list(g)[:2]
    return (intent, reply, ext)
