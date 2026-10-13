"""Live AI layer: Google Gemini (free tier) with an honest offline fallback.

No key / no network / API error  ->  caller falls back to the rule-based farmer
guide and SAYS SO in the UI. Nothing is ever presented as live AI when it is not.
"""
from __future__ import annotations
from pathlib import Path
import streamlit as st

from .config import BASE

KEY_FILE = BASE / ".env"
MODEL_DEFAULT = "gemini-2.0-flash"


def _read_env_file() -> dict:
    out = {}
    try:
        for line in KEY_FILE.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    except Exception:
        pass
    return out


def get_key() -> str:
    if st.session_state.get("gemini_key"):
        return str(st.session_state["gemini_key"])
    try:
        if st.secrets.get("GEMINI_API_KEY"):
            return str(st.secrets["GEMINI_API_KEY"])
    except Exception:
        pass
    import os
    if os.environ.get("GEMINI_API_KEY"):
        return str(os.environ["GEMINI_API_KEY"])
    return _read_env_file().get("GEMINI_API_KEY", "")


def save_key(key: str):
    key = (key or "").strip()
    lines = []
    try:
        if KEY_FILE.exists():
            lines = [l for l in KEY_FILE.read_text(encoding="utf-8").splitlines()
                     if not l.strip().startswith("GEMINI_API_KEY")]
    except Exception:
        lines = []
    lines.append(f"GEMINI_API_KEY={key}")
    KEY_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    st.session_state["gemini_key"] = key


def build_context(df, meta: dict, extracted: dict, comp_rows: list) -> str:
    """Pure function: grounding facts the model must stick to (testable offline)."""
    sim = ""
    if extracted:
        bits = ", ".join(f"{k}={v}" for k, v in extracted.items() if k != "suggested")
        sim = f"User's described conditions: {bits}. "
    return (
        "You are AgriPredict's farm guide. Answer ONLY from these verified project facts; "
        "say 'I don't know' otherwise. Never invent fertilizer/pesticide doses — refuse those and "
        "refer to the local extension officer.\n"
        f"Dataset: FAO harvests 1990-2013, {len(df):,} rows, {df['area'].nunique()} countries, "
        f"{df['item'].nunique()} crops ({', '.join(sorted(df['item'].unique()))}). "
        f"Columns: country, crop, year, rainfall mm/year, pesticides tonnes (national totals, NOT field doses), "
        f"temperature C. Target: yield hg/ha (divide by 10000 for tonnes/ha).\n"
        "Missing (never collected, never guess): N/P/K, soil pH, humidity, soil type, season, area, fertilizer.\n"
        f"Model scores: {comp_rows}. Best = highest test R². Shipped predictor keeps default settings.\n"
        f"{sim}Keep answers short, farmer-friendly, with units."
    )


def ask_live(question: str, history: list, context: str, model: str = MODEL_DEFAULT) -> str:
    """One live call. Raises on any failure — the caller must fall back openly."""
    import google.generativeai as genai
    key = get_key()
    if not key:
        raise RuntimeError("no API key saved")
    genai.configure(api_key=key)
    chat_hist = []
    for m in (history or [])[-6:]:
        chat_hist.append({"role": "user" if m.get("role") == "user" else "model",
                          "parts": [str(m.get("text", ""))[:1500]]})
    model_o = genai.GenerativeModel(model, system_instruction=context,
                                    generation_config={"temperature": 0.4, "max_output_tokens": 600})
    resp = model_o.generate_content(chat_hist + [{"role": "user", "parts": [question]}],
                                    request_options={"timeout": 60})
    text = (getattr(resp, "text", "") or "").strip()
    if not text:
        raise RuntimeError("empty model reply")
    return text
