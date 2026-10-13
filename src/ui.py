"""Shared shell v4-cream: flat deterministic theme (no wrapper divs), navbar, footer, photos, data.

Only single closed-markdown blocks, plain widgets, st.image photos and Streamlit's own
bordered containers are used — every pattern screenshot-verified. No unclosed-div wrappers,
no CSS photo backgrounds.
"""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

APP_VERSION = "v4-cream"

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "crop_yield.csv"
META = BASE / "models" / "metadata.json"
FIG = BASE / "outputs" / "figures"
ASSETS = BASE / "assets"
TEAM_FILE = BASE / "team.json"
DEFAULT_TEAM = {"name": "", "team": "", "guide": "", "college": ""}
SAFE = {"Linear Regression": "linear", "Polynomial Regression": "poly", "Decision Tree": "tree",
        "Random Forest": "forest", "KNN": "knn", "Gradient Boosting": "gb"}
MODELS6 = list(SAFE.keys())
VIEWS = ["Home", "Prep", "Crops", "Predict", "Lab", "Insights", "Simulator", "AI", "About"]
VIEW_LABEL = {"Home": "Home", "Prep": "Data Prep", "Crops": "Crop Explorer", "Predict": "Yield Prediction",
              "Lab": "ML Model Lab", "Insights": "Agricultural Insights", "Simulator": "Scenario Simulator",
              "AI": "Farmer AI", "About": "About Project"}

CROP_IMG = {"Rice, paddy": "rice.jpg", "Wheat": "wheat.jpg", "Maize": "maize.jpg",
            "Potatoes": "crops/potato.jpg", "Soybeans": "crops/soybean.jpg", "Sorghum": "crops/sorghum.jpg",
            "Cassava": "crops/cassava.jpg", "Sweet potatoes": "crops/sweetpotato.jpg",
            "Plantains and others": "crops/plantain.jpg", "Yams": "produce.jpg"}

THEME = """<style>
:root { color-scheme:light; }
.stApp { background:#F3EFE2; color:#22301F; }
.stApp p, .stApp li, .stApp div[data-testid="stMarkdownContainer"] p { color:#22301F; }
section[data-testid="stSidebar"] { display:none; }
header[data-testid="stHeader"] { display:none; }
div.block-container { max-width:1400px; padding:14px 10px 0; }
.herofull { position:relative; border-radius:22px; overflow:hidden; background:#0E3B24;
  box-shadow:0 24px 60px rgba(14,59,36,.30); }
.herofull img { width:100%; height:560px; object-fit:cover; display:block; }
.herofull .shade { position:absolute; inset:0;
  background:linear-gradient(100deg,rgba(10,30,20,.88) 25%,rgba(10,30,20,.45) 60%,rgba(10,30,20,.10)); }
.herofull .copy { position:absolute; left:0; right:0; bottom:0; padding:42px 44px; }
.herofull .eyebrow { color:#E9C46A; font-weight:800; letter-spacing:3px; font-size:13px; }
.herofull h1 { color:#fff !important; font-size:clamp(34px,4.6vw,64px); line-height:1.04; margin:8px 0 10px;
  text-shadow:0 4px 26px rgba(0,0,0,.55); }
.herofull p.sub { color:#EDF3E2 !important; font-size:16.5px; max-width:62ch; text-shadow:0 2px 12px rgba(0,0,0,.55); }
.herofull .chips { margin-top:12px; }
.herofull .chip { display:inline-block; background:rgba(233,185,73,.16); border:1.5px solid #E9B949; color:#F5E9C8;
  border-radius:999px; padding:5px 14px; font-size:12.5px; font-weight:700; margin:0 8px 6px 0; }
img.banner { width:100%; height:360px; object-fit:cover; border-radius:18px; display:block;
  box-shadow:0 12px 28px rgba(20,50,30,.16); }
.capline { text-align:center; color:#5B6B54; font-size:13px; margin:6px 0 2px; }
div[data-testid="stVerticalBlockBorderWrapper"] { border:2px solid #2E7D4F !important; border-radius:18px;
  box-shadow:0 12px 30px rgba(20,50,30,.10); background:#FFFFFF; }
div[data-testid="stImage"] img { border-radius:18px; box-shadow:0 12px 28px rgba(20,50,30,.16); }
.strip-dark { background:linear-gradient(100deg,#0E3B24,#1B5E35); color:#EAF4EC; border-radius:18px;
  padding:20px 26px; box-shadow:0 14px 32px rgba(14,59,36,.25); }
.strip-dark h1, .strip-dark h2, .strip-dark h3 { color:#fff !important; margin:4px 0 8px; }
.strip-dark p { color:#D9E5D2 !important; }
.strip-dark .eyebrow { color:#E9C46A; font-style:italic; font-weight:700; }
.titlebar { background:#0E3B24; border-radius:14px; padding:12px 20px; margin:22px 0 12px;
  border-left:8px solid #E9B949; }
.titlebar h2, .titlebar h3 { color:#fff !important; margin:0; }
.titlebar p { color:#C9D8C2 !important; margin:2px 0 0; font-size:14px; }
.titlebar .eyebrow { color:#E9C46A; font-style:italic; font-weight:700; font-size:13px; }
.statband { background:linear-gradient(120deg,#0E3B24,#1B5E35); border-radius:18px; padding:22px 26px; }
.statband .n { font-size:34px; font-weight:800; color:#E9B949; }
.statband .l { color:#EAF4EC; font-size:12px; font-weight:700; letter-spacing:1px; }
.mixband { background:#EAF4EC; border:2px solid #BFD8BF; border-radius:18px; padding:22px 26px; }
.mixband p, .mixband li { color:#22301F !important; }
.eyebrow { color:#2E7D4F; font-style:italic; font-weight:700; font-size:15px; }
h1.site, h2.site, h3.site { color:#101E12 !important; font-weight:800; letter-spacing:-.5px; }
.stApp h1, .stApp h2, .stApp h3 { color:#101E12; }
.u-yellow { text-decoration:underline; text-decoration-color:#E9B949; text-decoration-thickness:5px;
  text-underline-offset:6px; }
.harvest { border-radius:16px; padding:18px 20px; background:linear-gradient(100deg,#EDF7ED,#FEF6D8);
  border:2px solid #2E7D4F; color:#0E3B24 !important; font-size:18px; }
.harvest small { color:#3F6212 !important; }
.soilnote { background:#F7F3E6; border-left:5px solid #8A6D3B; padding:10px 14px; border-radius:8px;
  color:#4A3A26 !important; }
.causes { background:#FFF8E1; border-left:5px solid #E9B949; padding:10px 14px; border-radius:8px;
  color:#6B5312 !important; }
.scorebox { background:linear-gradient(120deg,#0E3B24,#1B5E35); color:#fff; border-radius:18px;
  padding:18px 22px; text-align:center; border:2px solid #E9B949; }
.scorebox .n { font-size:44px; color:#E9B949; font-weight:800; }
.card { background:#fff; border:2px solid #DDE5CF; border-radius:16px; padding:14px 16px; }
.card b { color:#0E3B24; } .card small, .card p { color:#5B6B54; }
.algocard { background:#fff; border:1.5px solid #DDE5CF; border-radius:16px; padding:14px 16px; height:100%; }
.algocard b { color:#0E3B24; } .algocard small { color:#5B6B54; }
.ver { font-size:11px; color:#8AA088; text-align:right; }
div[data-testid="stButton"] > button[kind="primary"] { background:linear-gradient(90deg,#2E7D4F,#65A30D) !important;
  color:#fff !important; border-radius:999px !important; font-weight:800 !important; padding:11px 24px !important; }
div[data-testid="stButton"] > button[kind="secondary"] { background:#fff !important; color:#0E3B24 !important;
  border:2px solid #2E7D4F !important; border-radius:999px !important; font-weight:800 !important; }
div[data-testid="stExpander"] summary p { color:#0E3B24 !important; font-weight:800 !important; }
div[data-testid="stMetricValue"] div { color:#0E3B24 !important; }
label[data-testid="stWidgetLabel"] p { color:#0E3B24 !important; font-weight:800 !important; }
</style>"""


def inject_theme():
    st.markdown(THEME, unsafe_allow_html=True)


def go(view: str, **state):
    for k, v in state.items():
        st.session_state[k] = v
    st.session_state.view = view
    st.rerun()


def navbar(active: str):
    st.markdown('<div class="strip-dark" style="padding:10px 22px;display:flex;justify-content:space-between;'
                'align-items:center;flex-wrap:wrap;gap:6px">'
                '<span style="font-size:19px;font-weight:800">🌾 AgriPredict '
                '<small style="color:#E9C46A">Smart Crop Yield Prediction</small></span>'
                '<span style="font-size:12.5px">FAO 1990–2013 · 6 live models · seed 42</span></div>',
                unsafe_allow_html=True)
    cols = st.columns([1.6, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    labels = ["🌾", "Home", "Prep", "Crops", "Predict", "Lab", "Insights", "Simulator", "AI", "About"]
    keys = ["logo", "Home", "Prep", "Crops", "Predict", "Lab", "Insights", "Simulator", "AI", "About"]
    for col, lab, v in zip(cols, labels, keys):
        with col:
            target = "Home" if v == "logo" else v
            if st.button(lab, key=f"nav_{v}",
                         type="primary" if target == active and v != "logo" else "secondary",
                         use_container_width=True):
                go(target)


def footer():
    st.markdown('<div class="strip-dark" style="margin-top:26px">'
                '<b style="color:#fff">🌾 AgriPredict</b> · Smart Crop Yield Prediction · MSc Machine Learning Lab<br>'
                '<small>Yield data: FAO via Zenodo (CC-BY-4.0) · Photos: Unsplash + Wikimedia Commons '
                '(assets/ATTRIBUTION.md) · Python · Streamlit · scikit-learn · © 2026 · '
                'historical estimates, never guarantees</small></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ver">AgriPredict {APP_VERSION} · cream theme · all text dark-on-light by design</div>',
                unsafe_allow_html=True)


def titlebar(eyebrow: str, title: str, sub: str = ""):
    st.markdown(f'<div class="titlebar"><div class="eyebrow">{eyebrow}</div><h2>{title}</h2>'
                + (f'<p>{sub}</p>' if sub else '') + '</div>', unsafe_allow_html=True)


def upick(userfile: str, fallback: str) -> str:
    """User-supplied photo if present in assets/user/, else the built-in fallback."""
    return f"user/{userfile}" if (ASSETS / "user" / userfile).exists() else fallback


@st.cache_data
def img_b64(path: str) -> str:
    p = ASSETS / path
    if not p.exists():
        return ""
    import base64
    return base64.b64encode(p.read_bytes()).decode()


def photo(name: str, caption: str = ""):
    b = img_b64(name)
    if not b:
        return
    cap = f'<div class="capline">{caption}</div>' if caption else ""
    st.markdown(f'<img class="banner" src="data:image/jpeg;base64,{b}" alt="{caption}">{cap}',
                unsafe_allow_html=True)


def fig_img(name: str, caption: str = ""):
    p = FIG / name
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)


@st.cache_data(show_spinner="Loading field records...")
def load_data():
    return pd.read_csv(DATA)


@st.cache_data
def load_meta():
    return json.loads(META.read_text())


@st.cache_data
def load_sweep():
    p = BASE / "models" / "degree_sweep.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_resource(show_spinner="Loading trained models...")
def load_best():
    return joblib.load(BASE / "models" / "best_pipeline.pkl")


def init_state():
    for k, v in {"view": "Home", "explorer_crop": "Rice, paddy", "lab_model": "Random Forest",
                 "pred_crop": None, "pred_log": [], "sim_log": [],
                 "game": None, "game_n": 0, "game_score": 0, "game_rounds": 0,
                 "upload_df": None, "upload_name": "", "farmer_chat": [],
                 "ai_mode": "Offline", "mongo_uri": "mongodb://localhost:27017/",
                 "mongo_ok": False, "mongo_msg": ""}.items():
        if k not in st.session_state:
            st.session_state[k] = v


def active_frame():
    """(base_df, source_label): uploaded data if present, else the FAO default."""
    if st.session_state.get("upload_df") is not None:
        return st.session_state["upload_df"], f"uploaded ({st.session_state.get('upload_name', 'file')})"
    return load_data(), "FAO default (data/crop_yield.csv)"


def backend():
    try:
        df = load_data(); meta = load_meta(); sweep = load_sweep(); pipe = load_best()
        return df, meta, sweep, pipe
    except Exception as e:
        st.error(f"Backend not ready ({e}). Run `py train.py` first, then relaunch.")
        st.stop()


def model_row(comp, name: str):
    return comp[comp["Model"] == name].iloc[0]


def load_team() -> dict:
    try:
        d = json.loads(TEAM_FILE.read_text(encoding="utf-8"))
        return {**DEFAULT_TEAM, **{k: str(v) for k, v in d.items()}}
    except Exception:
        return dict(DEFAULT_TEAM)


def save_team(team: dict):
    TEAM_FILE.write_text(json.dumps({k: str(team.get(k, "")) for k in DEFAULT_TEAM}, indent=2,
                                    ensure_ascii=False), encoding="utf-8")


def team_line(team: dict) -> str:
    name = team.get("name") or "[Your Name]"
    members = f" · Team: {team['team']}" if team.get("team") else ""
    guide = team.get("guide") or "[Guide Name]"
    college = f" · {team['college']}" if team.get("college") else ""
    return f"{name}{members} · MSc Machine Learning Lab · Guide: {guide}{college}"
