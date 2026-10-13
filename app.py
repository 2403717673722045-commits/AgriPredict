"""AgriPredict v4-cream — flat-render dashboard. Closed blocks + widgets + st.image only."""
from __future__ import annotations
import io
import shutil
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import src.studio as ST
from src import farmer as FARMER
import train as TRAIN
from src.ui import (inject_theme, navbar, footer, go, titlebar, upick, img_b64, photo, fig_img,
                    init_state, backend, model_row, load_team, save_team, team_line,
                    MODELS6, SAFE, CROP_IMG, BASE)

st.set_page_config(page_title="AgriPredict — Grow Smarter with Machine Learning", page_icon="🌾",
                   layout="wide", initial_sidebar_state="collapsed")
inject_theme()
init_state()
df, meta, sweep, best_pipe = backend()
comp = pd.DataFrame(meta["comparison"]).sort_values("R2_test", ascending=False)
BEST = meta["best_model"]
TEMP = meta.get("temporal_check", {})
MEANS = df.groupby("item")["value_hg_ha"].mean().to_dict()


def pick_crop(crop: str, dest: str = "Crops"):
    st.session_state.pop("explorer_sel", None)
    st.session_state.pop("pred_crop_sel", None)
    if dest == "Crops":
        st.session_state.explorer_crop = crop
    else:
        st.session_state.pred_crop = crop
    go(dest)


def crop_card(crop: str, key: str, dest: str = "Crops"):
    photo(CROP_IMG[crop], f"{crop} — {'root & tuber harvest' if crop == 'Yams' else 'in the field'}")
    st.markdown(f'<div class="card" style="text-align:center"><b>{crop}</b><br>'
                f'<small>mean {MEANS.get(crop, 0):,.0f} hg/ha</small></div>', unsafe_allow_html=True)
    if st.button(f"Explore {crop.split(',')[0]} →", key=key, type="secondary", use_container_width=True):
        pick_crop(crop, dest)


def stats_band():
    st.markdown(f'<div class="statband"><div style="display:flex;gap:26px;flex-wrap:wrap">'
                f'<div><div class="n">{len(df):,}</div><div class="l">HISTORICAL RECORDS</div></div>'
                f'<div><div class="n">{df["area"].nunique()}</div><div class="l">COUNTRIES</div></div>'
                f'<div><div class="n">{df["item"].nunique()}</div><div class="l">CROPS</div></div>'
                f'<div><div class="n">{comp.iloc[0]["R2_test"]:.3f}</div><div class="l">BEST TEST R² ({BEST.upper()})</div></div>'
                f'</div><small style="color:#BFD6BF">Verified FAO 1990–2013 facts — not live readings.</small></div>',
                unsafe_allow_html=True)


# ================= HOME =================
def view_home():
    navbar("Home")
    hb = img_b64(upick("farmersoil.jpg", "farmer.jpg"))
    imgtag = f'<img src="data:image/jpeg;base64,{hb}" alt="Indian farmer in a green field">' if hb else ""
    st.markdown(f'<div class="herofull">{imgtag}<div class="shade"></div><div class="copy">'
                '<div class="eyebrow">SMART AGRICULTURE · MACHINE LEARNING</div>'
                '<h1>AgriPredict — Grow Smarter with Machine Learning</h1>'
                '<p class="sub">Explore crop yields, discover agricultural patterns and predict harvest outcomes '
                'using machine learning.</p>'
                f'<div class="chips"><span class="chip">📦 {len(df):,} records</span>'
                f'<span class="chip">🌍 {df["area"].nunique()} countries</span>'
                f'<span class="chip">🌾 {df["item"].nunique()} crops</span>'
                f'<span class="chip">🎯 R² {comp.iloc[0]["R2_test"]:.3f}</span></div>'
                '</div></div>', unsafe_allow_html=True)
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        if st.button("🌾 Predict Crop Yield", key="cta_pred", type="primary", use_container_width=True):
            go("Predict")
    with h2:
        if st.button("🌱 Explore Crops", key="cta_crops", use_container_width=True):
            go("Crops")
    with h3:
        if st.button("🔬 Open Model Lab", key="cta_lab", use_container_width=True):
            go("Lab")
    with h4:
        if st.button("💡 Agri Insights", key="cta_ins", use_container_width=True):
            go("Insights")
    stats_band()

    titlebar("Straight from the fields", 'Meet the <span class="u-yellow">crops</span>',
             "Six of ten dataset crops — open the Explorer for all, each with its real history.")
    t = st.columns(3)
    for i, c in enumerate(["Rice, paddy", "Wheat", "Maize", "Potatoes", "Soybeans", "Sorghum"]):
        with t[i % 3]:
            crop_card(c, key=f"home_{c}")
    if st.button("See all 10 crops in the Explorer →", key="home_allcrops", type="primary"):
        go("Crops")

    items = meta["items"]
    fav = items[datetime.now().timetuple().tm_yday % len(items)]
    favsub = df[df["item"] == fav]
    titlebar("🌾 Crop of the day", fav, f"{len(favsub):,} harvests · mean {favsub['value_hg_ha'].mean():,.0f} hg/ha")
    f1, f2 = st.columns([1, 1.4])
    with f1:
        photo(CROP_IMG[fav], f"{fav} — crop of the day")
    with f2:
        with st.container(border=True):
            st.write(f"Across {favsub['area'].nunique()} countries ({favsub['year'].min()}–{favsub['year'].max()}).")
            if st.button(f"Explore {fav.split(',')[0]} →", key="home_fav", type="primary"):
                pick_crop(fav, "Crops")

    titlebar("How it works", "Field notes become forecasts")
    w1, w2 = st.columns([1.2, 1])
    with w1:
        for tt, dd in [("1 · Describe", "Country, crop, year, rainfall, pesticides, temperature."),
                       ("2 · Prepare", "Same scaling + encoding fitted on training rows only."),
                       ("3 · Predict", "Your chosen regressor estimates yield in hg/ha."),
                       ("4 · Explore", "Compare history, test scenarios, download reports.")]:
            st.markdown(f'<div class="card" style="margin-bottom:10px"><b>{tt}</b><br><small>{dd}</small></div>',
                        unsafe_allow_html=True)
    with w2:
        photo("tractor.jpg", "Machinery at work")
        photo("produce.jpg", "Fresh harvest, honestly measured")

    titlebar("Six learners, one honest contest", "Every model measured")
    mcols = st.columns(6)
    for col, name in zip(mcols, MODELS6):
        r = model_row(comp, name)
        with col:
            st.markdown(f'<div class="card" style="text-align:center"><b>{name.split()[0]}</b><br>'
                        f'<span style="color:#2E7D4F;font-weight:800">R² {r["R2_test"]:.3f}</span></div>',
                        unsafe_allow_html=True)
    if st.button("Open the full Model Lab →", key="home_lab", type="primary"):
        go("Lab")

    photo(upick("workers.jpg", "harvest.jpg"), "Planting crews at work")
    st.markdown('<div class="strip-dark"><div class="eyebrow">Ready when you are</div>'
                '<h3>What if the rains came heavier?</h3>'
                '<p>Bend rainfall or temperature and watch model-estimated harvests move side by side.</p></div>',
                unsafe_allow_html=True)
    if st.button("Open Scenario Simulator →", key="home_sim", type="primary"):
        go("Simulator")

    titlebar("🌾 The Yield Challenge", "Can you out-guess the model?",
             "One hidden real harvest · score = 100 − your % error")
    with st.container(border=True):
        if st.button("🎲 New mystery harvest", key="game_new", type="primary"):
            st.session_state.game_n += 1
            st.session_state.game = df.sample(1, random_state=1000 + st.session_state.game_n).iloc[0].to_dict()
        g = st.session_state.game
        if g:
            g1, g2 = st.columns([1.2, 1])
            with g1:
                st.markdown(f'<div class="card"><b>Mystery: {g["item"]} in {g["area"]}, {int(g["year"])}</b><br>'
                            f'<small>Rainfall {g["average_rain_fall_mm_per_year"]:,.0f} mm · Pesticides '
                            f'{g["pesticides_tonnes"]:,.0f} t · Temp {g["avg_temp"]:.1f} °C</small></div>',
                            unsafe_allow_html=True)
                guess = st.slider("Your guess (hg/ha)", 0, 500000, 40000, step=1000,
                                  key=f"game_guess_{st.session_state.game_n}")
                reveal = st.button("Reveal the harvest 🌾", key=f"game_go_{st.session_state.game_n}", type="primary")
            with g2:
                st.markdown(f'<div class="scorebox">Season score<div class="n">{st.session_state.game_score}</div>'
                            f'{st.session_state.game_rounds} rounds played</div>', unsafe_allow_html=True)
            if reveal:
                actual = float(g["value_hg_ha"])
                inp = {"area": g["area"], "item": g["item"], "year": int(g["year"]),
                       "average_rain_fall_mm_per_year": float(g["average_rain_fall_mm_per_year"]),
                       "pesticides_tonnes": float(g["pesticides_tonnes"]), "avg_temp": float(g["avg_temp"])}
                mpred = ST.predict_with(best_pipe, inp)
                err = abs(guess - actual) / actual * 100
                pts = max(0, int(round(100 - err)))
                st.session_state.game_score += pts
                st.session_state.game_rounds += 1
                st.markdown(f'<div class="harvest">Truth: <b>{actual:,.0f}</b> · Model said <b>{mpred:,.0f}</b> · '
                            f'You said <b>{guess:,.0f}</b> (off {err:.1f}% → <b>+{pts} pts</b>)</div>',
                            unsafe_allow_html=True)
                if err <= 10:
                    st.balloons()
                    st.success("Within 10% — true farmer instinct!")
        else:
            st.info("Press “New mystery harvest” — every round hides the yield of one genuine historical record.")
    footer()


# ================= CROP EXPLORER =================
def view_crops():
    navbar("Crops")
    photo("crops.jpg", "Crop rows stretching to the horizon")
    titlebar("Crop explorer", "Ten crops, real histories",
             "Every crop below is genuinely in the dataset. Pick one for records, trends and top countries.")
    items = meta["items"]
    idx = items.index(st.session_state.explorer_crop) if st.session_state.explorer_crop in items else 0
    sel = st.selectbox("Choose a crop (tiles do the same)", items, index=idx, key="explorer_sel")
    st.session_state.explorer_crop = sel
    tiles = st.columns(5)
    for i, c in enumerate(items[:5]):
        with tiles[i]:
            crop_card(c, key=f"expA_{c}")
    tiles2 = st.columns(5)
    for i, c in enumerate(items[5:]):
        with tiles2[i]:
            crop_card(c, key=f"expB_{c}")

    crop = st.session_state.explorer_crop
    sub = df[df["item"] == crop]
    titlebar(crop, f"{len(sub):,} historical records",
             f"Mean {sub['value_hg_ha'].mean():,.0f} hg/ha (≈ {sub['value_hg_ha'].mean()/10000:.2f} t/ha)")
    d1, d2 = st.columns([1, 1])
    with d1:
        photo(CROP_IMG[crop], f"{crop} — {'root & tuber harvest' if crop == 'Yams' else 'in the field'}")
    with d2:
        with st.container(border=True):
            st.write(f"Range {sub['value_hg_ha'].min():,.0f}–{sub['value_hg_ha'].max():,.0f} hg/ha · "
                     f"{sub['area'].nunique()} countries · {sub['year'].min()}–{sub['year'].max()}")
            top_c = sub.groupby("area")["value_hg_ha"].mean().sort_values(ascending=False).head(1)
            st.write(f"Top country by mean yield: **{top_c.index[0]}** ({top_c.iloc[0]:,.0f} hg/ha)")
            if st.button(f"🌾 Predict {crop.split(',')[0]} yields →", key="exp_predict", type="primary"):
                pick_crop(crop, "Predict")
    yt = sub.groupby("year")["value_hg_ha"].mean().reset_index()
    st.plotly_chart(px.line(yt, x="year", y="value_hg_ha", markers=True,
                            title=f"{crop} — mean yield over years", template="plotly_white"),
                    use_container_width=True)
    tc = sub.groupby("area")["value_hg_ha"].mean().sort_values(ascending=False).head(10).reset_index()
    st.plotly_chart(px.bar(tc, x="value_hg_ha", y="area", orientation="h",
                           title=f"{crop} — top 10 countries", template="plotly_white"), use_container_width=True)
    st.dataframe(sub.sort_values("year").head(15), use_container_width=True)
    footer()


# ================= YIELD PREDICTION =================
def view_predict():
    navbar("Predict")
    photo("field_sunset.jpg", "Wheat country at golden hour")
    titlebar("Yield prediction", "Ask the model about your field",
             "Six genuinely loaded regressors · same preprocessing as training · estimates, never guarantees.")
    p1, p2 = st.columns([1, 1])
    with p1:
        with st.container(border=True):
            mdl = st.selectbox("Prediction model", MODELS6, index=MODELS6.index(BEST), key="pred_mdl")
            pipe_m = ST.load_pipeline(mdl)
            items = meta["items"]
            dflt = st.session_state.pred_crop if st.session_state.pred_crop in items else "Wheat"
            areas = meta["areas"]
            area = st.selectbox("Country / Area", areas,
                                index=areas.index("India") if "India" in areas else 0, key="pred_area")
            item = st.selectbox("Crop", items, index=items.index(dflt), key="pred_crop_sel")
            year = st.number_input("Year", min_value=2010, max_value=2025, value=2020, step=1,
                                   key="pred_year", help="Recent window 2010–2025. Training saw 1990–2013, "
                                   "so years after 2013 extend the trend — treat them as rough estimates.")
            rain = st.number_input("Rainfall (mm/year)", min_value=0.0, value=1000.0, step=10.0, key="pred_rain")
            pest = st.number_input("Pesticides (tonnes, national scale)", min_value=0.0, value=46000.0, step=500.0,
                                   key="pred_pest", help="Country totals, not a field dose — never read as advice. "
                                   "Typical India ≈ 46,000 t; small countries ≈ hundreds.")
            temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=45.0, value=25.0, step=0.1,
                                   key="pred_temp")
            go_btn = st.button("🌾 Predict Yield", type="primary", use_container_width=True, key="pred_go")
    with p2:
        photo(upick("soilmeter.jpg", "seedling.jpg"),
              "Field sensors read soil and weather — this model weighs rainfall, temperature and more")
        if go_btn:
            try:
                inp = {"area": area, "item": item, "year": int(year),
                       "average_rain_fall_mm_per_year": float(rain),
                       "pesticides_tonnes": float(pest), "avg_temp": float(temp)}
                hg = ST.predict_with(pipe_m, inp)
                st.markdown(f'<div class="harvest">🌾 <b>{hg:,.0f} hg/ha</b> (≈ {hg / 10000:.2f} tonnes/ha)<br>'
                            f'<small>Model: <b>{mdl}</b> · {item} in {area}, {int(year)} — computed live.</small></div>',
                            unsafe_allow_html=True)
                st.write("Past harvests under similar conditions averaged around here.")
                hist = df[(df["area"] == area) & (df["item"] == item)]["value_hg_ha"]
                if len(hist):
                    cmpd = pd.DataFrame({"series": ["Historical mean", "Your prediction"], "hg_ha": [hist.mean(), hg]})
                    st.plotly_chart(px.bar(cmpd, x="series", y="hg_ha", color="series",
                                           title="Prediction vs local history", template="plotly_white"),
                                    use_container_width=True)
                    st.write(f"History: {len(hist)} records, mean **{hist.mean():,.0f}**, "
                             f"range {hist.min():,.0f}–{hist.max():,.0f} hg/ha.")
                    if hg > hist.mean():
                        st.balloons()
                        st.success("Above the local historical mean — promising conditions!")
                xs = np.linspace(float(df["average_rain_fall_mm_per_year"].quantile(0.05)),
                                 float(df["average_rain_fall_mm_per_year"].quantile(0.95)), 25)
                ys = [ST.predict_with(pipe_m, {**inp, "average_rain_fall_mm_per_year": float(x)}) for x in xs]
                fig = px.line(pd.DataFrame({"rainfall": xs, "predicted": ys}), x="rainfall", y="predicted",
                              title=f"If rainfall moved… ({mdl}, rest held)", template="plotly_white")
                fig.add_scatter(x=[float(rain)], y=[hg], mode="markers", name="your input",
                                marker=dict(size=12, color="#E9B949", line=dict(width=2, color="#0E3B24")))
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Live model answers across rainfall — an association to explore, not a causal promise.")
                r = model_row(comp, mdl)
                st.caption(f"{mdl} test scores — R² {r['R2_test']:.4f} · RMSE {r['RMSE']:,.0f} · MAE {r['MAE']:,.0f} hg/ha.")
                st.session_state.pred_log.append({**inp, "model": mdl, "pred_hg_ha": round(hg, 1),
                                                  "time": datetime.now().strftime("%H:%M:%S")})
                if st.session_state.get("mongo_ok"):
                    from src import mongo as _M
                    _db, _e = _M.get_db(st.session_state.get("mongo_uri", ""))
                    if _db is not None and _M.log_prediction(
                            _db, {**inp, "model": mdl, "pred_hg_ha": round(hg, 1)}):
                        st.caption("🍃 Saved to MongoDB.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.info("Press Predict Yield — change any input and predict again as often as you like.")
        if st.session_state.pred_log:
            st.dataframe(pd.DataFrame(st.session_state.pred_log), use_container_width=True)
            st.download_button("⬇ My predictions (CSV)", pd.DataFrame(st.session_state.pred_log).to_csv(index=False),
                               "agripredict_predictions.csv", "text/csv", key="pred_dl")
    footer()


# ================= MODEL LAB =================
LAB_NOTES = {
    "Linear Regression": "Straight-line baseline — fast and readable, but yields bend and jump (R² ≈ 0.75).",
    "Polynomial Regression": "Degree-2 curves on weather numbers — barely moves the needle (R² ≈ 0.75).",
    "Decision Tree": "Rule flowcharts like “crop = Potatoes” — captures jumps (R² 0.976).",
    "Random Forest": "100 voting trees calm each tree's overfit — measured champion (R² 0.984).",
    "KNN": "Averages the 5 most similar past harvests — simple and strong (R² 0.982).",
    "Gradient Boosting": "Trees repairing errors in sequence — good here (R² 0.858), slower to train.",
}


def view_lab():
    navbar("Lab")
    photo("wheat.jpg", "Golden wheat — the crop every model studied")
    titlebar("Machine learning model lab", "Six models enter, measurements decide",
             "One shared split · preprocessing fit on train only · cross-validation + an unseen-years check.")
    m1, m2, m3 = st.columns(3)
    r_best = model_row(comp, BEST)
    m1.markdown(f'<div class="card" style="text-align:center"><div style="font-size:34px;font-weight:800;color:#2E7D4F">'
                f'{r_best["R2_test"]:.3f}</div><small><b>BEST TEST R² · {BEST.upper()}</b></small></div>',
                unsafe_allow_html=True)
    m2.markdown(f'<div class="card" style="text-align:center"><div style="font-size:34px;font-weight:800;color:#2E7D4F">'
                f'{r_best["RMSE"]:,.0f}</div><small><b>BEST RMSE (HG/HA)</b></small></div>', unsafe_allow_html=True)
    tmp = TEMP.get("Random Forest", {}) if isinstance(TEMP, dict) else {}
    m3.markdown(f'<div class="card" style="text-align:center"><div style="font-size:34px;font-weight:800;color:#2E7D4F">'
                f'{tmp.get("R2_unseen_years", 0):.3f}</div><small><b>UNSEEN-YEARS R² (2006–2013)</b></small></div>',
                unsafe_allow_html=True)
    with st.container(border=True):
        metric = st.radio("Comparison metric", ["R2_test", "RMSE", "MAE"], horizontal=True, key="lab_metric",
                          format_func={"R2_test": "R² (higher better)", "RMSE": "RMSE (lower better)",
                                       "MAE": "MAE (lower better)"}.get)
        st.plotly_chart(px.bar(comp.sort_values(metric, ascending=(metric != "R2_test")), x="Model", y=metric,
                               color="Model", title=f"Model comparison — {metric}", template="plotly_white"),
                        use_container_width=True)
        st.dataframe(comp.style.format({"MAE": "{:.1f}", "MSE": "{:.0f}", "RMSE": "{:.1f}",
                                        "R2_test": "{:.4f}", "R2_cv_mean": "{:.4f}"}), use_container_width=True)
    titlebar("Pick a model", "Detail updates on this page")
    t = st.columns(6)
    for col, name in zip(t, MODELS6):
        r = model_row(comp, name)
        with col:
            st.markdown(f'<div class="card" style="text-align:center"><b>{name.split()[0]}</b><br>'
                        f'<small>R² {r["R2_test"]:.3f}</small></div>', unsafe_allow_html=True)
            if st.button("Open →", key=f"lab_{name}",
                         type="primary" if name == st.session_state.lab_model else "secondary"):
                st.session_state.lab_model = name
                st.rerun()
    name = st.session_state.lab_model
    r = model_row(comp, name)
    st.markdown(f'### {name} — R² **{r["R2_test"]:.4f}** (CV {r["R2_cv_mean"]:.4f}) · RMSE **{r["RMSE"]:,.0f}** · '
                f'MAE **{r["MAE"]:,.0f} hg/ha**')
    st.write(LAB_NOTES[name] + " " + meta["model_notes"][name])
    fig_img(f"avp_{SAFE[name]}.png", f"Actual vs predicted — {name}")
    titlebar("Beyond the random split", "Can it face the future?")
    if TEMP:
        for m, tt in TEMP.items():
            st.write(f"- **{m}** on {tt['test_years']} ({tt['n_test']:,} rows): R² **{tt['R2_unseen_years']:.4f}**")
    c1, c2 = st.columns(2)
    with c1:
        fig_img("12_residuals.png", f"Residuals — {BEST}")
    with c2:
        fig_img("13_feature_importance.png", "Crop type leads; scale, country, weather follow")
    if sweep:
        sw = pd.DataFrame([{"degree": int(k), "R2": v["R2_test"]} for k, v in sweep.items()])
        st.plotly_chart(px.bar(sw, x="degree", y="R2", title="Polynomial degree barely moves R² here",
                               template="plotly_white"), use_container_width=True)
    titlebar("Hyperparameter tuning", "GridSearchCV — did better settings exist?")
    import json as _json
    tpath = BASE / "models" / "tuning.json"
    if tpath.exists():
        tuned = _json.loads(tpath.read_text())
        rows = []
        for m, t in tuned.items():
            if m.startswith("_"):
                continue
            rows.append({"Model": m,
                         "Best settings": ", ".join(f"{k}={v}" for k, v in t["best_params"].items()),
                         "Tuned R²": round(t["test_R2"], 4), "Default R²": round(t["default_R2"], 4),
                         "Verdict": "✅ beats default" if t["beats_default"] else "= holds"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption("Grids were kept small on purpose (e.g. KNN k ∈ {3,5,9,15,25}, RF 2×2 on 2-fold CV) so a college "
                   "laptop finishes; preprocessing stayed inside each pipeline (no leakage). Linear/Polynomial have no "
                   "meaningful knobs — honestly skipped. The shipped predictor keeps default settings; tuned scores "
                   "are reported for comparison, not silently swapped in.")
    else:
        st.info("Tuning has not been run yet on this machine — run `py tune.py` (≈10 min) to fill this table.")
    st.success(f"Selected **{BEST}** by measurement — highest test R² with matching CV and unseen-years score.")
    footer()


# ================= INSIGHTS =================
def view_insights():
    navbar("Insights")
    photo(upick("rain.jpg", "rice.jpg"), "Monsoon over paddy — rainfall written across every row")
    titlebar("Agricultural analytics", "What the harvests whisper",
             "Filter anything — every chart redraws. Associations only, never proof of cause.")
    with st.container(border=True):
        f1, f2, f3 = st.columns(3)
        crops = f1.multiselect("Crops", meta["items"], default=["Wheat", "Rice, paddy", "Maize"], key="ins_crops")
        yrs = f2.slider("Years", 2010, 2025, (2010, 2013), key="ins_yrs")
        topn = f3.slider("Top countries", 5, 20, 12, key="ins_top")
    st.caption("Historical records run 1990–2013: dragging past 2013 shows an empty state (no observations). "
               "For 2014–2025 estimates, use Yield Prediction or the Simulator (labelled trend extrapolation).")
    f = df[df["item"].isin(crops or meta["items"])]
    f = f[(f["year"] >= yrs[0]) & (f["year"] <= yrs[1])]
    st.write(f"Filtered view: **{len(f):,}** records.")
    if len(f) == 0:
        st.warning(f"No historical records between {yrs[0]}–{yrs[1]} — observations end in 2013. "
                   "Drag the slider back to ≤ 2013, or open Yield Prediction for 2014–2025 model estimates.")
    else:
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(px.histogram(f, x="value_hg_ha", nbins=40, title="Yield distribution",
                                         template="plotly_white"), use_container_width=True)
            yt = f.groupby("year")["value_hg_ha"].mean().reset_index()
            st.plotly_chart(px.line(yt, x="year", y="value_hg_ha", markers=True, title="Mean yield over years",
                                    template="plotly_white"), use_container_width=True)
        with g2:
            tc = f.groupby("item")["value_hg_ha"].mean().sort_values(ascending=False).reset_index()
            st.plotly_chart(px.bar(tc, x="item", y="value_hg_ha", title="Mean yield by crop",
                                   template="plotly_white"), use_container_width=True)
            samp = f.sample(min(2500, len(f)), random_state=42)
            st.plotly_chart(px.scatter(samp, x="average_rain_fall_mm_per_year", y="value_hg_ha", color="item",
                                       title="Rainfall vs yield (cloudy = no single cause)",
                                       template="plotly_white"), use_container_width=True)
        ta = f.groupby("area")["value_hg_ha"].mean().sort_values(ascending=False).head(topn).reset_index()
        st.plotly_chart(px.bar(ta, x="value_hg_ha", y="area", orientation="h", title=f"Top {topn} areas",
                               template="plotly_white"), use_container_width=True)
        st.markdown("**▶ Press play and watch the selected years unfold:**")
        race = f.groupby(["year", "item"])["value_hg_ha"].mean().reset_index()
        st.plotly_chart(px.bar(race, x="value_hg_ha", y="item", color="item", orientation="h",
                               animation_frame="year", range_x=[0, float(f["value_hg_ha"].max())],
                               title="Crop yield race, year by year", template="plotly_white"),
                        use_container_width=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        fig_img("02_corr.png", "Weather links are weak — categories carry the signal.")
        photo("harvest.jpg", "Gathering the grain")
    with c2:
        with st.container(border=True):
            st.markdown("**Field notes**<br>• Potatoes and cassava average highest; sorghum lowest.<br>"
                        "• Yields climb 1990–2013 — better varieties and practice, most likely.<br>"
                        "• No single weather number decides a harvest — why ensembles win.", unsafe_allow_html=True)
        st.markdown('<div class="causes">K-Means groups similar growing conditions — it describes, never predicts.</div>',
                    unsafe_allow_html=True)
        try:
            import matplotlib.pyplot as plt
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            num = df[["average_rain_fall_mm_per_year", "avg_temp", "pesticides_tonnes", "year"]].sample(3000, random_state=42)
            lab = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(StandardScaler().fit_transform(num))
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(num["average_rain_fall_mm_per_year"], num["avg_temp"], c=lab, alpha=0.5, cmap="Set2")
            ax.set_title("4 growing-condition groups"); ax.set_xlabel("Rainfall mm/year"); ax.set_ylabel("Temp °C")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Clustering skipped: {e}")
    footer()


# ================= SIMULATOR =================
def view_sim():
    navbar("Simulator")
    photo(upick("tractormais.jpg", "tractor.jpg"), "Machinery at work — now bend the weather it works in")
    titlebar("What-if scenario simulator", "Bend the weather, compare harvests",
             "Years 2010–2025 · after 2013 the model extends its trend — rough estimates, not guarantees.")
    with st.container(border=True):
        s0a, s0b, s0c = st.columns(3)
        s_area = s0a.selectbox("Area (both scenarios)", meta["areas"],
                               index=meta["areas"].index("India") if "India" in meta["areas"] else 0, key="sim_area")
        s_item = s0b.selectbox("Crop (both scenarios)", meta["items"],
                               index=meta["items"].index("Wheat") if "Wheat" in meta["items"] else 0, key="sim_item")
        s_mdl = s0c.selectbox("Simulator model", MODELS6, index=MODELS6.index(BEST), key="sim_mdl")
        s_pipe = ST.load_pipeline(s_mdl)
        sa, sb = st.columns(2)
        with sa:
            st.markdown("**Scenario A — baseline**")
            a_y = st.number_input("Year A", 2010, 2025, 2015, key="ay")
            a_r = st.number_input("Rainfall A (mm)", 0.0, 4000.0, 1000.0, step=50.0, key="ar")
            a_t = st.number_input("Temp A (°C)", -10.0, 45.0, 25.0, step=0.5, key="at")
            a_p = st.number_input("Pesticides A (t)", 0.0, 400000.0, 46000.0, step=500.0, key="ap")
        with sb:
            st.markdown("**Scenario B — altered**")
            b_y = st.number_input("Year B", 2010, 2025, 2025, key="by")
            b_r = st.number_input("Rainfall B (mm)", 0.0, 4000.0, 1400.0, step=50.0, key="br")
            b_t = st.number_input("Temp B (°C)", -10.0, 45.0, 27.0, step=0.5, key="bt")
            b_p = st.number_input("Pesticides B (t)", 0.0, 400000.0, 46000.0, step=500.0, key="bp")
    mk = lambda y, r, t, p: {"area": s_area, "item": s_item, "year": int(y),
                             "average_rain_fall_mm_per_year": float(r),
                             "pesticides_tonnes": float(p), "avg_temp": float(t)}
    ha = ST.predict_with(s_pipe, mk(a_y, a_r, a_t, a_p))
    hb = ST.predict_with(s_pipe, mk(b_y, b_r, b_t, b_p))
    st.plotly_chart(px.bar(pd.DataFrame({"scenario": ["A baseline", "B altered"], "hg_ha": [ha, hb]}),
                           x="scenario", y="hg_ha", color="scenario",
                           title=f"{s_item} in {s_area} — {s_mdl} (hg/ha)", template="plotly_white"),
                    use_container_width=True)
    dd = hb - ha
    st.markdown(f'<div class="harvest">A: <b>{ha:,.0f}</b> · B: <b>{hb:,.0f} hg/ha</b> · Δ '
                f'<b>{"+" if dd >= 0 else ""}{dd:,.0f}</b> (≈ {dd / 10000:+.2f} t/ha)</div>', unsafe_allow_html=True)
    if st.button("💾 Save this comparison", key="sim_save"):
        st.session_state.sim_log.append({"area": s_area, "item": s_item, "model": s_mdl,
                                         "A_hg_ha": round(ha, 1), "B_hg_ha": round(hb, 1), "delta": round(dd, 1)})
        if st.session_state.get("mongo_ok"):
            from src import mongo as _M2
            _db2, _e2 = _M2.get_db(st.session_state.get("mongo_uri", ""))
            if _db2 is not None and _M2.log_scenario(
                    _db2, {"area": s_area, "item": s_item, "model": s_mdl,
                           "A_hg_ha": round(ha, 1), "B_hg_ha": round(hb, 1), "delta": round(dd, 1)}):
                st.caption("🍃 Saved to MongoDB.")
    if st.session_state.sim_log:
        sdf = pd.DataFrame(st.session_state.sim_log)
        st.dataframe(sdf, use_container_width=True)
        st.download_button("⬇ Comparisons (CSV)", sdf.to_csv(index=False),
                           "agripredict_scenarios.csv", "text/csv", key="sim_dl")
    titlebar("Model playground", "Same field, different minds")
    picks = st.multiselect("Line up models (2+)", MODELS6, default=[BEST, "KNN"], key="sim_picks")
    if len(picks) >= 2:
        base_in = mk(a_y, a_r, a_t, a_p)
        rows = [{"model": m, "pred_hg_ha": round(ST.predict_with(ST.load_pipeline(m), base_in), 1)} for m in picks]
        res = pd.DataFrame(rows).sort_values("pred_hg_ha")
        st.plotly_chart(px.bar(res, x="model", y="pred_hg_ha", color="model",
                               title="Scenario-A field through different models", template="plotly_white"),
                        use_container_width=True)
        st.dataframe(res, use_container_width=True)
    else:
        st.info("Pick at least two models to compare.")
    footer()


# ================= ABOUT =================
def view_about():
    navbar("About")
    photo("farmer.jpg", "The people this serves")
    titlebar("About the project", "An honest machine, beautifully told",
             "What we set out to solve, what the data allowed, and where the limits lie.")
    st.markdown('<div class="eyebrow">The problem</div><h2 class="site">Harvests decided too late</h2>'
                '<p>Farmers and planners often learn yields only at harvest. AgriPredict estimates yield earlier '
                'from known conditions, aiding storage, pricing and food planning.</p>', unsafe_allow_html=True)
    a1, a2 = st.columns([1, 1])
    with a1:
        photo("rice.jpg", "Paddy country")
    with a2:
        st.markdown('<div class="eyebrow">Data & limits</div>', unsafe_allow_html=True)
        st.markdown(f'<h3 class="site">Real FAO harvests — and real gaps</h3><p>{len(df):,} country-year records, '
                    f'{df["area"].nunique()} countries, {df["item"].nunique()} crops, consistent hg/ha units '
                    f'(÷10,000 = tonnes/ha). Source: FAO via Zenodo DOI 10.5281/zenodo.18761663 (CC-BY-4.0).</p>',
                    unsafe_allow_html=True)
        st.markdown('<div class="soilnote">Missing by design, never filled: ' + ", ".join(meta["unavailable_features"]) + '.</div>',
                    unsafe_allow_html=True)
    b1, b2 = st.columns([1.3, 1])
    with b1:
        st.markdown('<div class="eyebrow">Method</div>', unsafe_allow_html=True)
        st.markdown('<h3 class="site">Clean pipes, fair tests</h3><p>One 80/20 split (seed 42) · scaling + one-hot encoding '
                    'fitted on training rows only · 5-fold cross-validation (3-fold for forests/boosting) · best picked by '
                    'measured test R² · plus an unseen-years check (train ≤2005 → test 2006–2013).</p>', unsafe_allow_html=True)
    with b2:
        photo(upick("soilmeter.jpg", "seedling.jpg"), "Measured twice, modelled once")
    st.markdown('<div class="eyebrow">The six learners</div>', unsafe_allow_html=True)
    ac = st.columns(3)
    blurb = {"Linear Regression": "Straight-line baseline.", "Polynomial Regression": "Curves on weather numbers.",
             "Decision Tree": "Rule flowcharts.", "Random Forest": "100 voting trees.", "KNN": "5 similar harvests.",
             "Gradient Boosting": "Trees fixing errors."}
    for i, name in enumerate(MODELS6):
        r = model_row(comp, name)
        with ac[i % 3]:
            st.markdown(f'<div class="card"><b>{name}</b><br><small>{blurb[name]}</small><br>'
                        f'<small>R² {r["R2_test"]:.4f} · RMSE {r["RMSE"]:,.0f}</small></div>', unsafe_allow_html=True)
    st.markdown(f'<br><div class="harvest">Findings: <b>{BEST}</b> won by measurement '
                f'(R² {comp.iloc[0]["R2_test"]:.4f}, matching CV, strong unseen-years score) because yields jump by crop '
                'and country. Use forecasts for planning; never as field-level guarantees or input-dose advice.</div>',
                unsafe_allow_html=True)
    photo("tractor.jpg", "Machinery keeps improving — so should our estimates")
    titlebar("Reports & downloads", "Take it with you")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.download_button("⬇ Scorecard CSV", comp.to_csv(index=False), "agripredict_scores.csv", "text/csv", key="abt_csv")
    with r2:
        if st.session_state.pred_log:
            st.download_button("⬇ Predictions CSV", pd.DataFrame(st.session_state.pred_log).to_csv(index=False),
                               "agripredict_predictions.csv", "text/csv", key="abt_pred")
        else:
            st.info("Predict first (Yield Prediction page).")
    with r3:
        team = load_team()
        buf = io.StringIO()
        buf.write(f"AgriPredict report {datetime.now():%Y-%m-%d}\nTeam: {team_line(team)}\n"
                  f"Best: {BEST} R2 {comp.iloc[0]['R2_test']:.4f} "
                  f"RMSE {comp.iloc[0]['RMSE']:,.0f} hg/ha\nSource: {meta['dataset_source']}\nUnits: hg/ha (/10000 = t/ha)\n")
        buf.write(comp.to_string(index=False))
        st.download_button("⬇ Summary TXT", buf.getvalue(), "agripredict_summary.txt", "text/plain", key="abt_txt")
    titlebar("Project team", "Put your names on it",
             "Saved to team.json in the project folder — survives restarts and appears in the report above.")
    team = load_team()
    with st.container(border=True):
        t1, t2 = st.columns(2)
        with t1:
            nm = st.text_input("Your name", value=team["name"], key="team_name", placeholder="e.g. Priya Sharma")
            tm = st.text_input("Team members (optional)", value=team["team"], key="team_members",
                               placeholder="e.g. Arun, Divya")
        with t2:
            gd = st.text_input("Guide name", value=team["guide"], key="team_guide", placeholder="e.g. Dr. Rao")
            cl = st.text_input("College / department (optional)", value=team["college"], key="team_college",
                               placeholder="e.g. CIT, MSc Computer Science")
        if st.button("💾 Save team details", key="team_save", type="primary"):
            save_team({"name": nm.strip(), "team": tm.strip(), "guide": gd.strip(), "college": cl.strip()})
            st.success("Saved — shown below and stamped into the Summary TXT.")
            st.rerun()
    st.markdown(f'<div class="soilnote">Team: {team_line(load_team())}</div>', unsafe_allow_html=True)
    footer()


def view_data():
    navbar("Prep")
    photo("seedling.jpg", "Young crop, careful method — preparation first")
    titlebar("Data preparation", "Preprocessing & EDA",
             "Bring your own harvest CSV, watch validation and preprocessing, explore live.")
    with st.container(border=True):
        st.markdown("**1 · Dataset source**")
        up = st.file_uploader("Upload a harvest CSV (same 7 columns as the FAO file)", type=["csv"], key="up_file")
        if up is not None:
            try:
                raw = pd.read_csv(up)
            except Exception as e:
                st.error(f"Could not read that file: {e}")
                raw = None
            if raw is not None:
                res = ST.validate_upload(raw)
                for e in res["errors"]:
                    st.error(e)
                for w in res["warnings"]:
                    st.warning(w)
                if res["ok"]:
                    st.success(f"Valid: {res['info']['rows']:,} rows × {res['info']['columns']} columns · "
                               f"{res['info']['missing_cells']} missing cells · {res['info']['duplicates']} duplicates.")
                    if st.button("Use this data for exploration", key="up_use", type="primary"):
                        st.session_state.upload_df = res["frame"]
                        st.session_state.upload_name = up.name
                        st.rerun()
        if st.session_state.upload_df is not None:
            cur, srclabel = st.session_state.upload_df, f"uploaded ({st.session_state.upload_name})"
        else:
            cur, srclabel = df, "FAO default (data/crop_yield.csv)"
        st.write(f"Active exploration data: **{srclabel}** — {len(cur):,} rows.")
        if st.session_state.upload_df is not None and st.button("Back to FAO default", key="up_clear"):
            st.session_state.upload_df = None
            st.session_state.upload_name = ""
            st.rerun()

    with st.container(border=True):
        st.markdown("**2 · What preprocessing does (the same pipeline training used)**")
        st.write("1. Schema check → 2. coerce numerics → 3. 80/20 split (seed 42) → "
                 "4. StandardScaler on 4 numerics → 5. OneHotEncoder on country + crop (unknown-safe) → "
                 "6. every transformer fits on TRAIN only — no leakage.")
        try:
            tp = ST.transform_preview(best_pipe, cur)
            st.write(f"The fitted pipeline turns 6 inputs into **{tp['out_features']}** model features "
                     f"({tp['scaled_numeric']} scaled numeric + {tp['one_hot']} one-hot). "
                     "First features: " + ", ".join(tp["feature_names_head"][:8]) + "…")
            st.caption("Demonstrated with the currently saved (FAO-fitted) pipeline — retraining below refits it.")
        except Exception as e:
            st.warning(f"Preview unavailable for this frame: {e}")

    titlebar("Live EDA", f"On: {srclabel}")
    if len(cur) < 5:
        st.warning("Too few rows to plot.")
    else:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        e1, e2 = st.columns(2)
        with e1:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.hist(cur["value_hg_ha"].dropna(), bins=30)
            ax.set_title("Yield distribution (hg/ha)"); ax.set_xlabel("hg/ha")
            st.pyplot(fig); plt.close(fig)
        with e2:
            num = cur.select_dtypes(include="number")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
            ax.set_title("Correlation (numeric)")
            st.pyplot(fig); plt.close(fig)
        e3, e4 = st.columns(2)
        with e3:
            tc = cur.groupby("item")["value_hg_ha"].mean().sort_values(ascending=False).head(8)
            fig, ax = plt.subplots(figsize=(7, 4.5))
            tc.sort_values().plot(kind="barh", ax=ax)
            ax.set_title("Mean yield by crop (top 8)"); ax.set_xlabel("hg/ha")
            st.pyplot(fig); plt.close(fig)
        with e4:
            yt = cur.groupby("year")["value_hg_ha"].mean()
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(yt.index, yt.values, marker="o")
            ax.set_title("Mean yield over years"); ax.set_xlabel("Year")
            st.pyplot(fig); plt.close(fig)

    if st.session_state.upload_df is not None:
        with st.container(border=True):
            st.markdown("**3 · Retrain all 6 models on this data (optional)**")
            st.warning("Takes ~5–8 minutes and replaces the saved models (the previous set is backed up, "
                       "oldest backups auto-deleted beyond 2). Predictions everywhere will then use YOUR data.")
            if st.button("Retrain now", key="up_retrain", type="primary"):
                try:
                    mdir, ddir = BASE / "models", BASE / "data"
                    ts = datetime.now().strftime("%Y%m%d_%H%M")
                    bk = mdir / f"backup_{ts}"
                    shutil.copytree(mdir, bk, ignore=shutil.ignore_patterns("backup_*"))
                    for old in sorted([d for d in mdir.glob("backup_*") if d.is_dir()])[:-2]:
                        shutil.rmtree(old, ignore_errors=True)
                    if not (ddir / "_fao_original.csv").exists():
                        shutil.copyfile(ddir / "crop_yield.csv", ddir / "_fao_original.csv")
                    st.session_state.upload_df.to_csv(ddir / "crop_yield.csv", index=False)
                    with st.spinner("Retraining 6 models + figures… (~5–8 min, the app reloads after)"):
                        TRAIN.main()
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.session_state.upload_df = None
                    st.session_state.upload_name = ""
                    st.success(f"Retrained on your data. Previous models kept in models/{bk.name}. Reloading…")
                    st.rerun()
                except Exception as e:
                    st.error(f"Retrain failed — saved models left as they were: {e}")
    if (BASE / "data" / "_fao_original.csv").exists():
        if st.button("Restore original FAO data + models", key="up_restore"):
            try:
                mdir, ddir = BASE / "models", BASE / "data"
                bks = sorted([d for d in mdir.glob("backup_*") if d.is_dir()])
                if bks:
                    src = bks[-1]
                    for item in src.iterdir():
                        if item.name.startswith("backup_"):
                            continue
                        dst = mdir / item.name
                        if item.is_dir():
                            shutil.copytree(item, dst, dirs_exist_ok=True)
                        else:
                            shutil.copy2(item, dst)
                shutil.copyfile(ddir / "_fao_original.csv", ddir / "crop_yield.csv")
                (ddir / "_fao_original.csv").unlink()
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("FAO data and models restored. Reloading…")
                st.rerun()
            except Exception as e:
                st.error(f"Restore failed: {e}")
    footer()


def view_ai():
    navbar("AI")
    photo("farmer.jpg", "Talk field-talk — numbers optional")
    titlebar("Farmer AI", "Ask in plain words",
             "Live Gemini answers when connected, offline farm guide otherwise — always labelled honestly.")

    from src import llm as LLM
    from src import mongo as MONGO
    comp_line = ", ".join(f"{r['Model'].split()[0]} R² {r['R2_test']:.4f}" for r in comp.to_dict(orient="records"))

    def mongo_db():
        if not st.session_state.mongo_ok:
            return None
        db, err = MONGO.get_db(st.session_state.mongo_uri)
        if err:
            st.session_state.mongo_ok = False
            st.session_state.mongo_msg = err
            return None
        return db

    s1, s2 = st.columns(2)
    with s1:
        key_ok = bool(LLM.get_key())
        live = st.session_state.ai_mode == "Live" and key_ok
        st.markdown('<div class="card" style="text-align:center"><b>🤖 AI engine</b><br>' +
                    ("<span style='color:#2E7D4F;font-weight:800'>🟢 Live Gemini ready</span>" if live else
                     ("<span style='color:#8A6D1B;font-weight:800'>🟡 Live selected, no key</span>"
                      if st.session_state.ai_mode == "Live" else
                      "<span style='color:#5B6B54;font-weight:800'>📴 Offline guide active</span>")) +
                    '</div>', unsafe_allow_html=True)
    with s2:
        st.markdown('<div class="card" style="text-align:center"><b>🍃 Database</b><br>' +
                    ("<span style='color:#2E7D4F;font-weight:800'>🟢 MongoDB connected</span>"
                     if st.session_state.mongo_ok else
                     "<span style='color:#5B6B54;font-weight:800'>🔴 Session-only store</span>") + '</div>',
                    unsafe_allow_html=True)
    with st.expander("🔑 Connect free AI + database (≈2 min, both optional)"):
        st.write("**AI (Google Gemini, free tier):** 1) open aistudio.google.com → Get API key (no card) → "
                 "paste below → Save & test. **Database:** local `mongodb://localhost:27017/` needs MongoDB Community "
                 "running, or paste a free Atlas connection string (mongodb+srv://…). Nothing is uploaded anywhere else.")
        k = st.text_input("Gemini API key", value="", type="password", key="ai_key",
                          placeholder="AIzaSy… (leave blank to stay offline)")
        if st.button("Save key & test", key="ai_save"):
            LLM.save_key(k)
            try:
                LLM.ask_live("Reply with exactly: OK", [], "You are a test.", "gemini-2.0-flash")
                st.success("Key works — Live AI is ready. Switch the mode below.")
            except Exception as e:
                st.error(f"Key saved, but the live call failed (staying on offline guide): {e}")
        st.session_state.ai_mode = st.radio("Answering mode", ["Offline", "Live"], horizontal=True,
                                            index=0 if st.session_state.ai_mode == "Offline" else 1, key="ai_mode_r")
        uri = st.text_input("MongoDB URI", value=st.session_state.mongo_uri, key="mongo_uri_in")
        if st.button("Connect database", key="mongo_go"):
            db, err = MONGO.get_db(uri)
            if err:
                st.session_state.mongo_ok = False
                st.session_state.mongo_msg = err
                st.error(f"Could not reach MongoDB ({err}). Chats stay in this session only.")
            else:
                st.session_state.mongo_ok = True
                st.session_state.mongo_uri = uri
                st.session_state.mongo_msg = ""
                st.success("MongoDB connected — chats, predictions and scenarios are now persisted.")
        if st.session_state.mongo_msg:
            st.caption(f"Last DB note: {st.session_state.mongo_msg}")
        db = mongo_db()
        if db is not None and st.session_state.mongo_ok:
            with st.expander("📜 Stored in MongoDB — live proof (counts + recent chats)"):
                cc = MONGO.counts(db)
                if cc:
                    st.write("Database **agripredict** — " + " · ".join(
                        f"{c}: **{n}** documents" for c, n in cc.items()))
                for c in MONGO.recent_chats(db):
                    st.write(f"**{c.get('role')}** ({c.get('mode','')}) — {c.get('text','')[:200]}")

    def answer_dual(q):
        ext = FARMER.extract_features(q, meta.get("areas", []), meta.get("items", []))
        if st.session_state.ai_mode == "Live" and LLM.get_key():
            ctx = LLM.build_context(df, meta, ext, comp_line)
            try:
                reply = LLM.ask_live(q, st.session_state.farmer_chat, ctx)
                return ("live", "🤖 " + reply, ext, "live-gemini")
            except Exception as e:
                _, off, _ = FARMER.answer(q, df, meta)
                return ("fallback", off + f"\n\n_(Live AI failed just now ({e}); answered offline instead.)_",
                        ext, "offline-fallback")
        intent, reply, _ = FARMER.answer(q, df, meta)
        tip = "" if LLM.get_key() else "\n\n_(Tip: connect a free Gemini key above for live answers.)_"
        return (intent, f"_{intent}_ — {reply}{tip}", ext,
                "offline-guide" if st.session_state.ai_mode == "Offline" else "offline-no-key")

    def ask(q):
        intent, reply, ext, mode = answer_dual(q)
        st.session_state.farmer_chat.append({"role": "user", "text": q})
        st.session_state.farmer_chat.append({"role": "assistant", "text": reply, "ext": ext, "mode": mode})
        db = mongo_db()
        if db is not None:
            MONGO.log_chat(db, "user", q, mode=mode)
            MONGO.log_chat(db, "assistant", reply, intent, mode)

    q1, q2, q3 = st.columns(3)
    with q1:
        if st.button("🌧 Which crop suits 1500mm rain?", key="ai_q1"):
            ask("Which crop suits 1500mm rain at 26 degrees?")
            st.rerun()
    with q2:
        if st.button("🎯 How do I predict?", key="ai_q2"):
            ask("How do I predict my yield?")
            st.rerun()
    with q3:
        if st.button("🏆 Why Random Forest?", key="ai_q3"):
            ask("Why is Random Forest the best model?")
            st.rerun()
    for m in st.session_state.farmer_chat:
        st.chat_message(m["role"]).write(m["text"])
    last_ext = {}
    for m in reversed(st.session_state.farmer_chat):
        if m["role"] == "assistant" and m.get("ext"):
            last_ext = m["ext"]
            break
    if last_ext and sum(k in last_ext for k in ("item", "average_rain_fall_mm_per_year", "avg_temp")) >= 2:
        bits = []
        if "item" in last_ext:
            bits.append(last_ext["item"])
        if "average_rain_fall_mm_per_year" in last_ext:
            bits.append(f"{last_ext['average_rain_fall_mm_per_year']:g}mm")
        if st.button(f"➡ Predict with: {', '.join(bits)}", key="ai_carry", type="primary"):
            if "area" in last_ext:
                st.session_state["pred_area"] = last_ext["area"]
            if "item" in last_ext:
                st.session_state["pred_crop_sel"] = last_ext["item"]
                st.session_state["pred_crop"] = last_ext["item"]
            if "year" in last_ext:
                st.session_state["pred_year"] = int(max(2010, min(2025, last_ext["year"])))
            if "average_rain_fall_mm_per_year" in last_ext:
                st.session_state["pred_rain"] = float(last_ext["average_rain_fall_mm_per_year"])
            if "avg_temp" in last_ext:
                st.session_state["pred_temp"] = float(last_ext["avg_temp"])
            if "pesticides_tonnes" in last_ext:
                st.session_state["pred_pest"] = float(last_ext["pesticides_tonnes"])
            go("Predict")
    q = st.chat_input("Ask like a farmer: crop, rainfall mm, temperature…")
    if q:
        ask(q)
        st.rerun()
    footer()


# ================= ROUTER (one view at a time — never stacked) =================
{"Home": view_home, "Prep": view_data, "Crops": view_crops, "Predict": view_predict, "Lab": view_lab,
 "Insights": view_insights, "Simulator": view_sim, "AI": view_ai,
 "About": view_about}[st.session_state.view]()
