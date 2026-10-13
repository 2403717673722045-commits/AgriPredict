# AgriPredict — Grow Smarter with Machine Learning

Image-led single-app dashboard (session-state views replace the page in the same tab —
no new tabs, no stacked sections, no sidebar). Full-bleed farm photography behind content,
dark-green overlays, gold accents. 17 verified local photos (Unsplash + Wikimedia Commons).
FAO primary dataset (Zenodo DOI 10.5281/zenodo.18761663, CC-BY-4.0), 25,932 rows, 1990–2013.
Target `value_hg_ha` in **hg/ha** (÷10,000 = tonnes/ha). Only the 10 dataset crops are shown —
no sugarcane/cotton tiles. No soil values invented.

Views: Home (hero + crop of day + Yield Challenge game) · Crop Explorer (10 photo tiles → detail + predict shortcuts) · Yield Prediction ·
ML Model Lab (6 models, selectable detail) · Agricultural Insights (filters redraw charts + animated year race) ·
Scenario Simulator (A/B + playground) · About Project (+ report downloads).
Drop your 5 chat photos into assets/user/ as rain.jpg, soilmeter.jpg, farmersoil.jpg,
tractormais.jpg, workers.jpg — backgrounds upgrade automatically (see assets/user/README.txt).

Six regressors, one fair test (80/20 seed 42, preprocessing fit on train only):
Linear 0.7497 · Poly 0.7496 · Tree 0.9757 · **Forest 0.9840** · KNN 0.9824 · GB 0.8576 (test R²).
Unseen-years check (train ≤2005 → test 2006–2013): Forest 0.9395 · Linear 0.7207.

## Structure
```
AGRIPREDICT/
  app.py (9 views + router)  train.py  tune.py  requirements.txt  README.md
  data/crop_yield.csv  (upload here via Data Prep; _fao_original.csv = restore point)
  src/config.py  data_loader.py  preprocessing.py  models.py  evaluate.py
     predict.py  eda.py  assistant.py  farmer.py  studio.py  ui.py (theme+nav+shared)
  models/best_pipeline.pkl  pipelines/*.pkl (6)  metadata.json (+temporal_check)
     metrics.csv  test_predictions.csv  degree_sweep.json  feature_importance.csv
     tuning.json (after py tune.py)  backup_*/ (auto model backups)
  outputs/figures/*.png (20)
  assets/*.jpg (11)  assets/crops/*.jpg (6)  assets/user/ (your photos)  ATTRIBUTION.md
  reports/  _backup_*/ (reference only — do not demo)
```

## Windows run (CMD)
```cmd
cd "E:\MSC - 3rd yr\AGRIPREDICT"
py -m pip install -r requirements.txt
py train.py
py tune.py
py -m streamlit run app.py
```
Open http://localhost:8501. Views: Home · Data Prep (upload + preprocessing + EDA + optional retrain) ·
Crops · Predict · Lab (6 models + GridSearchCV tuning table) · Insights · Simulator · Farmer AI (live/offline chat) ·
About.
`tune.py` ≈ 10 min (GridSearchCV, sequential — this PC's parallel backend is broken); retrain ≈ 5–8 min.

## Live AI + MongoDB (both optional, graceful offline fallbacks)
- **AI:** free Gemini key from aistudio.google.com → Farmer AI view → “Connect free AI” → Save & test.
  Without a key (or on any API failure) the offline farm guide answers instead, labelled as such.
  Model answers only from retrieved project facts (dataset stats, real scores, your described conditions).
- **MongoDB:** local `mongodb://localhost:27017/` (MongoDB Community running) or a free Atlas
  `mongodb+srv://…` string → Farmer AI view → Connect. Chats, predictions and scenario saves persist
  to `agripredict.{chats,predictions,scenarios}`; a local MongoDB was detected on this machine.
  DB unreachable → everything keeps working on session logs, and the UI says so.
- Secrets: key lives in `.env` (git-ignored); never shown back after saving.

## Lab demo script (6 minutes)
1. Home hero → buttons navigate in-tab; tiles jump to Explorer with crop selected.
2. Data Prep: schema validation + 115-feature preprocessing demo + live EDA.
3. Predict → ~32,316 hg/ha for India/Wheat/2020 (pesticides 46,000 t) + history bar.
4. Lab: model tiles switch detail; tuning table (KNN k=3 ≈ RF); unseen-years card.
5. Farmer AI: “which crop suits 1500mm rain?” → top-3 from real records → carry to predictor.
Viva line: "Best by measurement — R² 0.984, matching CV, 0.94 on unseen years; tuning confirmed defaults."
