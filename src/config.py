"""Central configuration — single source of truth for paths, features, units."""
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA_PATH = BASE / "data" / "crop_yield.csv"
MODELS_DIR = BASE / "models"
FIG_DIR = BASE / "outputs" / "figures"
ASSETS_DIR = BASE / "assets"

TARGET = "value_hg_ha"  # FAO yield in hectograms per hectare
HG_PER_TONNE = 10_000.0  # 1 tonne/ha = 10,000 hg/ha

NUM_FEATURES = ["year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
CAT_FEATURES = ["area", "item"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

SEED = 42
TEST_SIZE = 0.2

# Honest feature-availability statement (do NOT invent these columns)
AVAILABLE = {
    "area": "Country/region (categorical, 101 values)",
    "item": "Crop type (categorical, 10 values)",
    "year": "Calendar year 1990-2013 (numeric)",
    "average_rain_fall_mm_per_year": "Mean annual rainfall mm/year (numeric)",
    "pesticides_tonnes": "Total pesticide use tonnes/year (numeric)",
    "avg_temp": "Mean annual temperature C (numeric)",
    "value_hg_ha": "TARGET: crop yield, hg/ha (numeric, continuous)",
}
UNAVAILABLE = [
    "nitrogen (N)", "phosphorus (P)", "potassium (K)", "soil pH",
    "humidity", "soil type", "season", "cultivation area (ha)",
    "fertilizer usage",
]
DATASET_SOURCE = "FAO via Zenodo DOI 10.5281/zenodo.18761663 (mirrors Kaggle patelris/crop-yield-prediction-dataset), CC-BY-4.0"
