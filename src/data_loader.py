"""Data loading + inspection (no preprocessing here)."""
import pandas as pd
from .config import DATA_PATH, TARGET

def load_data(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def inspect(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "target_desc": df[TARGET].describe().to_dict(),
        "areas": int(df["area"].nunique()),
        "crops": sorted(df["item"].unique().tolist()),
        "year_range": (int(df["year"].min()), int(df["year"].max())),
    }
