"""EDA figure generation (saved to outputs/figures, also rendered live in app)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

sns.set(style="whitegrid")

def generate_eda_figures(df, outdir: Path) -> list:
    outdir.mkdir(parents=True, exist_ok=True)
    made = []
    num = ["year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp", "value_hg_ha"]

    def save(name):
        plt.tight_layout(); plt.savefig(outdir / name); plt.close(); made.append(name)

    plt.figure(figsize=(8, 5)); sns.histplot(df["value_hg_ha"], bins=50, kde=True)
    plt.title("Yield distribution (hg/ha)"); plt.xlabel("Yield (hg/ha)"); save("01_yield_dist.png")

    plt.figure(figsize=(7, 5)); sns.heatmap(df[num].corr(), annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Correlation heatmap (numeric)"); save("02_corr.png")

    plt.figure(figsize=(9, 5))
    order = df.groupby("item")["value_hg_ha"].mean().sort_values(ascending=False).index
    sns.barplot(data=df, x="item", y="value_hg_ha", estimator=np.mean, errorbar=None, order=order)
    plt.xticks(rotation=30, ha="right"); plt.title("Mean yield by crop"); plt.ylabel("Mean hg/ha"); save("03_crop_mean.png")

    plt.figure(figsize=(9, 5))
    yr = df.groupby("year")["value_hg_ha"].mean()
    plt.plot(yr.index, yr.values, marker="o"); plt.title("Mean yield by year (1990-2013)")
    plt.xlabel("Year"); plt.ylabel("Mean hg/ha"); save("04_year_trend.png")

    sub = df.sample(min(3000, len(df)), random_state=42)
    for x, t, fn in [("average_rain_fall_mm_per_year", "Rainfall vs yield", "05_rain.png"),
                     ("pesticides_tonnes", "Pesticides vs yield", "06_pest.png"),
                     ("avg_temp", "Temperature vs yield", "07_temp.png")]:
        plt.figure(figsize=(7, 5)); sns.scatterplot(data=sub, x=x, y="value_hg_ha", alpha=0.35)
        plt.title(t); save(fn)

    plt.figure(figsize=(8, 6))
    top = df.groupby("area")["value_hg_ha"].mean().sort_values(ascending=False).head(15)
    sns.barplot(x=top.values, y=top.index, orient="h"); plt.title("Top 15 areas by mean yield"); save("08_top_areas.png")

    plt.figure(figsize=(9, 5))
    crops = df.groupby("item")["value_hg_ha"].mean().sort_values(ascending=False).head(6).index.tolist()
    sns.boxplot(data=df[df["item"].isin(crops)], x="item", y="value_hg_ha", order=crops)
    plt.xticks(rotation=25, ha="right"); plt.title("Yield spread — top 6 crops (box plot)"); save("09_box_crops.png")
    return made
