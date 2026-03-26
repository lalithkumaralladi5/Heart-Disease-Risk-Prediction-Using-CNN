"""
download_data.py
----------------
Downloads the UCI Cleveland Heart Disease dataset and saves it locally.
Run this script ONCE before training: python data/download_data.py
"""

import pandas as pd
import os

# ── Column names as defined by UCI ─────────────────────────────────────────
COLUMN_NAMES = [
    "age",        # Age in years
    "sex",        # 1 = male; 0 = female
    "cp",         # Chest pain type (0-3)
    "trestbps",   # Resting blood pressure (mm Hg)
    "chol",       # Serum cholesterol (mg/dl)
    "fbs",        # Fasting blood sugar > 120 mg/dl (1=true; 0=false)
    "restecg",    # Resting ECG results (0-2)
    "thalach",    # Maximum heart rate achieved
    "exang",      # Exercise induced angina (1=yes; 0=no)
    "oldpeak",    # ST depression induced by exercise
    "slope",      # Slope of peak exercise ST segment (0-2)
    "ca",         # Number of major vessels (0-3) colored by fluoroscopy
    "thal",       # Thalassemia type (1=normal; 2=fixed defect; 3=reversible)
    "target"      # 0 = No Disease, 1-4 = Disease (we binarise to 0/1)
]

# UCI Cleveland dataset URL
URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

SAVE_PATH = os.path.join(os.path.dirname(__file__), "heart_disease.csv")


def download():
    print("⬇  Downloading UCI Cleveland Heart Disease dataset …")
    try:
        df = pd.read_csv(URL, header=None, names=COLUMN_NAMES, na_values="?")
    except Exception as e:
        print(f"❌  Download failed: {e}")
        print("   Generating a synthetic fallback dataset instead …")
        df = _synthetic_fallback()

    # Binarise target: 0 = no disease, 1 = disease
    df["target"] = (df["target"] > 0).astype(int)

    df.to_csv(SAVE_PATH, index=False)
    print(f"✅  Saved to {SAVE_PATH}  ({len(df)} rows, {df['target'].sum()} positive cases)")


def _synthetic_fallback():
    """
    Creates a small synthetic dataset that mirrors the UCI schema.
    Useful when the UCI URL is unavailable (offline / firewall).
    """
    import numpy as np
    rng = np.random.RandomState(42)
    n = 303

    df = pd.DataFrame({
        "age":      rng.randint(29, 78, n).astype(float),
        "sex":      rng.randint(0, 2, n).astype(float),
        "cp":       rng.randint(0, 4, n).astype(float),
        "trestbps": rng.randint(94, 200, n).astype(float),
        "chol":     rng.randint(126, 565, n).astype(float),
        "fbs":      rng.randint(0, 2, n).astype(float),
        "restecg":  rng.randint(0, 3, n).astype(float),
        "thalach":  rng.randint(71, 202, n).astype(float),
        "exang":    rng.randint(0, 2, n).astype(float),
        "oldpeak":  np.round(rng.uniform(0, 6.2, n), 1),
        "slope":    rng.randint(0, 3, n).astype(float),
        "ca":       rng.randint(0, 4, n).astype(float),
        "thal":     rng.choice([1.0, 2.0, 3.0], n),
        "target":   rng.randint(0, 5, n).astype(float),
    })
    # Introduce ~5 missing values to demonstrate cleaning
    for col in ["ca", "thal"]:
        idx = rng.choice(n, 2, replace=False)
        df.loc[idx, col] = float("nan")
    return df


if __name__ == "__main__":
    download()
