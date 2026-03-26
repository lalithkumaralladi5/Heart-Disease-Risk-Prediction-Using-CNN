"""
utils/preprocessing.py
-----------------------
All data-loading, cleaning, and transformation logic lives here.
Keeping preprocessing separate from training makes the code reusable
and easier to test.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import joblib
import os

# ── Paths ───────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "heart_disease.csv")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]
TARGET_COL = "target"


# ── Why CNN for tabular data? ────────────────────────────────────────────────
# Standard CNNs excel at detecting *local patterns* in a sequence.
# By treating each feature as a time-step (shape: [samples, 13, 1])
# a 1-D CNN can learn interactions between *neighbouring* features —
# e.g. age + sex together, or thalach + exang + oldpeak together.
# This gives the model an inductive bias that a plain MLP lacks.
# ────────────────────────────────────────────────────────────────────────────


def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Loads the CSV dataset.

    Parameters
    ----------
    path : str
        Path to heart_disease.csv

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Run:  python data/download_data.py"
        )
    df = pd.read_csv(path)
    print(f"[Data] Loaded {len(df)} rows × {len(df.columns)} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values and removes duplicates.

    Strategy
    --------
    • Numeric NaNs → replaced by column *median* (robust to outliers).
    • Duplicate rows → dropped.
    """
    print(f"[Data] Missing values before cleaning:\n{df.isnull().sum()}\n")

    # Impute missing values with column median
    imputer = SimpleImputer(strategy="median")
    df[FEATURE_COLS] = imputer.fit_transform(df[FEATURE_COLS])

    # Drop duplicate rows
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[Data] Dropped {before - len(df)} duplicate rows")
    print(f"[Data] Target distribution:\n{df[TARGET_COL].value_counts()}\n")
    return df


def preprocess(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    fit_scaler: bool = True,
) -> dict:
    """
    Full preprocessing pipeline:
        1. Separate features / labels
        2. Normalise features to [0, 1]
        3. Reshape to 3-D for CNN  →  (samples, 13, 1)
        4. Train / test split

    Parameters
    ----------
    df           : cleaned DataFrame
    test_size    : fraction used for testing
    random_state : reproducibility seed
    fit_scaler   : if True, fit a new MinMaxScaler and save it;
                   if False, load a pre-fitted scaler (for inference)

    Returns
    -------
    dict with keys:
        X_train, X_test, y_train, y_test, scaler
    """
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    # ── Normalisation ────────────────────────────────────────────────────────
    if fit_scaler:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        print(f"[Data] Scaler saved → {SCALER_PATH}")
    else:
        scaler = joblib.load(SCALER_PATH)
        X = scaler.transform(X)
        print(f"[Data] Loaded pre-fitted scaler from {SCALER_PATH}")

    # ── Reshape for CNN  (samples, features, 1) ──────────────────────────────
    # Think of the 13 features as 13 "time steps" with 1 channel each.
    # A 1-D Conv layer slides a window over these features to detect patterns.
    X = X.reshape(X.shape[0], X.shape[1], 1)   # → (N, 13, 1)

    # ── Train / Test Split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[Data] Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"[Data] Train positives: {y_train.sum():.0f} / {len(y_train)}")
    print(f"[Data] Test  positives: {y_test.sum():.0f} / {len(y_test)}\n")

    return {
        "X_train": X_train,
        "X_test":  X_test,
        "y_train": y_train,
        "y_test":  y_test,
        "scaler":  scaler,
    }


def preprocess_single(raw_values: list) -> np.ndarray:
    """
    Preprocesses a single patient record for real-time inference.

    Parameters
    ----------
    raw_values : list of 13 raw feature values (same order as FEATURE_COLS)

    Returns
    -------
    np.ndarray of shape (1, 13, 1)
    """
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            "Scaler not found. Train the model first: python train.py"
        )
    scaler = joblib.load(SCALER_PATH)
    x = np.array(raw_values, dtype=np.float32).reshape(1, -1)
    x = scaler.transform(x)
    x = x.reshape(1, 13, 1)
    return x
