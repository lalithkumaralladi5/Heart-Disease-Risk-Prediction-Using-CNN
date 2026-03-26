"""
train.py
--------
Main entry-point for training the Heart Disease CNN.

Run:
    python train.py

What this script does:
    1. Loads and cleans the dataset
    2. Runs the full preprocessing pipeline
    3. Builds the CNN model
    4. Trains with EarlyStopping + ReduceLROnPlateau callbacks
    5. Saves the trained model to models/cnn_heart_model.keras
    6. Plots training history
"""

import sys
import os

# ── Make sure sibling packages are importable ────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)

from utils.preprocessing import load_raw_data, clean_data, preprocess, FEATURE_COLS
from utils.visualization  import plot_training_history, plot_feature_distribution, plot_correlation_heatmap
from models.cnn_model     import build_cnn, MODEL_PATH

# ── Reproducibility seed ─────────────────────────────────────────────────────
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ── Hyper-parameters ─────────────────────────────────────────────────────────
EPOCHS        = 100       # upper bound — EarlyStopping will cut this short
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.15   # 15 % of training set used for in-training validation


def main():
    print("=" * 60)
    print("  Heart Disease Risk Prediction — Training Pipeline")
    print("=" * 60)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    df = load_raw_data()

    # ── 2. Explore (optional plots) ───────────────────────────────────────────
    print("\n[Info] Generating EDA plots …")
    plot_feature_distribution(df, FEATURE_COLS, save=True)
    plot_correlation_heatmap(df, FEATURE_COLS, save=True)

    # ── 3. Clean ─────────────────────────────────────────────────────────────
    df = clean_data(df)

    # ── 4. Preprocess ────────────────────────────────────────────────────────
    data = preprocess(df, test_size=0.20, random_state=SEED, fit_scaler=True)
    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]

    # ── 5. Build model ───────────────────────────────────────────────────────
    model = build_cnn(input_shape=(X_train.shape[1], 1), learning_rate=LEARNING_RATE)
    model.summary()

    # ── 6. Callbacks ─────────────────────────────────────────────────────────
    callbacks = [
        # Stop training when val_loss stops improving for 15 epochs
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        # Halve LR when val_loss plateaus for 7 epochs
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
        # Always keep a checkpoint of the best model weights
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # ── 7. Train ─────────────────────────────────────────────────────────────
    print("\n[Train] Starting training …\n")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1,
    )

    # ── 8. Plot history ───────────────────────────────────────────────────────
    plot_training_history(history, save=True)

    # ── 9. Quick test-set evaluation ──────────────────────────────────────────
    print("\n[Eval] Evaluating on held-out test set …")
    results = model.evaluate(X_test, y_test, verbose=0)
    metric_names = model.metrics_names
    for name, val in zip(metric_names, results):
        print(f"       {name:12s}: {val:.4f}")

    print(f"\n✅  Model saved → {MODEL_PATH}")
    print("    Next step: python evaluate.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
