"""
evaluate.py
-----------
Loads the trained CNN and produces a full evaluation report.

Run:
    python evaluate.py

Outputs:
    • Classification report (accuracy, precision, recall, F1)
    • Confusion matrix  →  models/confusion_matrix.png
    • ROC-AUC curve     →  models/roc_curve.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from utils.preprocessing import load_raw_data, clean_data, preprocess, FEATURE_COLS
from utils.visualization  import plot_confusion_matrix
from models.cnn_model     import load_model, MODEL_PATH

SEED = 42


def plot_roc_curve(y_true, y_prob, save: bool = True):
    """
    Plots the ROC curve and annotates AUC.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, linewidth=2.5, color="#2196F3", label=f"CNN (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2196F3")
    ax.set_title("ROC Curve — Heart Disease Prediction", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join("models", "roc_curve.png")
        os.makedirs("models", exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Plot] ROC curve saved → {path}")
    plt.show()


def main():
    print("=" * 60)
    print("  Heart Disease Risk Prediction — Evaluation")
    print("=" * 60)

    # ── Load & preprocess data ────────────────────────────────────────────────
    df   = load_raw_data()
    df   = clean_data(df)
    data = preprocess(df, test_size=0.20, random_state=SEED, fit_scaler=False)

    X_test  = data["X_test"]
    y_test  = data["y_test"]

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(MODEL_PATH)

    # ── Raw probabilities ─────────────────────────────────────────────────────
    y_prob = model.predict(X_test, verbose=0).flatten()   # shape: (N,)
    y_pred = (y_prob >= 0.5).astype(int)                  # threshold at 0.5

    # ── Classification Report ─────────────────────────────────────────────────
    print("\n── Classification Report ─────────────────────────────")
    print(classification_report(
        y_test, y_pred,
        target_names=["Low Risk (0)", "High Risk (1)"],
        digits=4,
    ))

    # ── AUC ───────────────────────────────────────────────────────────────────
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score : {auc:.4f}")

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    print("\n── Confusion Matrix ──────────────────────────────────")
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"  True Negatives  (TN): {tn}")
    print(f"  False Positives (FP): {fp}  ← predicted disease, actually healthy")
    print(f"  False Negatives (FN): {fn}  ← missed disease cases (critical!)")
    print(f"  True Positives  (TP): {tp}")
    plot_confusion_matrix(y_test, y_pred, save=True)

    # ── ROC Curve ─────────────────────────────────────────────────────────────
    plot_roc_curve(y_test, y_prob, save=True)

    print("\n✅  Evaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
