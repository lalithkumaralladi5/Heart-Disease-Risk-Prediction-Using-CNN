"""
utils/visualization.py
-----------------------
All plotting helpers.  Kept separate so training / evaluation scripts
stay clean and these can be imported by both the notebook and the app.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def plot_training_history(history, save: bool = True):
    """
    Plots training vs validation accuracy AND loss side-by-side.

    Parameters
    ----------
    history : Keras History object returned by model.fit()
    save    : if True, saves the figure to models/training_history.png
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CNN Training History", fontsize=16, fontweight="bold")

    # ── Accuracy ─────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(history.history["accuracy"],     label="Train Accuracy",  linewidth=2, color="#2196F3")
    ax.plot(history.history["val_accuracy"], label="Val Accuracy",    linewidth=2, color="#FF5722", linestyle="--")
    ax.set_title("Accuracy over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Loss ─────────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(history.history["loss"],     label="Train Loss",  linewidth=2, color="#4CAF50")
    ax.plot(history.history["val_loss"], label="Val Loss",    linewidth=2, color="#F44336", linestyle="--")
    ax.set_title("Loss over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "training_history.png")
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Training history saved → {path}")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, save: bool = True):
    """
    Plots a styled confusion matrix heatmap.

    Parameters
    ----------
    y_true : array of ground-truth labels
    y_pred : array of predicted binary labels (0 or 1)
    save   : if True, saves to models/confusion_matrix.png
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Low Risk\n(No Disease)", "High Risk\n(Disease)"]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
        annot_kws={"size": 16, "weight": "bold"},
    )
    ax.set_title("Confusion Matrix", fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)

    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Confusion matrix saved → {path}")
    plt.show()


def plot_feature_distribution(df, feature_cols, save: bool = True):
    """
    Plots histograms for every feature, colour-coded by target class.

    Parameters
    ----------
    df           : cleaned pandas DataFrame
    feature_cols : list of feature column names
    save         : save the figure to models/feature_distribution.png
    """
    n = len(feature_cols)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3.5))
    axes = axes.flatten()

    palette = {0: "#2196F3", 1: "#F44336"}
    labels  = {0: "Low Risk", 1: "High Risk"}

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        for target_val in [0, 1]:
            subset = df[df["target"] == target_val][col]
            ax.hist(
                subset,
                bins=20,
                alpha=0.6,
                color=palette[target_val],
                label=labels[target_val],
                edgecolor="white",
            )
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel("")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by Heart Disease Status", fontsize=15, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, "feature_distribution.png")
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Feature distributions saved → {path}")
    plt.show()


def plot_correlation_heatmap(df, feature_cols, save: bool = True):
    """
    Pearson correlation heatmap for all features + target.
    """
    cols = feature_cols + ["target"]
    corr = df[cols].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))  # upper triangle mask

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=15, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Correlation heatmap saved → {path}")
    plt.show()
