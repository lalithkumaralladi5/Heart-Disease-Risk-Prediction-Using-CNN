"""
train_pytorch.py
---------------
Main entry-point for training the Heart Disease CNN using PyTorch.

Run:
    python train_pytorch.py

What this script does:
    1. Loads and cleans the dataset
    2. Runs the full preprocessing pipeline
    3. Builds the CNN model
    4. Trains with EarlyStopping
    5. Saves the trained model to models/cnn_heart_model_pytorch.pth
    6. Plots training history
"""

import sys
import os

# ── Make sure sibling packages are importable ────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from utils.preprocessing import load_raw_data, clean_data, preprocess, FEATURE_COLS
from utils.visualization  import plot_training_history, plot_feature_distribution, plot_correlation_heatmap
from models.cnn_model_pytorch import build_cnn, save_model, MODEL_PATH

# ── Reproducibility seed ─────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Hyper-parameters ─────────────────────────────────────────────────────────
EPOCHS        = 100       # upper bound — EarlyStopping will cut this short
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.15   # 15 % of training set used for in-training validation

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


def main():
    print("=" * 60)
    print("  Heart Disease Risk Prediction — Training Pipeline (PyTorch)")
    print("=" * 60)

    # ── 1. Load and preprocess data ───────────────────────────────────────────────
    print("\n[1] Loading and preprocessing data...")
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    processed_data = preprocess(clean_df)
    X_train = processed_data["X_train"]
    X_test = processed_data["X_test"]
    y_train = processed_data["y_train"]
    y_test = processed_data["y_test"]

    # Split training data into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VALIDATION_SPLIT, random_state=SEED, stratify=y_train
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train.reshape(X_train.shape[0], X_train.shape[1])).to(device)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val_tensor = torch.FloatTensor(X_val.reshape(X_val.shape[0], X_val.shape[1])).to(device)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    X_test_tensor = torch.FloatTensor(X_test.reshape(X_test.shape[0], X_test.shape[1])).to(device)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(device)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ── 2. Build model ───────────────────────────────────────────────────────────
    print("\n[2] Building CNN model...")
    model, criterion, optimizer = build_cnn(input_size=X_train.shape[1], learning_rate=LEARNING_RATE)
    model = model.to(device)

    print(f"Model architecture:")
    print(model)

    # ── 3. Training loop ────────────────────────────────────────────────────────
    print("\n[3] Starting training...")
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == batch_y).sum().item()
            train_total += batch_y.size(0)

        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_y.size(0)

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

        # Early stopping
        if early_stopping(val_loss, model):
            print(f"Early stopping at epoch {epoch+1}")
            break

    # ── 4. Save model ───────────────────────────────────────────────────────────
    print("\n[4] Saving model...")
    save_model(model, MODEL_PATH)

    # ── 5. Evaluate on test set ─────────────────────────────────────────────────
    print("\n[5] Evaluating on test set...")
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            predicted = (outputs > 0.5).float()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    test_accuracy = accuracy_score(all_labels, all_predictions)
    test_precision = precision_score(all_labels, all_predictions, zero_division=0)
    test_recall = recall_score(all_labels, all_predictions, zero_division=0)
    test_auc = roc_auc_score(all_labels, all_predictions)

    print(f"Test Accuracy:  {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print(f"Test AUC:       {test_auc:.4f}")

    # ── 6. Plot training history ─────────────────────────────────────────────────
    print("\n[6] Plotting training history...")
    # Skip plotting to avoid visualization issues
    print("Skipping training history plot...")

    # ── 7. Plot feature distributions ────────────────────────────────────────────
    print("\n[7] Plotting feature distributions...")
    # Skip plotting to avoid visualization issues
    print("Skipping feature distribution plot...")

    # ── 8. Plot correlation heatmap ───────────────────────────────────────────────
    print("\n[8] Plotting correlation heatmap...")
    # Skip plotting to avoid visualization issues
    print("Skipping correlation heatmap plot...")

    print("\n" + "=" * 60)
    print("  Training completed successfully!")
    print(f"  Model saved to: {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
