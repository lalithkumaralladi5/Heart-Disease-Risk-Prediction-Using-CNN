"""
models/cnn_model_pytorch.py
--------------------
Defines the 1-D CNN architecture for Heart Disease Risk Prediction using PyTorch.

Why CNN for tabular data?
──────────────────────────
• A 1-D Convolutional layer slides a small kernel across the feature vector.
• This lets the model learn *local feature interactions* automatically —
  e.g., the relationship between (age, sex) or (thalach, exang, oldpeak)
  without hand-crafting interaction terms.
• Multiple Conv layers stack to capture increasingly complex combinations.
• Dropout and BatchNorm prevent overfitting on the small UCI dataset.
• Final Linear → Sigmoid outputs a probability in [0, 1].
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "cnn_heart_model_pytorch.pth")


class HeartDiseaseCNN(nn.Module):
    def __init__(self, input_size=13):
        super(HeartDiseaseCNN, self).__init__()
        
        # ── Block 1: shallow patterns ─────────────────────────────────────────
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        
        # ── Block 2: deeper patterns ──────────────────────────────────────────
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        
        # ── Pooling: compress to a single vector per filter ───────────────────
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # ── Fully-connected head ──────────────────────────────────────────────
        self.fc1 = nn.Linear(64, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        
        # ── Output ────────────────────────────────────────────────────────────
        self.output = nn.Linear(64, 1)
        
    def forward(self, x):
        # Input shape: (batch_size, 13) -> (batch_size, 1, 13)
        x = x.unsqueeze(1)
        
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        # Global Average Pooling
        x = self.global_avg_pool(x)  # (batch_size, 64, 1)
        x = x.squeeze(-1)  # (batch_size, 64)
        
        # Fully-connected layers
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.dropout2(x)
        
        # Output
        x = self.output(x)
        x = torch.sigmoid(x)
        
        return x


def build_cnn(input_size=13, learning_rate=1e-3):
    """
    Builds and compiles a 1-D CNN for binary classification.

    Parameters
    ----------
    input_size     : number of features — default 13
    learning_rate  : Adam learning rate

    Returns
    -------
    model, criterion, optimizer
    """
    model = HeartDiseaseCNN(input_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    return model, criterion, optimizer


def load_model(path: str = MODEL_PATH):
    """
    Loads a saved PyTorch model from disk.

    Parameters
    ----------
    path : path to .pth file

    Returns
    -------
    Loaded PyTorch model
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved model found at '{path}'.\n"
            "Train first:  python train.py"
        )
    print(f"[Model] Loading from {path}")
    
    model = HeartDiseaseCNN()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    
    return model


def save_model(model, path: str = MODEL_PATH):
    """
    Saves a PyTorch model to disk.

    Parameters
    ----------
    model : PyTorch model to save
    path  : path to save .pth file
    """
    torch.save(model.state_dict(), path)
    print(f"[Model] Saved to {path}")
