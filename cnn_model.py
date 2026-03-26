"""
models/cnn_model.py
--------------------
Defines the 1-D CNN architecture for Heart Disease Risk Prediction.

Why CNN for tabular data?
─────────────────────────
• A 1-D Convolutional layer slides a small kernel across the feature vector.
• This lets the model learn *local feature interactions* automatically —
  e.g., the relationship between (age, sex) or (thalach, exang, oldpeak)
  without hand-crafting interaction terms.
• Multiple Conv layers stack to capture increasingly complex combinations.
• Dropout and BatchNorm prevent overfitting on the small UCI dataset.
• Final Dense → Sigmoid outputs a probability in [0, 1].
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "cnn_heart_model.keras")


def build_cnn(input_shape=(13, 1), learning_rate: float = 1e-3) -> tf.keras.Model:
    """
    Builds and compiles a 1-D CNN for binary classification.

    Architecture Overview
    ─────────────────────
    Input (13, 1)
        │
    Conv1D(32, kernel=3) ──► BatchNorm ──► ReLU
        │
    Conv1D(64, kernel=3) ──► BatchNorm ──► ReLU
        │
    GlobalAveragePooling1D              (replaces MaxPool to keep all info)
        │
    Dense(128) ──► BatchNorm ──► ReLU ──► Dropout(0.4)
        │
    Dense(64)  ──► ReLU ──► Dropout(0.3)
        │
    Dense(1)   ──► Sigmoid              (probability of heart disease)

    Parameters
    ----------
    input_shape    : (features, channels) — default (13, 1)
    learning_rate  : Adam learning rate

    Returns
    -------
    Compiled tf.keras.Model
    """
    inputs = tf.keras.Input(shape=input_shape, name="features")

    # ── Block 1: shallow patterns ─────────────────────────────────────────
    x = layers.Conv1D(
        filters=32,
        kernel_size=3,
        padding="same",                 # keeps the same length after conv
        kernel_regularizer=regularizers.l2(1e-4),
        name="conv1",
    )(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)

    # ── Block 2: deeper patterns ──────────────────────────────────────────
    x = layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4),
        name="conv2",
    )(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)

    # ── Pooling: compress to a single vector per filter ───────────────────
    x = layers.GlobalAveragePooling1D(name="gap")(x)

    # ── Fully-connected head ──────────────────────────────────────────────
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4), name="fc1")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.Dropout(0.4, name="dropout1")(x)

    x = layers.Dense(64, activation="relu", name="fc2")(x)
    x = layers.Dropout(0.3, name="dropout2")(x)

    # ── Output ────────────────────────────────────────────────────────────
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="HeartDiseaseCNN")

    # ── Compile ───────────────────────────────────────────────────────────
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def load_model(path: str = MODEL_PATH) -> tf.keras.Model:
    """
    Loads a saved Keras model from disk.

    Parameters
    ----------
    path : path to .keras file

    Returns
    -------
    Loaded tf.keras.Model
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved model found at '{path}'.\n"
            "Train first:  python train.py"
        )
    print(f"[Model] Loading from {path}")
    return tf.keras.models.load_model(path)
