# ❤️ Heart Disease Risk Prediction using CNN (Deep Learning)

A complete, industry-standard deep learning project that predicts
whether a patient is at **High Risk** or **Low Risk** of heart disease
using a **1-D Convolutional Neural Network** (CNN) built with TensorFlow / Keras.

---

## 📁 Project Structure

```
heart_disease_cnn/
├── data/
│   ├── download_data.py        ← Downloads UCI dataset (run once)
│   └── heart_disease.csv       ← Generated after download
│
├── models/
│   ├── cnn_model.py            ← CNN architecture definition
│   ├── cnn_heart_model.keras   ← Saved model (generated after training)
│   ├── scaler.pkl              ← Saved MinMaxScaler
│   ├── training_history.png    ← Loss / Accuracy plots
│   ├── confusion_matrix.png    ← Confusion matrix heatmap
│   └── roc_curve.png           ← ROC-AUC curve
│
├── notebooks/
│   └── eda.ipynb               ← Exploratory Data Analysis notebook
│
├── app/
│   └── streamlit_app.py        ← Streamlit web UI
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py        ← Data cleaning & transformation
│   └── visualization.py        ← All plotting helpers
│
├── train.py                    ← Main training script
├── evaluate.py                 ← Evaluation & metrics script
├── requirements.txt
└── README.md
```

---

## 🧠 Why CNN for Tabular Data?

A 1-D Convolutional Neural Network treats the **13 clinical features as a
sequence** (like a short time series).

| Benefit | Explanation |
|---|---|
| **Local patterns** | A Conv kernel learns interactions between neighbouring features automatically (e.g., age + sex, or thalach + exang + oldpeak). |
| **Parameter efficiency** | Fewer weights than a fully-connected network of the same capacity. |
| **Translation invariance** | Patterns are detected wherever they appear in the feature vector. |
| **Stacking** | Multiple Conv layers capture increasingly complex feature combinations. |

---

## ⚙️ Tech Stack

| Component | Library |
|---|---|
| Deep Learning | TensorFlow 2.15 / Keras |
| Data Science | NumPy, Pandas, Scikit-learn |
| Visualisation | Matplotlib, Seaborn |
| Web UI | Streamlit |
| Serialisation | joblib |

---

## 🚀 How to Run (Step-by-Step)

### 1. Clone or download the project

```bash
cd heart_disease_cnn
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

```bash
python data/download_data.py
```

> If the UCI server is unavailable, a synthetic fallback dataset is generated automatically.

### 5. (Optional) Run EDA notebook

```bash
jupyter notebook notebooks/eda.ipynb
```

### 6. Train the model

```bash
python train.py
```

This will:
- Clean and preprocess the data
- Build and train the CNN
- Save the model to `models/cnn_heart_model.keras`
- Generate training plots in `models/`

### 7. Evaluate the model

```bash
python evaluate.py
```

Outputs accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix.

### 8. Launch the web UI

```bash
streamlit run app/streamlit_app.py
```

Open your browser at **http://localhost:8501**

---

## 📊 Model Architecture

```
Input (13, 1)
    │
Conv1D(32, kernel=3, padding='same') → BatchNorm → ReLU
    │
Conv1D(64, kernel=3, padding='same') → BatchNorm → ReLU
    │
GlobalAveragePooling1D
    │
Dense(128) → BatchNorm → ReLU → Dropout(0.4)
    │
Dense(64)  → ReLU → Dropout(0.3)
    │
Dense(1)   → Sigmoid  →  P(heart disease) ∈ [0, 1]
```

**Loss:** Binary Cross-Entropy  
**Optimizer:** Adam (lr = 1e-3, with ReduceLROnPlateau)  
**Callbacks:** EarlyStopping (patience=15), ReduceLROnPlateau, ModelCheckpoint

---

## 📈 Expected Performance (UCI Cleveland dataset)

| Metric | Value (approx.) |
|---|---|
| Accuracy | 85 – 88 % |
| Precision | 84 – 87 % |
| Recall | 86 – 90 % |
| F1-Score | 85 – 88 % |
| ROC-AUC | 0.90 – 0.94 |

*(Exact values vary slightly with random seed.)*

---

## 🖥️ Web UI Preview

The Streamlit app features:
- 13 clinical input sliders / dropdowns
- Real-time CNN inference
- **High Risk / Low Risk** verdict
- Probability score with confidence bar
- Medical disclaimer

---

## ⚠️ Disclaimer

This project is for **educational and portfolio purposes only**.
It is **NOT** a medical device and should **NOT** be used for clinical decision-making.

---

## 📄 Dataset

**UCI Cleveland Heart Disease Dataset**  
Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease)  
Rows: 303 patients | Features: 13 clinical attributes

---

## 👤 Author

Built as a portfolio / academic project demonstrating end-to-end deep learning
with TensorFlow, following industry best practices.
