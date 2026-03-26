# ❤️ Heart Disease CNN Prediction - Quick Start Guide

## 🚀 Quick Setup

### 1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/heart_disease_cnn.git
cd heart_disease_cnn
```

### 2. **Create Virtual Environment** (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. **Install Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ **Note**: Python 3.10-3.12 recommended. Python 3.14+ doesn't yet have PyTorch/TensorFlow wheels.

### 4. **Download Dataset**
```bash
python data/download_data.py
```

### 5. **Train the Model (Optional)**
```bash
# Using PyTorch (Recommended)
python train_pytorch.py

# Using TensorFlow (Requires Python <= 3.11)
# python train.py
```

### 6. **Run the Streamlit App**
```bash
streamlit run app/streamlit_app_pytorch.py
```

Then open your browser to `http://localhost:8501`

---

## 📁 App Files

| File | Purpose | Backend |
|------|---------|---------|
| `streamlit_app_pytorch.py` | **Main app** - Recommended | PyTorch |
| `streamlit_app.py` | Alternative UI | PyTorch |
| `streamlit_app_final.py` | Legacy version | TensorFlow |

---

## 📊 Usage

1. **Enter patient clinical features** (age, sex, blood pressure, cholesterol, etc.)
2. **Click "Predict"** to get risk assessment
3. **View probability scores** and interpretation

---

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"
```bash
pip install -r requirements.txt
```

### Issue: "No module named 'utils'"
Make sure you're running commands from the project root directory.

### Issue: Model file not found
Download the pre-trained model or train one:
```bash
python train_pytorch.py
```

---

## 📚 Learn More

- [Model Architecture](models/cnn_model_pytorch.py)
- [Data Preprocessing](utils/preprocessing.py)
- [Full README](README.md)
- [Contributing Guidelines](CONTRIBUTING.md)

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

---

**Made with ❤️ for heart disease prediction**
