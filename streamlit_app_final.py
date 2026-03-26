"""
app/streamlit_app_final.py
---------------------
Streamlit web interface for Heart Disease Risk Prediction.
"""

import os
import sys

import pandas as pd
import streamlit as st
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="wide")


@st.cache_resource
def load_resources():
    from models.cnn_model_pytorch import MODEL_PATH, load_model
    from utils.preprocessing import preprocess_single

    model = load_model(MODEL_PATH)
    return model, preprocess_single


st.markdown(
    """
<style>
    :root {
        --bg-1: #171032;
        --bg-2: #2c155a;
        --bg-3: #4f2d8f;
        --card-border: rgba(255, 255, 255, 0.18);
        --text-main: #f7f2ff;
        --text-soft: #d7c9ff;
        --accent: #ff5f9e;
        --accent-2: #ff9a57;
    }

    .stApp {
        background:
            radial-gradient(circle at 20% 15%, #6b42c9 0%, rgba(107, 66, 201, 0) 40%),
            radial-gradient(circle at 85% 85%, #8f35c9 0%, rgba(143, 53, 201, 0) 35%),
            linear-gradient(130deg, var(--bg-1) 0%, var(--bg-2) 55%, var(--bg-3) 100%);
        color: var(--text-main);
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1rem;
        max-width: 1300px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10, 8, 28, 0.9), rgba(35, 19, 73, 0.9));
        border-right: 1px solid rgba(255, 255, 255, 0.14);
    }

    section[data-testid="stSidebar"] * {
        color: var(--text-main) !important;
    }

    .main-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        color: #ff8aac;
        margin-bottom: 0.1rem;
        letter-spacing: 0.2px;
    }

    .subtitle {
        text-align: center;
        color: var(--text-soft);
        margin-top: 0;
        margin-bottom: 0.8rem;
        font-size: 1.01rem;
    }

    .disclaimer {
        background: linear-gradient(90deg, rgba(255, 196, 114, 0.95), rgba(255, 209, 127, 0.92));
        color: #3f2b00;
        border-radius: 10px;
        padding: 0.7rem 0.9rem;
        border: 1px solid rgba(255, 255, 255, 0.35);
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }

    .glass-card {
        background: linear-gradient(140deg, rgba(33, 25, 72, 0.72), rgba(61, 38, 120, 0.55));
        border: 1px solid var(--card-border);
        border-radius: 16px;
        box-shadow: 0 10px 28px rgba(6, 3, 20, 0.32);
        backdrop-filter: blur(5px);
        padding: 0.95rem 1rem;
    }

    .card-title {
        font-size: 1.9rem;
        font-weight: 800;
        margin: 0 0 0.6rem 0;
        color: var(--text-main);
    }

    .form-heading {
        color: #ffb8d4;
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 1px;
        margin-top: 0.7rem;
        margin-bottom: 0.35rem;
        border-bottom: 1px solid rgba(255, 111, 160, 0.7);
        padding-bottom: 0.2rem;
        text-transform: uppercase;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--card-border);
        border-radius: 12px;
        overflow: hidden;
    }

    .stButton > button {
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        color: #fff !important;
        border: none;
        border-radius: 10px;
        font-weight: 700;
    }

    .stButton > button:hover {
        border: none;
        color: #fff !important;
        filter: brightness(1.04);
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #7fdb8f, #ffd166 55%, #ff5a74);
    }

    .prediction-sub {
        color: #e2d6ff;
        margin-top: 0.2rem;
        margin-bottom: 0.7rem;
    }

    .risk-badge {
        border-radius: 12px;
        padding: 0.95rem;
        margin-bottom: 0.7rem;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.14);
    }

    .risk-level {
        color: #e9defe;
        font-size: 1.45rem;
        font-weight: 600;
        margin-bottom: 0.15rem;
    }

    .risk-value-high {
        color: #ff6e8b;
        font-size: 2.15rem;
        font-weight: 800;
    }

    .risk-value-low {
        color: #67e5a5;
        font-size: 2.15rem;
        font-weight: 800;
    }

    .doctor-box {
        margin-top: 0.7rem;
        border-radius: 10px;
        padding: 0.65rem 0.8rem;
        background: linear-gradient(120deg, rgba(123, 233, 255, 0.34), rgba(255, 173, 201, 0.28));
        border: 1px solid rgba(255, 255, 255, 0.22);
        color: #f7f2ff;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<p class="main-title">❤️ Heart Disease Risk Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Powered by a 1-D Convolutional Neural Network trained on the UCI Cleveland Heart Disease dataset</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="disclaimer">⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only and is not a substitute for professional medical advice.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 📋 Patient Information")
    st.caption("Fill in the clinical parameters below.")
    st.markdown('<div class="form-heading">Demographics</div>', unsafe_allow_html=True)

    age = st.slider("Age (years)", 20, 80, 54)
    sex = st.selectbox("Sex", options=["Male (1)", "Female (0)"])
    sex_val = 1 if sex.startswith("Male") else 0

    st.markdown('<div class="form-heading">Chest Pain & Blood Pressure</div>', unsafe_allow_html=True)
    cp = st.selectbox(
        "Chest Pain Type",
        options=[
            "0 - Typical Angina",
            "1 - Atypical Angina",
            "2 - Non-Anginal Pain",
            "3 - Asymptomatic",
        ],
    )
    cp_val = int(cp[0])
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 130)

    st.markdown('<div class="form-heading">Blood Tests</div>', unsafe_allow_html=True)
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 246)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["No (0)", "Yes (1)"])
    fbs_val = int(fbs[-2])

    st.markdown('<div class="form-heading">ECG & Exercise</div>', unsafe_allow_html=True)
    restecg = st.selectbox(
        "Resting ECG Result",
        options=["0 - Normal", "1 - ST-T Abnormality", "2 - LV Hypertrophy"],
    )
    restecg_val = int(restecg[0])
    thalach = st.slider("Max Heart Rate Achieved", 70, 210, 150)
    exang = st.selectbox("Exercise Induced Angina", options=["No (0)", "Yes (1)"])
    exang_val = int(exang[-2])
    oldpeak = st.slider("ST Depression", 0.0, 6.5, 1.0, step=0.1)

    st.markdown('<div class="form-heading">Advanced</div>', unsafe_allow_html=True)
    slope = st.selectbox("Slope of ST Segment", options=["0 - Upsloping", "1 - Flat", "2 - Downsloping"])
    slope_val = int(slope[0])
    ca = st.selectbox("Major Vessels (Fluoroscopy)", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", options=["1 - Normal", "2 - Fixed Defect", "3 - Reversible Defect"])
    thal_val = int(thal[0])

    predict_btn = st.button("Predict Risk", use_container_width=True)

feature_names = [
    "Age",
    "Sex",
    "Chest Pain Type",
    "Resting BP",
    "Cholesterol",
    "Fasting Blood Sugar",
    "Resting ECG",
    "Max Heart Rate",
    "Exercise Angina",
    "ST Depression",
    "ST Slope",
    "Major Vessels",
    "Thalassemia",
]
raw_values = [
    age,
    sex_val,
    cp_val,
    trestbps,
    chol,
    fbs_val,
    restecg_val,
    thalach,
    exang_val,
    oldpeak,
    slope_val,
    ca,
    thal_val,
]

col_left, col_right = st.columns([1.25, 1], gap="large")

with col_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Input Summary")
    summary_df = pd.DataFrame({"Feature": feature_names, "Value": raw_values})
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Prediction")

    if predict_btn:
        try:
            model, preprocess_single = load_resources()
            with st.spinner("Running CNN inference..."):
                x = preprocess_single(raw_values)
                x_tensor = torch.FloatTensor(x.reshape(x.shape[0], x.shape[1]))
                with torch.no_grad():
                    prob = float(model(x_tensor).item())

            risk_pct = prob * 100
            is_high = prob >= 0.5
            risk_class = "risk-value-high" if is_high else "risk-value-low"
            risk_label = "High Risk" if is_high else "Low Risk"
            st.markdown(
                f"""
                <div class="risk-badge">
                    <div class="risk-level">Heart Disease Risk Level:</div>
                    <div class="{risk_class}">{risk_label}</div>
                    <p class="prediction-sub">Probability Score: <strong>{risk_pct:.1f}%</strong></p>
                </div>
                <div class="doctor-box">🩺 Keep this prediction as a screening aid and confirm with a clinical assessment.</div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(prob, text=f"Probability Score: {risk_pct:.1f}%")
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.info("Train the model first: `python train_pytorch.py`")
        except Exception as exc:
            st.error(f"Prediction error: {exc}")
    else:
        st.info("Fill in patient details and click `Predict Risk`.")
    st.markdown("</div>", unsafe_allow_html=True)
"""
app/streamlit_app_final.py
-----------------------------
Streamlit web interface for Heart Disease Risk Prediction using PyTorch.

Run from the project ROOT directory:
    streamlit run app/streamlit_app_final.py

The app lets a user enter 13 clinical parameters and instantly
see whether the model predicts Low Risk or High Risk, along with
a probability score and a confidence bar.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import torch

# ── Page config must be the very first Streamlit call ────────────────────────
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
)

# ── Lazy import so the app doesn't crash before the model is trained ─────────
@st.cache_resource
def load_resources():
    from models.cnn_model_pytorch import load_model, MODEL_PATH
    from utils.preprocessing import preprocess_single
    model = load_model(MODEL_PATH)
    return model, preprocess_single


# ══════════════════════════════════════════════════════════════════════════════
# Custom CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Main header */
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #e53935, #b71c1c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #555;
        font-size: 1.05rem;
        margin-top: 0.2rem;
        margin-bottom: 2rem;
    }
    /* Prediction cards */
    .high-risk {
        background: linear-gradient(135deg, #ffcdd2, #ef9a9a);
        border-left: 6px solid #e53935;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
    }
    .low-risk {
        background: linear-gradient(135deg, #c8e6c9, #a5d6a7);
        border-left: 6px solid #2e7d32;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
    }
    .risk-label {
        font-size: 2rem;
        font-weight: 800;
    }
    .prob-text {
        font-size: 1.15rem;
        margin-top: 0.5rem;
    }
    /* Input section label */
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #c62828;
        border-bottom: 2px solid #ef9a9a;
        padding-bottom: 4px;
        margin-bottom: 0.8rem;
    }
    /* Info box */
    .info-box {
        background: #fff8e1;
        border-left: 5px solid #f9a825;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        font-size: 0.92rem;
    }
    /* Card styling */
    .prediction-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    /* Risk level styling */
    .risk-level {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .probability-score {
        font-size: 2.5rem;
        font-weight: 900;
        color: #e53935;
        margin: 1rem 0;
    }
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #4caf50, #ffc107, #e53935);
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="main-title">❤️ Heart Disease Risk Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Powered by a 1-D Convolutional Neural Network trained on the '
    'UCI Cleveland Heart Disease dataset</p>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="info-box">⚠️ <strong>Disclaimer:</strong> This tool is for '
    'educational purposes only and is <strong>not</strong> a substitute for '
    'professional medical advice.</div>',
    unsafe_allow_html=True,
)
st.write("")

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar — Patient Information
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    st.header("📋 Patient Information")
    st.caption("Fill in the clinical parameters below.")

    st.markdown('<p class="section-header">Demographics</p>', unsafe_allow_html=True)
    age = st.slider("Age (years)", 20, 80, 54)
    sex = st.selectbox("Sex", options=["Male (1)", "Female (0)"])
    sex_val = 1 if sex.startswith("Male") else 0

    st.markdown('<p class="section-header">Chest Pain & Blood Pressure</p>', unsafe_allow_html=True)
    cp = st.selectbox(
        "Chest Pain Type",
        options=[
            "0 — Typical Angina",
            "1 — Atypical Angina",
            "2 — Non-Anginal Pain",
            "3 — Asymptomatic",
        ],
    )
    cp_val = int(cp[0])
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 130)

    st.markdown('<p class="section-header">Blood Tests</p>', unsafe_allow_html=True)
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 246)
    fbs  = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["No (0)", "Yes (1)"])
    fbs_val = int(fbs[-2])

    st.markdown('<p class="section-header">ECG & Exercise</p>', unsafe_allow_html=True)
    restecg = st.selectbox(
        "Resting ECG Result",
        options=["0 — Normal", "1 — ST-T Abnormality", "2 — LV Hypertrophy"],
    )
    restecg_val = int(restecg[0])
    thalach = st.slider("Max Heart Rate Achieved", 70, 210, 150)
    exang   = st.selectbox("Exercise Induced Angina", options=["No (0)", "Yes (1)"])
    exang_val = int(exang[-2])
    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.5, 1.0, step=0.1)

    st.markdown('<p class="section-header">Advanced</p>', unsafe_allow_html=True)
    slope = st.selectbox(
        "Slope of ST Segment",
        options=["0 — Upsloping", "1 — Flat", "2 — Downsloping"],
    )
    slope_val = int(slope[0])
    ca   = st.selectbox("Major Vessels (Fluoroscopy)", options=[0, 1, 2, 3])
    thal = st.selectbox(
        "Thalassemia",
        options=["1 — Normal", "2 — Fixed Defect", "3 — Reversible Defect"],
    )
    thal_val = int(thal[0])

    predict_btn = st.button("🔍  Predict Risk", use_container_width=True, key="predict_button")
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main area — results
# ══════════════════════════════════════════════════════════════════════════════
col1, col2 = st.columns([1.4, 1])

with col1:
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    st.subheader("📊 Input Summary")
    feature_names = [
        "Age", "Sex", "Chest Pain Type", "Resting BP",
        "Cholesterol", "Fasting Blood Sugar", "Resting ECG",
        "Max Heart Rate", "Exercise Angina", "ST Depression",
        "ST Slope", "Major Vessels", "Thalassemia",
    ]
    raw_values = [
        age, sex_val, cp_val, trestbps, chol, fbs_val,
        restecg_val, thalach, exang_val, oldpeak,
        slope_val, ca, thal_val,
    ]
    import pandas as pd
    summary_df = pd.DataFrame({"Feature": feature_names, "Value": raw_values})
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    st.subheader("🎯 Prediction")

    if predict_btn:
        try:
            model, preprocess_single = load_resources()

            with st.spinner("Running CNN inference …"):
                x = preprocess_single(raw_values)
                x_tensor = torch.FloatTensor(x.reshape(x.shape[0], x.shape[1]))
                with torch.no_grad():
                    prob = float(model(x_tensor).sigmoid().item())

            risk_pct = prob * 100

            # Risk level display
            risk_level = "High Risk" if prob >= 0.5 else "Low Risk"
            risk_color = "#e53935" if prob >= 0.5 else "#2e7d32"
            
            st.markdown(
                f'<div class="risk-level" style="color: {risk_color}">Heart Disease Risk Level</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="risk-label" style="color: {risk_color}">{risk_level}</div>',
                unsafe_allow_html=True,
            )
            
            st.markdown(
                f'<div class="probability-score" style="color: {risk_color}">{risk_pct:.1f}%</div>',
                unsafe_allow_html=True,
            )
            
            st.markdown(
                f'<div class="prob-text">Probability Score</div>',
                unsafe_allow_html=True,
            )

            st.write("")
            st.progress(prob, text=f"Disease probability: {risk_pct:.1f}%")

            # Confidence interpretation
            if prob < 0.3:
                st.success("Model is highly confident: Low Risk.")
            elif prob < 0.5:
                st.info("Model leans Low Risk, but borderline — consult a doctor.")
            elif prob < 0.7:
                st.warning("Model leans High Risk — please seek medical evaluation.")
            else:
                st.error("Model is highly confident: High Risk — urgent evaluation recommended.")

        except FileNotFoundError as e:
            st.error(str(e))
            st.info("Train the model first:\n```\npython train_pytorch.py\n```")
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.info("👈 Fill in the patient details in the sidebar and click **Predict Risk**.")
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.caption(
    "**Model:** 1-D CNN | **Dataset:** UCI Cleveland Heart Disease | "
    "**Framework:** PyTorch"
)
