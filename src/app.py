import streamlit as st
import numpy as np
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Voice Disorder Detection", layout="centered")

# =========================
# FEATURE EXTRACTION
# =========================
def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

# =========================
# TRAIN MODEL
# =========================
@st.cache_resource
def train_model():
    dataset_path = "../dataset"

    healthy_path = os.path.join(dataset_path, "healthy")
    pathological_path = os.path.join(dataset_path, "pathological")

    X = []
    y = []

    for file in os.listdir(healthy_path):
        try:
            features = extract_mfcc(os.path.join(healthy_path, file))
            X.append(features)
            y.append(0)
        except:
            continue

    for file in os.listdir(pathological_path):
        try:
            features = extract_mfcc(os.path.join(pathological_path, file))
            X.append(features)
            y.append(1)
        except:
            continue

    X = np.array(X)
    y = np.array(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X, y)

    return model, scaler

model, scaler = train_model()

# =========================
# UI HEADER
# =========================
st.title("🎤 Voice Disorder Detection System")
st.caption("Upload a voice sample to analyze vocal health")

uploaded_file = st.file_uploader("Upload .wav file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    # Save temp file
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    # Load audio
    audio, sr = librosa.load("temp.wav", sr=22050)

    # =========================
    # WAVEFORM
    # =========================
    st.markdown("### 📊 Waveform")
    fig1, ax1 = plt.subplots()
    librosa.display.waveshow(audio, sr=sr, ax=ax1)
    ax1.set_title("")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Amplitude")
    st.pyplot(fig1)

    # =========================
    # MFCC
    # =========================
    st.markdown("### 📊 MFCC Features")
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    fig2, ax2 = plt.subplots()
    img = ax2.imshow(mfcc, aspect='auto', origin='lower')
    ax2.set_xlabel("Time")
    ax2.set_ylabel("MFCC Coefficients")
    fig2.colorbar(img, ax=ax2)
    st.pyplot(fig2)

    # =========================
    # PREDICTION
    # =========================
    features = extract_mfcc("temp.wav")
    features = scaler.transform([features])

    prediction = model.predict(features)
    probability = model.predict_proba(features)
    confidence = float(np.max(probability))

    st.markdown("### 🔍 Prediction")

    if prediction[0] == 0:
        st.success("Healthy Voice")
    else:
        st.error("Pathological Voice")

    # =========================
    # CONFIDENCE BAR
    # =========================
    st.markdown("### 📈 Confidence Level")
    st.progress(confidence)

    st.write(f"Confidence Score: {confidence:.2f}")