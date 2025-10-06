# -----------------------------
# 🎤 Streamlit Speech Emotion Recognition App
# -----------------------------
import streamlit as st
import joblib
import os
import soundfile as sf
import numpy as np
import librosa

# -----------------------------
# Load pipeline
# -----------------------------
PIPELINE_PATH = r"enhanced_speech_emotion_pipeline.joblib"  # place in same folder as app.py
pipeline = joblib.load(PIPELINE_PATH)

# -----------------------------
# Feature extraction
# -----------------------------
def extract_feature_from_array(X, sr, mfcc=True, chroma=True, mel=True,
                               n_mfcc=40, n_mels=128, n_fft=2048):
    feats = []
    try:
        if mfcc:
            mfccs = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)
            feats.append(np.mean(mfccs, axis=1))
            feats.append(np.mean(librosa.feature.delta(mfccs), axis=1))
        if chroma:
            stft = np.abs(librosa.stft(y=X, n_fft=n_fft))
            chroma_feat = librosa.feature.chroma_stft(S=stft, sr=sr)
            feats.append(np.mean(chroma_feat, axis=1))
        if mel:
            mel_spec = librosa.feature.melspectrogram(y=X, sr=sr, n_mels=n_mels, n_fft=n_fft)
            feats.append(np.mean(mel_spec, axis=1))
        spectral_contrast = librosa.feature.spectral_contrast(y=X, sr=sr, n_fft=n_fft)
        feats.append(np.mean(spectral_contrast, axis=1))
        zcr = librosa.feature.zero_crossing_rate(y=X)
        feats.append(np.mean(zcr, axis=1))
        return np.concatenate(feats).reshape(1, -1)
    except Exception:
        return None

def extract_feature(file_path):
    try:
        X, sr = sf.read(file_path, dtype='float32')
        if X.ndim > 1:
            X = np.mean(X, axis=1)
        return extract_feature_from_array(X, sr)
    except Exception:
        return None

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎤 Speech Emotion Recognition")
st.write("Upload a `.wav` file and the app will predict the emotion.")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

if uploaded_file is not None:
    # Save temporarily
    temp_path = "temp.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Play audio
    st.audio(temp_path, format="audio/wav")

    # Extract features and predict
    features = extract_feature(temp_path)
    if features is not None:
        try:
            emotion = pipeline.predict(features)[0]
            st.success(f"Predicted emotion: **{emotion}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.error("Could not extract features from the audio.")

    # Clean up
    os.remove(temp_path)
