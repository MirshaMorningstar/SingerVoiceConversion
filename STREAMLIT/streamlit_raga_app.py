import streamlit as st
import numpy as np
import pandas as pd
import librosa
from tensorflow.keras.models import load_model
import tempfile
import os

# =========================
# Streamlit page config
# =========================
st.set_page_config(page_title="Automatic Raga Detection", layout="centered")

st.title("🎵 Automatic Raga Detection from Audio")
st.write("Upload an audio file to predict the raga and extract swara timeline.")

# =========================
# Load trained CNN model
# =========================
@st.cache_resource
def load_raga_model():
    return load_model("final_raga_model.h5")

model = load_raga_model()

# =========================
# Raga labels taken from MEL image folder names
# IMPORTANT: Order must match training order
# =========================
raga_labels = sorted(os.listdir("mel"))

# =========================
# File uploader
# =========================
uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3"])

if uploaded_file is not None:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    st.audio(uploaded_file)

    # =========================
    # Load audio
    # =========================
    y, sr = librosa.load(audio_path, sr=22050)

    # =========================
    # Create Mel spectrogram for CNN
    # =========================
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=231)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = librosa.util.fix_length(mel_db, size=232, axis=1)
    mel_db = mel_db[np.newaxis, ..., np.newaxis]

    # =========================
    # Predict raga
    # =========================
    pred = model.predict(mel_db, verbose=0)[0]
    pred_index = np.argmax(pred)

    pred_raga = raga_labels[pred_index]
    confidence = float(pred[pred_index] * 100)

    st.success(f"Predicted Raga: {pred_raga}")
    st.info(f"Confidence: {confidence:.2f}%")

    # =========================
    # Swara detection
    # =========================
    n_mels = 128
    hop_length = 512

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_db2 = librosa.power_to_db(mel_spec, ref=np.max)

    times = librosa.frames_to_time(np.arange(mel_db2.shape[1]), sr=sr, hop_length=hop_length)
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmax=sr / 2)

    SA_FREQ = 240
    ratios = {
        "Sa": 1, "Ri1": 16/15, "Ri2": 9/8, "Ga2": 6/5, "Ga3": 5/4,
        "Ma1": 4/3, "Ma2": 45/32, "Pa": 3/2, "Da1": 8/5, "Da2": 5/3,
        "Ni2": 9/5, "Ni3": 15/8, "Sa_high": 2
    }
    swara_freqs = {k: SA_FREQ * v for k, v in ratios.items()}

    rows = []
    for i in range(mel_db2.shape[1]):
        idx = np.argmax(mel_db2[:, i])
        freq = mel_freqs[idx]
        swara = min(swara_freqs, key=lambda s: abs(swara_freqs[s] - freq))

        rows.append({
            "Time_sec": round(times[i], 2),
            "Freq_Hz": round(freq, 2),
            "Swara": swara
        })

    df_swara = pd.DataFrame(rows)

    # Remove consecutive duplicate swaras
    df_swara = df_swara.loc[df_swara["Swara"].shift() != df_swara["Swara"]]

    st.subheader("Detected Swara Timeline")
    st.dataframe(df_swara, use_container_width=True)

    # =========================
    # Save CSV results
    # =========================
    summary_df = pd.DataFrame([{
        "Audio_File": uploaded_file.name,
        "Predicted_Raga": pred_raga,
        "Confidence_%": round(confidence, 2)
    }])

    summary_csv = summary_df.to_csv(index=False).encode("utf-8")
    swara_csv = df_swara.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Raga Prediction CSV",
        data=summary_csv,
        file_name="raga_prediction_summary.csv",
        mime="text/csv"
    )

    st.download_button(
        label="Download Swara Timeline CSV",
        data=swara_csv,
        file_name="swara_timeline.csv",
        mime="text/csv"
    )

    # Cleanup temp file
    os.remove(audio_path)

else:
    st.info("Please upload an audio file to begin.")
