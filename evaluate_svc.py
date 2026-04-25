import librosa
import numpy as np
from jiwer import wer, cer
from resemblyzer import VoiceEncoder
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
from transformers import pipeline
import os

# --------------------------------------------------
# FILE PATHS (CHANGE THESE)
# --------------------------------------------------
SOURCE_PATH = "audio/source.wav"
CONVERTED_PATH = "output/converted.wav"
TARGET_PATH = "audio/target.wav"

REPORT_PATH = "output/svc_evaluation.txt"

# --------------------------------------------------
# 1️⃣ LOAD AUDIO
# --------------------------------------------------
print("Loading audio files...")

src, sr = librosa.load(SOURCE_PATH, sr=16000)
conv, _ = librosa.load(CONVERTED_PATH, sr=16000)
tgt, _ = librosa.load(TARGET_PATH, sr=16000)

print("Audio Loaded Successfully")

# --------------------------------------------------
# 2️⃣ LYRICS PRESERVATION (WER / CER)
# --------------------------------------------------
print("\nRunning Whisper ASR for lyric comparison...")

asr = pipeline("automatic-speech-recognition",
               model="openai/whisper-small")

src_text = asr(SOURCE_PATH)["text"]
conv_text = asr(CONVERTED_PATH)["text"]

wer_score = wer(src_text, conv_text)
cer_score = cer(src_text, conv_text)

# --------------------------------------------------
# 3️⃣ SPEAKER SIMILARITY (IDENTITY TRANSFER)
# --------------------------------------------------
print("\nComputing Speaker Embeddings...")

encoder = VoiceEncoder()

src_embed = encoder.embed_utterance(src)
conv_embed = encoder.embed_utterance(conv)
tgt_embed = encoder.embed_utterance(tgt)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarity_to_target = cosine_similarity(conv_embed, tgt_embed)
similarity_to_source = cosine_similarity(conv_embed, src_embed)

# --------------------------------------------------
# 4️⃣ RAGA / MELODY PRESERVATION (F0 CORRELATION)
# --------------------------------------------------
print("\nExtracting pitch contours...")

f0_src = librosa.yin(src, fmin=80, fmax=800)
f0_conv = librosa.yin(conv, fmin=80, fmax=800)

min_len = min(len(f0_src), len(f0_conv))

f0_src = f0_src[:min_len]
f0_conv = f0_conv[:min_len]

# remove NaNs before correlation
valid_idx = ~(np.isnan(f0_src) | np.isnan(f0_conv))
f0_corr, _ = pearsonr(f0_src[valid_idx], f0_conv[valid_idx])

# --------------------------------------------------
# 5️⃣ TIMBRE SHIFT (MFCC DISTANCE)
# --------------------------------------------------
print("\nMeasuring spectral timbre shift...")

mfcc_src = librosa.feature.mfcc(y=src, sr=sr, n_mfcc=13)
mfcc_conv = librosa.feature.mfcc(y=conv, sr=sr, n_mfcc=13)
mfcc_tgt = librosa.feature.mfcc(y=tgt, sr=sr, n_mfcc=13)

def avg_distance(A, B):
    min_len = min(A.shape[1], B.shape[1])
    return np.mean([
        euclidean(A[:, i], B[:, i]) for i in range(min_len)
    ])

dist_source = avg_distance(mfcc_src, mfcc_conv)
dist_target = avg_distance(mfcc_tgt, mfcc_conv)

# --------------------------------------------------
# 6️⃣ EXPRESSION / ENERGY PRESERVATION
# --------------------------------------------------
energy_src = np.mean(librosa.feature.rms(y=src))
energy_conv = np.mean(librosa.feature.rms(y=conv))

energy_diff = abs(energy_src - energy_conv)

# --------------------------------------------------
# FINAL REPORT (PRINT + SAVE)
# --------------------------------------------------
report = f"""
==================================================
        SINGING VOICE CONVERSION REPORT
==================================================

🎵 Linguistic Preservation:
WER : {wer_score:.3f}
CER : {cer_score:.3f}

🎤 Singer Identity Transfer:
Similarity to TARGET : {similarity_to_target:.3f}
Residual similarity to SOURCE : {similarity_to_source:.3f}

🎼 Musical Preservation:
F0 Correlation (Raga Retention) : {f0_corr:.3f}

🎧 Timbre Transformation:
Distance from SOURCE : {dist_source:.3f}
Distance to TARGET   : {dist_target:.3f}

🔊 Expression Preservation:
Energy Deviation : {energy_diff:.5f}

==================================================
Interpretation:
"""

if f0_corr > 0.8:
    report += "\n✔ Melody preserved successfully."

if similarity_to_target > similarity_to_source:
    report += "\n✔ Voice shifted toward target singer."

if wer_score < 0.2:
    report += "\n✔ Lyrics intelligibility maintained."

report += "\n==================================================\n"

print(report)

os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report)

print(f"\nReport saved to: {REPORT_PATH}")
