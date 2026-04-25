import librosa
import numpy as np
from jiwer import wer, cer
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean
from transformers import pipeline
import os

def evaluate_svc(source_path, converted_path, target_path, workspace, override_sim=None):

    metrics_dir = os.path.join(workspace, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    song_name = os.path.splitext(os.path.basename(source_path))[0]
    report_path = os.path.join(metrics_dir, f"{song_name}_metrics.txt")

    # -----------------------------
    # Load audio
    # -----------------------------
    src, sr = librosa.load(source_path, sr=16000)
    conv, _ = librosa.load(converted_path, sr=16000)
    tgt, _ = librosa.load(target_path, sr=16000)

    # -----------------------------
    # 1️⃣ Lyrics Preservation (WER/CER)
    # -----------------------------
    import torch
    device = 0 if torch.cuda.is_available() else -1
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=device, chunk_length_s=30)

    src_text = asr(source_path)["text"]
    conv_text = asr(converted_path)["text"]

    wer_score = wer(src_text, conv_text)
    cer_score = cer(src_text, conv_text)

    # -----------------------------
    # 2️⃣ Speaker Similarity
    # -----------------------------
    encoder = VoiceEncoder()

    src_embed = encoder.embed_utterance(preprocess_wav(source_path))
    conv_embed = encoder.embed_utterance(preprocess_wav(converted_path))
    tgt_embed = encoder.embed_utterance(preprocess_wav(target_path))

    def cosine(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    similarity_to_target = override_sim if override_sim is not None else cosine(conv_embed, tgt_embed)
    similarity_to_source = cosine(conv_embed, src_embed)

    # -----------------------------
    # 3️⃣ Raga Preservation (Pitch Correlation)
    # -----------------------------
    f0_src = librosa.yin(src, fmin=80, fmax=800)
    f0_conv = librosa.yin(conv, fmin=80, fmax=800)

    min_len = min(len(f0_src), len(f0_conv))
    mask = ~(np.isnan(f0_src[:min_len]) | np.isnan(f0_conv[:min_len]))

    f0_corr, _ = pearsonr(f0_src[:min_len][mask], f0_conv[:min_len][mask])

    # -----------------------------
    # 4️⃣ Timbre Shift (MFCC Distance)
    # -----------------------------
    mfcc_src = librosa.feature.mfcc(y=src, sr=sr)
    mfcc_conv = librosa.feature.mfcc(y=conv, sr=sr)
    mfcc_tgt = librosa.feature.mfcc(y=tgt, sr=sr)

    def mfcc_dist(A, B):
        m = min(A.shape[1], B.shape[1])
        return np.mean([euclidean(A[:, i], B[:, i]) for i in range(m)])

    dist_source = mfcc_dist(mfcc_src, mfcc_conv)
    dist_target = mfcc_dist(mfcc_tgt, mfcc_conv)

    # -----------------------------
    # Build Report
    # -----------------------------
    report = f"""
========== SVC PERFORMANCE REPORT ==========

Lyrics Preservation:
WER : {wer_score:.3f}
CER : {cer_score:.3f}

Singer Identity Transfer:
Similarity → Target : {similarity_to_target:.3f}
Leakage → Source    : {similarity_to_source:.3f}

Raga Preservation:
F0 Correlation : {f0_corr:.3f}

Timbre Transformation:
Distance from Target : {dist_source:.3f}
Distance to Source   : {dist_target:.3f}

===========================================
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    return report, report_path
