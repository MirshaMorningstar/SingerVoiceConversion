"""
RAGA EXTRACTOR
==============
Extracts pitch/swaras from audio, identifies closest raga.
Uses lyrical section (30s-90s) for better accuracy on Tamil songs.
"""

import numpy as np
import librosa
from core.raga_knowledge_base import RAGA_SWARAS, SEMITONE_TO_SWARA, SWARA_TO_SEMITONE, EMOTION_RAGA_MAP


def extract_features(audio_path: str) -> dict:
    y_full, sr = librosa.load(audio_path, mono=True)

    # Skip to lyrical section (30s to 90s)
    start = int(30 * sr)
    end   = int(90 * sr)
    if len(y_full) < end:
        mid   = len(y_full) // 2
        start = max(0, mid - int(30 * sr))
        end   = min(len(y_full), mid + int(30 * sr))
    y = y_full[start:end]

    chroma      = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=36)
    mean_chroma = np.mean(chroma, axis=1)
    mean_chroma /= (mean_chroma.max() + 1e-9)

    tonic_idx = int(np.argmax(mean_chroma))
    rolled    = np.roll(mean_chroma, -tonic_idx)

    dominant       = [i for i in range(12) if rolled[i] > 0.25]
    dominant_names = [SEMITONE_TO_SWARA[i] for i in dominant]

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo    = float(np.atleast_1d(tempo)[0])

    return {
        "y":              y,
        "y_full":         y_full,
        "sr":             sr,
        "chroma_vector":  rolled,
        "tonic_idx":      tonic_idx,
        "tempo_bpm":      tempo,
        "dominant_names": dominant_names,
        "duration_sec":   len(y_full) / sr,
        "lyric_start":    start,
        "lyric_end":      end,
    }


def identify_raga(chroma_vector: np.ndarray) -> list:
    results = []
    for raga_name, info in RAGA_SWARAS.items():
        # Build binary chroma vector using semitones (integers only)
        raga_vec = np.zeros(12)
        for s in info.get("semitones", []):
            if isinstance(s, (int, float)):
                raga_vec[int(s) % 12] = 1.0

        dot   = np.dot(chroma_vector, raga_vec)
        norm  = np.linalg.norm(chroma_vector) * np.linalg.norm(raga_vec)
        score = float(dot / (norm + 1e-9))

        results.append({
            "raga":    raga_name,
            "emotion": info["emotion"],
            "score":   round(score, 4),
            "swaras":  info.get("swaras", []),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def full_extraction_report(audio_path: str) -> dict:
    feats   = extract_features(audio_path)
    matches = identify_raga(feats["chroma_vector"])
    best    = matches[0]

    return {
        "file":             audio_path,
        "duration_sec":     round(feats["duration_sec"], 1),
        "tempo_bpm":        round(feats["tempo_bpm"], 1),
        "tonic_semitone":   feats["tonic_idx"],
        "dominant_swaras":  feats["dominant_names"],
        "detected_raga":    best["raga"],
        "detected_emotion": best["emotion"],
        "match_score":      best["score"],
        "all_matches":      matches,
        "_y":               feats["y"],
        "_y_full":          feats["y_full"],
        "_sr":              feats["sr"],
        "_chroma":          feats["chroma_vector"],
        "_tonic":           feats["tonic_idx"],
        "_lyric_start":     feats["lyric_start"],
        "_lyric_end":       feats["lyric_end"],
    }
