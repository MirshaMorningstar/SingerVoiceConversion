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
    # Load at 16kHz mono — 40% faster than 22050Hz, sufficient for chroma
    y_full, sr = librosa.load(audio_path, mono=True, sr=16000)

    # Use only 60s max — no need for full song
    MAX_SEC = 60
    if len(y_full) > MAX_SEC * sr:
        # Take 30s–90s window (lyrical section), capped at 60s
        start = min(int(30 * sr), len(y_full) // 3)
        end   = min(start + MAX_SEC * sr, len(y_full))
    else:
        start = 0
        end   = len(y_full)
    y = y_full[start:end]

    # chroma_stft is 3–4× faster than chroma_cqt, good enough for raga ID
    hop = 2048   # larger hop = fewer frames = faster
    chroma      = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop, n_fft=4096)
    mean_chroma = np.mean(chroma, axis=1)
    mean_chroma /= (mean_chroma.max() + 1e-9)

    tonic_idx = int(np.argmax(mean_chroma))
    rolled    = np.roll(mean_chroma, -tonic_idx)

    dominant       = [i for i in range(12) if rolled[i] > 0.25]
    dominant_names = [SEMITONE_TO_SWARA[i] for i in dominant]

    # Beat tracking on short segment — fast
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
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


# ── Fast note sequence — frame by frame swara timeline ───────────────────────

SEMITONE_SHORT = {
    0:"Sa", 1:"Ri1", 2:"Ri2", 3:"Ri3",
    4:"Ga3", 5:"Ma1", 6:"Ma2", 7:"Pa",
    8:"Da1", 9:"Da2", 10:"Ni2", 11:"Ni3",
}
SEMITONE_SOLFEGE = {
    0:"Sa", 1:"Ri", 2:"Ri", 3:"Ri",
    4:"Ga", 5:"Ma", 6:"Ma", 7:"Pa",
    8:"Da", 9:"Da", 10:"Ni", 11:"Ni",
}
SEMITONE_FULL_NAME = {
    0:"Sa (Shadjam)", 1:"Ri1 (Shuddha Rishabham)", 2:"Ri2 (Chatushruti Rishabham)",
    3:"Ri3 (Shatshruti Rishabham)", 4:"Ga3 (Antara Gandharam)", 5:"Ma1 (Shuddha Madhyamam)",
    6:"Ma2 (Prati Madhyamam)", 7:"Pa (Panchamam)", 8:"Da1 (Shuddha Dhaivatam)",
    9:"Da2 (Chatushruti Dhaivatam)", 10:"Ni2 (Kaisika Nishadam)", 11:"Ni3 (Kakali Nishadam)",
}

# Reference note patterns per emotion
EMOTION_NOTE_PATTERNS = {
    "Happy":    {"raga":"Mohanam",        "scale":"Sa – Ga – Pa – Da – Ni",           "semitones":[0,4,7,9,11],      "character":"Bright ascending pentatonic — no Ri or Ma",         "typical_seq":["Sa","Ga","Pa","Da","Ni","Sa","Pa","Ga","Sa","Ga","Pa","Da","Ni","Pa","Ga","Sa"]},
    "Sad":      {"raga":"Bhairavi",        "scale":"Sa – Ri1 – G2 – Ma1 – Pa – Da1 – Ni2","semitones":[0,1,3,5,7,8,10],"character":"Heavy descending — flat Ga, Ni create grief",          "typical_seq":["Sa","Ni","Da","Pa","Ma","Ga","Ri","Sa","Ri","Ga","Ma","Pa","Da","Ni","Sa","Pa"]},
    "Angry":    {"raga":"Dhanyasi",        "scale":"Sa – G2 – Ma1 – Pa – Ni2",         "semitones":[0,3,5,7,10],      "character":"Tense minor pentatonic — forceful and direct",        "typical_seq":["Sa","Ga","Ma","Pa","Ni","Sa","Pa","Ma","Ga","Sa","Ni","Pa","Ma","Ga","Ma","Pa"]},
    "Fearful":  {"raga":"Todi",            "scale":"Sa – Ri1 – G2 – Ma2 – Pa – Da1 – Ni3","semitones":[0,1,3,6,7,8,11],"character":"Augmented Ma2 creates eerie anxious tension",         "typical_seq":["Sa","Ri","Ga","Ma","Pa","Da","Ni","Sa","Ni","Da","Pa","Ma","Ga","Ri","Sa","Ga"]},
    "Peaceful": {"raga":"Hamsadhwani",     "scale":"Sa – Ga3 – Pa – Ni3",              "semitones":[0,4,7,11],        "character":"Sparse 4-note — serene, open space between notes",    "typical_seq":["Sa","Ga","Pa","Ni","Sa","Ni","Pa","Ga","Sa","Pa","Ni","Sa","Ga","Pa","Ga","Sa"]},
    "Romantic": {"raga":"Kharaharapriya",  "scale":"Sa – Ri2 – G2 – Ma1 – Pa – Da2 – Ni2","semitones":[0,2,3,5,7,9,10],"character":"Natural minor — intimate, longing, expressive",        "typical_seq":["Sa","Ri","Ga","Ma","Pa","Da","Ni","Sa","Ni","Da","Pa","Ma","Ri","Ga","Sa","Pa"]},
    "Surprised":{"raga":"Kalyani",         "scale":"Sa – Ri2 – Ga3 – Ma2 – Pa – Da2 – Ni3","semitones":[0,2,4,6,7,9,11],"character":"Raised Ma2 Lydian — bright, wonder, sudden uplift",   "typical_seq":["Sa","Ri","Ga","Ma","Pa","Da","Ni","Sa","Ni","Sa","Pa","Ma","Ga","Ri","Sa","Ga"]},
    "Disgusted":{"raga":"Kambhoji",        "scale":"Sa – Ri2 – Ga3 – Ma1 – Pa – Da2",  "semitones":[0,2,4,5,7,9],    "character":"Ni absent in ascent — heavy, unresolved, flat",      "typical_seq":["Sa","Ri","Ga","Ma","Pa","Da","Pa","Ma","Ga","Ri","Sa","Ma","Pa","Da","Pa","Ga"]},
}


def extract_note_sequence(audio_path: str, duration_sec: int = 60) -> dict:
    """
    Fast frame-by-frame swara extraction using chroma (not piptrack).
    Each frame → dominant semitone → swara name.
    Groups into one note per second for clean display.
    Runtime: ~3-5 seconds for 60s audio.
    """
    # Load at 16kHz — fast
    y, sr = librosa.load(audio_path, mono=True, sr=16000,
                         duration=duration_sec)

    hop     = 1024   # ~64ms per frame at 16kHz — good time resolution
    chroma  = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop, n_fft=2048)
    times   = librosa.frames_to_time(np.arange(chroma.shape[1]),
                                     sr=sr, hop_length=hop)

    # Detect tonic
    mean_chr  = np.mean(chroma, axis=1)
    mean_chr /= (mean_chr.max() + 1e-9)
    tonic_idx = int(np.argmax(mean_chr))

    # Per-frame: dominant semitone relative to tonic
    frame_semitones = []
    for t in range(chroma.shape[1]):
        col = chroma[:, t]
        if col.max() < 0.15:          # silence / too weak
            frame_semitones.append(None)
        else:
            raw = int(np.argmax(col))
            rel = (raw - tonic_idx) % 12
            frame_semitones.append(rel)

    # One dominant note per second
    per_second = []
    total_secs = int(times[-1]) + 1 if len(times) else 0
    for sec in range(min(total_secs, duration_sec)):
        notes_in_sec = [
            frame_semitones[k]
            for k, t in enumerate(times)
            if sec <= t < sec + 1 and frame_semitones[k] is not None
        ]
        if notes_in_sec:
            dominant = max(set(notes_in_sec), key=notes_in_sec.count)
            per_second.append({
                "second":   sec,
                "time":     f"{sec//60}:{sec%60:02d}",
                "semitone": dominant,
                "note":     SEMITONE_SHORT.get(dominant, str(dominant)),
                "solfege":  SEMITONE_SOLFEGE.get(dominant, "–"),
                "full_name":SEMITONE_FULL_NAME.get(dominant, "–"),
            })
        else:
            per_second.append({
                "second":sec, "time":f"{sec//60}:{sec%60:02d}",
                "semitone":None, "note":"–", "solfege":"–", "full_name":"(silence)",
            })

    # Consecutive note events (group same notes)
    note_events = []
    prev = None
    for row in per_second:
        if row["solfege"] != "–" and row["solfege"] != prev:
            note_events.append(row["solfege"])
            prev = row["solfege"]

    # Histogram
    hist = {}
    for row in per_second:
        n = row["note"]
        if n != "–":
            hist[n] = hist.get(n, 0) + 1

    return {
        "per_second":      per_second,
        "note_events":     note_events,       # consecutive unique notes
        "note_histogram":  hist,
        "tonic_semitone":  tonic_idx,
        "tonic_name":      SEMITONE_FULL_NAME.get(tonic_idx, str(tonic_idx)),
        "total_notes":     len(note_events),
        "duration_analysed": round(float(times[-1]), 1) if len(times) else 0,
    }


def extract_swara_profile(audio_path: str) -> dict:
    """Lightweight swara energy profile — uses same fast chroma."""
    feats   = extract_features(audio_path)
    matches = identify_raga(feats["chroma_vector"])
    chroma  = feats["chroma_vector"]

    per_semitone = []
    for i in range(12):
        energy = float(chroma[i])
        bar    = int(energy * 20)
        per_semitone.append({
            "semitone":   i,
            "name":       SEMITONE_TO_SWARA[i],
            "full_name":  SEMITONE_FULL_NAME[i],
            "energy":     round(energy, 4),
            "energy_pct": round(energy * 100, 1),
            "present":    energy > 0.25,
            "strong":     energy > 0.50,
            "bar":        bar,
        })

    SWARA_GROUPS = {
        "Sa":[0], "Ri":[1,2,3], "Ga":[2,3,4],
        "Ma":[5,6], "Pa":[7], "Da":[8,9,10], "Ni":[9,10,11],
    }
    swara_groups = {}
    for grp, semitones in SWARA_GROUPS.items():
        energies = [float(chroma[s]) for s in semitones]
        best_idx = int(np.argmax(energies))
        swara_groups[grp] = {
            "max_energy":   round(max(energies), 4),
            "energy_pct":   round(max(energies)*100, 1),
            "best_variant": SEMITONE_FULL_NAME[semitones[best_idx]],
            "present":      max(energies) > 0.25,
        }

    return {
        "per_semitone":    per_semitone,
        "swara_groups":    swara_groups,
        "tonic_note":      SEMITONE_FULL_NAME.get(feats["tonic_idx"], "–"),
        "top_swaras":      sorted(per_semitone, key=lambda x: x["energy"], reverse=True)[:5],
        "chroma_vector":   chroma.tolist(),
        "all_raga_scores": matches,
        "tempo_bpm":       feats["tempo_bpm"],
        "duration_sec":    feats["duration_sec"],
    }

def extract_advanced_acoustics(audio_path: str, duration_sec: int = 60) -> dict:
    """
    Extracts advanced mathematical and DSP features from the audio.
    Used for the mathematical report in the Streamlit UI.
    """
    y, sr = librosa.load(audio_path, mono=True, sr=16000, duration=duration_sec)
    
    rms = librosa.feature.rms(y=y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    flatness = librosa.feature.spectral_flatness(y=y)
    
    return {
        "RMS Energy (Mean)": f"{float(np.mean(rms)):.4f}",
        "Spectral Centroid (Mean Hz)": f"{float(np.mean(centroid)):.2f}",
        "Spectral Bandwidth (Mean Hz)": f"{float(np.mean(bandwidth)):.2f}",
        "Spectral Rolloff (Mean Hz)": f"{float(np.mean(rolloff)):.2f}",
        "Zero Crossing Rate (Mean)": f"{float(np.mean(zcr)):.4f}",
        "Spectral Flatness (Mean)": f"{float(np.mean(flatness)):.6f}",
        "Estimated SNR (dB)": f"{float(10 * np.log10(np.mean(rms) / (np.min(rms) + 1e-9))):.2f}"
    }