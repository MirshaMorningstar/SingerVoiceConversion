import os
import numpy as np
import pandas as pd
import librosa
import parselmouth
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks

# ===============================
# CONFIG
# ===============================
SR = 22050
HOP_LENGTH = 512
N_FFT = 2048

BASE_DATASET = "dataset/reference"
OUTPUT_DIR = "CSVs/IDENTITY_PROFILES"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===============================
# UTILITY FUNCTIONS
# ===============================

def remove_nan(x):
    return x[~np.isnan(x)]

def statistical_features(arr, prefix):
    arr = remove_nan(arr)
    features = {}
    if len(arr) == 0:
        return {f"{prefix}_{k}": 0 for k in 
                ["mean","std","min","max","range",
                 "p5","p95","iqr","skew","kurt"]}

    features[f"{prefix}_mean"] = np.mean(arr)
    features[f"{prefix}_std"] = np.std(arr)
    features[f"{prefix}_min"] = np.min(arr)
    features[f"{prefix}_max"] = np.max(arr)
    features[f"{prefix}_range"] = np.max(arr) - np.min(arr)
    features[f"{prefix}_p5"] = np.percentile(arr, 5)
    features[f"{prefix}_p95"] = np.percentile(arr, 95)
    features[f"{prefix}_iqr"] = np.percentile(arr, 75) - np.percentile(arr, 25)
    features[f"{prefix}_skew"] = skew(arr)
    features[f"{prefix}_kurt"] = kurtosis(arr)
    return features


# ===============================
# PITCH & VIBRATO FEATURES
# ===============================

def pitch_features(y, sr):
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=75, fmax=500,
        sr=sr, hop_length=HOP_LENGTH
    )

    voiced_f0 = remove_nan(f0)

    features = statistical_features(voiced_f0, "f0")

    # voiced frame ratio
    features["voiced_ratio"] = np.sum(~np.isnan(f0)) / len(f0)

    # pitch stability
    if len(voiced_f0) > 1:
        features["pitch_stability"] = np.mean(np.abs(np.diff(voiced_f0)))
    else:
        features["pitch_stability"] = 0

    # Vibrato estimation
    if len(voiced_f0) > 10:
        f0_detrended = voiced_f0 - np.mean(voiced_f0)
        peaks, _ = find_peaks(f0_detrended)
        duration = len(voiced_f0) * HOP_LENGTH / sr
        vibrato_rate = len(peaks) / duration if duration > 0 else 0
        vibrato_depth = np.std(f0_detrended)
    else:
        vibrato_rate = 0
        vibrato_depth = 0

    features["vibrato_rate"] = vibrato_rate
    features["vibrato_depth"] = vibrato_depth
    features["f0_modulation_variance"] = np.var(voiced_f0)

    return features


# ===============================
# FORMANT FEATURES (PRAAT)
# ===============================

def formant_features(audio_path):
    snd = parselmouth.Sound(audio_path)
    formant = snd.to_formant_burg()

    times = np.arange(0, snd.duration, 0.01)

    f1, f2, f3 = [], [], []

    for t in times:
        f1.append(formant.get_value_at_time(1, t))
        f2.append(formant.get_value_at_time(2, t))
        f3.append(formant.get_value_at_time(3, t))

    f1 = np.array(f1)
    f2 = np.array(f2)
    f3 = np.array(f3)

    features = {}
    features.update(statistical_features(f1, "F1"))
    features.update(statistical_features(f2, "F2"))
    features.update(statistical_features(f3, "F3"))

    # distances
    features["F1_F2_distance_mean"] = np.mean(remove_nan(f2 - f1))
    features["F2_F3_distance_mean"] = np.mean(remove_nan(f3 - f2))

    return features


# ===============================
# VOICE QUALITY (PRAAT)
# ===============================

def voice_quality_features(audio_path):
    snd = parselmouth.Sound(audio_path)

    try:
        pulses = parselmouth.praat.call(
            snd, "To PointProcess (periodic, cc)", 75, 500)

        jitter = parselmouth.praat.call(
            pulses, "Get jitter (local)", 0, 0, 75, 500, 1.3)

        shimmer = parselmouth.praat.call(
            [snd, pulses], "Get shimmer (local)",
            0, 0, 75, 500, 1.3, 1.6)

        if jitter is None or np.isnan(jitter):
            jitter = 0

        if shimmer is None or np.isnan(shimmer):
            shimmer = 0

    except:
        jitter = 0
        shimmer = 0

    try:
        harmonicity = snd.to_harmonicity()
        hnr = parselmouth.praat.call(
            harmonicity, "Get mean", 0, 0)

        if hnr is None or np.isnan(hnr):
            hnr = 0
    except:
        hnr = 0

    return {
        "jitter_mean": float(jitter),
        "shimmer_mean": float(shimmer),
        "HNR_mean": float(hnr)
    }

# ===============================
# SPECTRAL FEATURES
# ===============================

def spectral_features(y, sr):
    features = {}

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    for name, feat in zip(
        ["centroid","bandwidth","rolloff","flatness"],
        [centroid, bandwidth, rolloff, flatness]):

        clean_feat = feat.flatten()
        clean_feat = clean_feat[clean_feat > 0]  # remove zero frames
        features.update(statistical_features(clean_feat, name))

    # spectral contrast per band
    for i in range(contrast.shape[0]):
        features.update(statistical_features(
            contrast[i], f"contrast_band{i+1}"
        ))

    return features


# ===============================
# MFCC FEATURES
# ===============================

def mfcc_features(y, sr):
    features = {}
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13)

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    for i in range(13):
        features.update(statistical_features(
            mfcc[i], f"mfcc{i+1}"
        ))

        features[f"mfcc{i+1}_delta_mean"] = float(np.nan_to_num(np.mean(delta[i])))
        features[f"mfcc{i+1}_delta2_mean"] = float(np.nan_to_num(np.mean(delta2[i])))

    return features


# ===============================
# MASTER EXTRACTOR
# ===============================

def extract_identity_features(audio_path):
    print(f"Processing {audio_path}")

    y, sr = librosa.load(audio_path, sr=SR, mono=True)

    # Remove silence (very important)
    intervals = librosa.effects.split(y, top_db=30)
    if len(intervals) > 0:
        y = np.concatenate([y[start:end] for start, end in intervals])

    features = {}

    features.update(pitch_features(y, sr))
    features.update(formant_features(audio_path))
    features.update(voice_quality_features(audio_path))
    features.update(spectral_features(y, sr))
    features.update(mfcc_features(y, sr))

    # Final safety cleanup (no NaN allowed in identity vector)
    for key in features:
        if features[key] is None or np.isnan(features[key]):
            features[key] = 0.0

    return features


# ===============================
# PROCESS ALL SINGERS
# ===============================

def build_identity_profiles():
    files = {
        "arr": "arr_ref.wav",
        "ilayaraja": "ilayaraja_ref.wav",
        "mano": "mano_ref.wav",
        "spb": "spb_ref.wav"
    }

    all_profiles = []

    for singer, filename in files.items():
        path = os.path.join(BASE_DATASET, filename)

        features = extract_identity_features(path)
        features["singer"] = singer

        df = pd.DataFrame([features])
        df.to_csv(os.path.join(
            OUTPUT_DIR, f"{singer}_identity.csv"), index=False)

        all_profiles.append(features)

    # combined file
    combined_df = pd.DataFrame(all_profiles)
    combined_df.to_csv(
        os.path.join(OUTPUT_DIR, "ALL_SINGER_IDENTITY_PROFILES.csv"),
        index=False
    )

    print("\nIdentity profiles saved successfully.")


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    build_identity_profiles()