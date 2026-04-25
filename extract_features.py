import librosa
import numpy as np
import pandas as pd
import parselmouth
from scipy.stats import skew, kurtosis
import os

# Create CSV output folder
OUTPUT_DIR = "CSVs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------- PRAAT BASED FEATURES (Jitter, Shimmer, HNR) -----------
def voice_quality_features(audio_path):
    snd = parselmouth.Sound(audio_path)

    pitch = snd.to_pitch()
    pulses = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)

    jitter = parselmouth.praat.call(pulses, "Get jitter (local)", 0, 0, 75, 500, 1.3)
    shimmer = parselmouth.praat.call([snd, pulses], "Get shimmer (local)", 0, 0, 75, 500, 1.3, 1.6)

    harmonicity = snd.to_harmonicity()
    hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)

    return jitter, shimmer, hnr


# ----------- LOAD AUDIO -----------
def load_audio(path):
    y, sr = librosa.load(path, sr=22050, mono=True)
    return y, sr


# ----------- PITCH FEATURES -----------
def pitch_features(y, sr):
    f0, _, _ = librosa.pyin(y, fmin=75, fmax=500)
    f0 = f0[~np.isnan(f0)]

    return {
        "f0_mean": np.mean(f0),
        "f0_std": np.std(f0),
        "f0_min": np.min(f0),
        "f0_max": np.max(f0)
    }


# ----------- MFCC FEATURES -----------
def mfcc_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    features = {}
    for i in range(13):
        features[f"mfcc{i+1}_mean"] = np.mean(mfcc[i])
        features[f"mfcc{i+1}_std"] = np.std(mfcc[i])

    return features


# ----------- SPECTRAL FEATURES -----------
def spectral_features(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)

    return {
        "centroid_mean": np.mean(centroid),
        "bandwidth_mean": np.mean(bandwidth),
        "rolloff_mean": np.mean(rolloff),
        "flatness_mean": np.mean(flatness)
    }


# ----------- ENERGY FEATURES -----------
def energy_features(y):
    rms = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)

    return {
        "rms_mean": np.mean(rms),
        "rms_std": np.std(rms),
        "zcr_mean": np.mean(zcr)
    }


# ----------- MASTER FUNCTION -----------
def extract_all_features(audio_path):
    y, sr = load_audio(audio_path)

    features = {}
    features.update(pitch_features(y, sr))
    features.update(mfcc_features(y, sr))
    features.update(spectral_features(y, sr))
    features.update(energy_features(y))

    jitter, shimmer, hnr = voice_quality_features(audio_path)
    features["hnr"] = hnr

    return features


# ----------- PROCESS MULTIPLE FILES -----------
def process_singer(name, audio_path):
    features = extract_all_features(audio_path)
    df = pd.DataFrame([features])
    output_path = os.path.join(OUTPUT_DIR, f"{name}_features.csv")
    df.to_csv(output_path, index=False)

    print(f"Saved {name}_features.csv")


if __name__ == "__main__":

    files = {
        "arr_ref": "dataset/reference/arr_ref.wav",
        "arr_conv": "dataset/converted/arr.wav",

        "spb_ref": "dataset/reference/spb_ref.wav",
        "spb_conv": "dataset/converted/spb.wav",

        "mano_ref": "dataset/reference/mano_ref.wav",
        "mano_conv": "dataset/converted/mano.wav",

        "ilai_ref": "dataset/reference/ilayaraja_ref.wav",
        "ilai_conv": "dataset/converted/ilayaraja.wav",
    }

    for name, path in files.items():
        process_singer(name, path)
