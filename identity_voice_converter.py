import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import os

SR = 22050
N_MFCC = 13
HOP = 512
N_FFT = 2048


# ============================
# LOAD ALL IDENTITY PROFILES
# ============================

def load_profiles(csv_path):

    df = pd.read_csv(csv_path)

    profiles = {}

    for _, row in df.iterrows():
        singer = row["singer"]
        profiles[singer] = row

    return profiles


# ============================
# EXTRACT SONG FEATURES
# ============================

def extract_song_features(audio_path):

    y, sr = librosa.load(audio_path, sr=SR)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        hop_length=HOP
    )

    return y, sr, mfcc


# ============================
# APPLY IDENTITY SHIFT
# ============================

def apply_identity_shift(mfcc, source_profile, target_profile):

    transformed = mfcc.copy()

    for i in range(N_MFCC):

        src_mean = source_profile[f"mfcc{i+1}_mean"]
        tgt_mean = target_profile[f"mfcc{i+1}_mean"]

        shift = tgt_mean - src_mean

        transformed[i] = mfcc[i] + shift

    return transformed


# ============================
# RECONSTRUCT AUDIO
# ============================

def reconstruct_audio(mfcc):

    mel = librosa.feature.inverse.mfcc_to_mel(mfcc)

    stft = librosa.feature.inverse.mel_to_stft(mel)

    y = librosa.griffinlim(stft, hop_length=HOP)

    return y


# ============================
# CONVERT TO MULTIPLE SINGERS
# ============================

def convert_to_all_targets(input_song, identity_csv, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    print("Loading identity profiles...")
    profiles = load_profiles(identity_csv)

    source_profile = profiles["arr"]

    print("Extracting input song features...")
    y, sr, mfcc = extract_song_features(input_song)

    targets = ["spb", "mano", "ilayaraja"]

    for target in targets:

        print(f"\nConverting ARR → {target}")

        target_profile = profiles[target]

        transformed_mfcc = apply_identity_shift(
            mfcc,
            source_profile,
            target_profile
        )

        converted_audio = reconstruct_audio(transformed_mfcc)

        output_path = os.path.join(
            output_folder,
            f"arr_to_{target}.wav"
        )

        sf.write(output_path, converted_audio, SR)

        print("Saved:", output_path)


# ============================
# MAIN
# ============================

if __name__ == "__main__":

    input_song = "dataset/reference/arr_ref.wav"

    identity_csv = "CSVs/IDENTITY_PROFILES/ALL_SINGER_IDENTITY_PROFILES.csv"

    output_folder = "dataset/identity_converted"

    convert_to_all_targets(
        input_song,
        identity_csv,
        output_folder
    )