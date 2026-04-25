import os
import glob
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import pyworld as pw
from scipy.signal import savgol_filter
from scipy.linalg import sqrtm

# ===============================
# CONFIGURATION
# ===============================
SR = 22050
CSV_PROFILE_PATH = "../CSVs/IDENTITY_PROFILES/ALL_SINGER_IDENTITY_PROFILES.csv"
SEED_VC_OUT_DIR = "../dataset/converted"
REFERENCE_DIR = "../dataset/reference"
REFINED_OUT_DIR = "refined_outputs"
MATRICES_DIR = "MATRICES"
os.makedirs(REFINED_OUT_DIR, exist_ok=True)
os.makedirs(MATRICES_DIR, exist_ok=True)

SINGER_FILES = {
    "arr": "arr_ref.wav",
    "ilayaraja": "ilayaraja_ref.wav",
    "mano": "mano_ref.wav",
    "spb": "spb_ref.wav"
}

# ===============================
# N x N MATRIX TRANSFORMATIONS
# ===============================

def compute_transformation_matrix(source_features, target_features, epsilon=1e-5):
    """
    Computes an N x N Whitening and Coloring Matrix to transform
    source feature vectors to the target feature covariance space.
    """
    mu_s = np.mean(source_features, axis=0)
    mu_t = np.mean(target_features, axis=0)
    
    # Center the data
    s_centered = source_features - mu_s
    t_centered = target_features - mu_t
    
    # Covariance matrices (N x N)
    cov_s = np.cov(s_centered.T) + np.eye(s_centered.shape[1]) * epsilon
    cov_t = np.cov(t_centered.T) + np.eye(t_centered.shape[1]) * epsilon
    
    # Whitening and Coloring (using Matrix Square Root)
    inv_sqrt_cov_s = np.real(sqrtm(np.linalg.inv(cov_s)))
    sqrt_cov_t = np.real(sqrtm(cov_t))
    
    # Transformation Matrix W (N x N)
    W = sqrt_cov_t @ inv_sqrt_cov_s
    
    return W, mu_s, mu_t, cov_s, cov_t

def apply_transformation(features, W, mu_s, mu_t):
    """ Applies the N x N learned projection """
    features_centered = features - mu_s
    transformed = (W @ features_centered.T).T + mu_t
    return transformed

# ===============================
# F0 FREQUENCY / RAGA SMOOTHING
# ===============================

def smooth_f0(f0, window_length=15, polyorder=3):
    """
    Applies Savitzky-Golay smoothing on the continuous F0 contour 
    to remove abrupt synthesizer artifacts while keeping raga gamakas.
    """
    if len(f0) < window_length:
        return f0
    
    # Extract voiced regions
    voiced_idx = np.where(f0 > 0)[0]
    if len(voiced_idx) < window_length:
        return f0
        
    f0_continuous = np.copy(f0)
    # Interpolate unvoiced regions to create a continuous contour for smoothing
    unvoiced_idx = np.where(f0 == 0)[0]
    if len(voiced_idx) > 0 and len(unvoiced_idx) > 0:
        f0_continuous[unvoiced_idx] = np.interp(unvoiced_idx, voiced_idx, f0[voiced_idx])
        
    # Apply polynomial smoothing
    smoothed = savgol_filter(f0_continuous, window_length, polyorder)
    
    # Restore unvoiced regions
    smoothed[f0 == 0] = 0
    return np.maximum(smoothed, 0)

def match_f0_statistics(f0, target_mean, target_std):
    """ Vector operation to map F0 to the exact tessitura of the target singer """
    voiced_idx = np.where(f0 > 0)[0]
    if len(voiced_idx) == 0:
        return f0
        
    log_f0 = np.log2(f0[voiced_idx])
    source_mean = np.mean(log_f0)
    source_std = np.std(log_f0)
    
    if source_std == 0: return f0
        
    mapped_log_f0 = (log_f0 - source_mean) / source_std * target_std + target_mean
    f0_mapped = np.copy(f0)
    f0_mapped[voiced_idx] = np.power(2, mapped_log_f0)
    
    return f0_mapped

# ===============================
# MAIN PIPELINE
# ===============================

def process_audio(target_singer):
    print(f"\n--- Refining {target_singer.upper()} ---")
    
    seed_vc_audio = os.path.join(SEED_VC_OUT_DIR, f"{target_singer}.wav")
    ref_audio = os.path.join(REFERENCE_DIR, SINGER_FILES[target_singer])
    
    if not os.path.exists(seed_vc_audio) or not os.path.exists(ref_audio):
        print(f"Skipping {target_singer}, files missing.")
        return

    # Load stats from CSV
    profiles = pd.read_csv(CSV_PROFILE_PATH)
    target_profile = profiles[profiles['singer'] == target_singer].iloc[0]
    
    # Load audio
    print("[1] Loading WORLD Vocoder...")
    x, _ = librosa.load(seed_vc_audio, sr=SR, dtype=np.float64)
    y, _ = librosa.load(ref_audio, sr=SR, dtype=np.float64)
    
    # 1. WORLD Feature Extraction
    print("[2] Extracting frame-level acoustic structures...")
    f0_x, t_x = pw.harvest(x, SR)
    sp_x = pw.cheaptrick(x, f0_x, t_x, SR)
    ap_x = pw.d4c(x, f0_x, t_x, SR)
    
    f0_y, t_y = pw.harvest(y, SR)
    sp_y = pw.cheaptrick(y, f0_y, t_y, SR)
    
    # 2. Convert 513-bin Spectral Envelope to 64-band Mel scale spanning entire spectrum
    DIM = 64
    mel_basis = librosa.filters.mel(sr=SR, n_fft=1024, n_mels=DIM)
    
    x_mel = (mel_basis @ sp_x.T).T
    y_mel = (mel_basis @ sp_y.T).T
    
    # Use log compression for safe matrix mapping
    x_mel_log = np.log(np.maximum(x_mel, 1e-10))
    y_mel_log = np.log(np.maximum(y_mel, 1e-10))
    
    # 3. N x N Matrix Transformation construction
    print(f"[3] Constructing {DIM} x {DIM} Whitening/Coloring transformation matrices...")
    W, mu_s, mu_t, cov_s, cov_t = compute_transformation_matrix(x_mel_log, y_mel_log)
    
    pd.DataFrame(cov_s).to_csv(os.path.join(MATRICES_DIR, f"{target_singer}_SeedVC_Covariance_{DIM}x{DIM}.csv"), index=False)
    pd.DataFrame(cov_t).to_csv(os.path.join(MATRICES_DIR, f"{target_singer}_Target_Covariance_{DIM}x{DIM}.csv"), index=False)
    pd.DataFrame(W).to_csv(os.path.join(MATRICES_DIR, f"{target_singer}_Transformation_Matrix_W_{DIM}x{DIM}.csv"), index=False)
    
    # Apply vector projections in the continuous Mel log-space
    print(f"    -> Applying vector projection onto target '{target_singer}' space")
    x_mel_log_projected = apply_transformation(x_mel_log, W, mu_s, mu_t)
    
    # Inverse Log and Inverse Mel
    x_mel_projected = np.exp(x_mel_log_projected)
    pinv_mel_basis = np.linalg.pinv(mel_basis)
    sp_refined = (pinv_mel_basis @ x_mel_projected.T).T
    
    # SAFETY CLAMP: Prevent extreme volume distortion/tearing (max 4.0x, min 0.25x scaling per bin)
    sp_refined = np.maximum(sp_refined, 1e-10)
    sp_ratio = sp_refined / np.maximum(sp_x, 1e-10)
    sp_ratio_clamped = np.clip(sp_ratio, 0.25, 4.0)
    sp_refined_clamped = sp_x * sp_ratio_clamped
    
    # 4. F0 Raga / Frequency Correction (Histogram mapping from CSV)
    print(f"[4] Correcting Pitch tessitura to exact {target_singer} constraints...")
    # target_log_mean and std from CSV. Wait, CSV has f0_mean in Hz. We'll approximate.
    target_f0_mean_log2 = np.log2(target_profile['f0_mean'])
    target_f0_std_log2 = target_profile['f0_std'] / (target_profile['f0_mean'] * np.log(2)) # Taylor approx
    
    f0_refined = match_f0_statistics(f0_x, target_f0_mean_log2, target_f0_std_log2)
    
    # 5. F0 Smoothing 
    print(f"    -> Applying raga smoothing (Savitzky-Golay)...")
    f0_smoothed = smooth_f0(f0_refined, window_length=21, polyorder=3)
    
    # 6. Final Resynthesis (using our safe clamped spectrum)
    print(f"[5] Re-synthesizing refined audio...")
    y_refined = pw.synthesize(f0_smoothed, np.ascontiguousarray(sp_refined_clamped), ap_x, SR, pw.default_frame_period)
    
    # Save output
    out_path = os.path.join(REFINED_OUT_DIR, f"{target_singer}_refined.wav")
    sf.write(out_path, y_refined.astype(np.float32), SR)
    print(f"    -> SAVED successfully to {out_path}\n")

if __name__ == "__main__":
    for singer in SINGER_FILES.keys():
        process_audio(singer)
