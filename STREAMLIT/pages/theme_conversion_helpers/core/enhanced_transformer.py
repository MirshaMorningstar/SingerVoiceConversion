"""
ENHANCED SIGNAL PROCESSOR
==========================
Applies additional mathematical transformations ON TOP of the existing
raga_transformer.py output. Does NOT replace or modify raga_transformer.py.

Enhancements applied:
  1. Formant Shifting     — shifts vocal resonance (timbre) per emotion
                            using spectral envelope manipulation
  2. Harmonic Emphasis    — boosts harmonics aligned with the target raga's
                            vadi (king) note frequency
  3. Spectral Tilt        — reshapes frequency balance per emotion
                            (bright/sharp for Happy, dark/warm for Sad)
  4. Energy Envelope      — reshapes amplitude dynamics per emotion
                            (punchy attacks for Angry, soft fades for Peaceful)
  5. Vibrato Modulation   — adds/reduces pitch micro-modulation matching
                            the gamaka character of the target raga

All processing is done in the frequency domain using numpy + scipy.
Output is blended with original converted signal (blend ratio configurable).
"""

import numpy as np
from scipy.signal import butter, sosfilt, lfilter
from scipy.fft    import rfft, irfft, rfftfreq

# ── Emotion-specific processing parameters ────────────────────────────────────

EMOTION_PARAMS = {
    "Happy": {
        "formant_shift_ratio": 1.06,      # slight upward formant shift — bright
        "spectral_tilt_db":    +3.0,       # boost highs
        "tilt_cutoff_hz":      2000,
        "attack_sharpness":    1.2,        # punchier transients
        "vibrato_depth_st":    0.10,       # subtle vibrato
        "vibrato_rate_hz":     5.5,
        "harmonic_boost_db":   2.0,
        "blend":               0.40,       # 40% enhanced, 60% existing output
    },
    "Sad": {
        "formant_shift_ratio": 0.94,       # downward formant shift — darker
        "spectral_tilt_db":    -4.0,       # cut highs, boost lows
        "tilt_cutoff_hz":      1500,
        "attack_sharpness":    0.7,        # softer transients
        "vibrato_depth_st":    0.18,       # slow, deeper vibrato (andolita)
        "vibrato_rate_hz":     3.8,
        "harmonic_boost_db":   1.5,
        "blend":               0.45,
    },
    "Angry": {
        "formant_shift_ratio": 1.08,
        "spectral_tilt_db":    +5.0,       # very bright, harsh edge
        "tilt_cutoff_hz":      1800,
        "attack_sharpness":    1.6,        # very punchy
        "vibrato_depth_st":    0.05,       # minimal vibrato — tense, direct
        "vibrato_rate_hz":     7.0,
        "harmonic_boost_db":   3.0,
        "blend":               0.45,
    },
    "Fearful": {
        "formant_shift_ratio": 1.03,
        "spectral_tilt_db":    +1.5,
        "tilt_cutoff_hz":      2500,
        "attack_sharpness":    1.4,        # irregular, startled feel
        "vibrato_depth_st":    0.22,       # trembling vibrato
        "vibrato_rate_hz":     6.5,
        "harmonic_boost_db":   1.0,
        "blend":               0.35,
    },
    "Peaceful": {
        "formant_shift_ratio": 0.98,
        "spectral_tilt_db":    -2.0,       # warm, gentle
        "tilt_cutoff_hz":      3000,
        "attack_sharpness":    0.5,        # very soft transients
        "vibrato_depth_st":    0.08,
        "vibrato_rate_hz":     4.5,
        "harmonic_boost_db":   1.0,
        "blend":               0.35,
    },
    "Romantic": {
        "formant_shift_ratio": 0.97,
        "spectral_tilt_db":    -1.5,       # warm mid-range
        "tilt_cutoff_hz":      2200,
        "attack_sharpness":    0.8,
        "vibrato_depth_st":    0.15,       # expressive vibrato
        "vibrato_rate_hz":     4.8,
        "harmonic_boost_db":   2.0,
        "blend":               0.40,
    },
    "Surprised": {
        "formant_shift_ratio": 1.10,       # sharp upward shift
        "spectral_tilt_db":    +4.0,
        "tilt_cutoff_hz":      2000,
        "attack_sharpness":    1.8,        # very sharp attacks
        "vibrato_depth_st":    0.25,       # wide, sudden
        "vibrato_rate_hz":     6.0,
        "harmonic_boost_db":   2.5,
        "blend":               0.40,
    },
    "Disgusted": {
        "formant_shift_ratio": 0.92,       # heavy, low formant
        "spectral_tilt_db":    -3.0,
        "tilt_cutoff_hz":      1200,
        "attack_sharpness":    0.9,
        "vibrato_depth_st":    0.06,       # flat, unexpressive
        "vibrato_rate_hz":     3.0,
        "harmonic_boost_db":   0.5,
        "blend":               0.35,
    },
}

# Vadi note semitone for each raga (for harmonic emphasis)
RAGA_VADI_SEMITONE = {
    "Mohanam":        4,    # Ga3
    "Bhairavi":       5,    # Ma1
    "Dhanyasi":       3,    # G2
    "Todi":           8,    # Dha1
    "Kambhoji":       7,    # Pa
    "Kalyani":        6,    # Ma2
    "Hamsadhwani":    4,    # Ga3
    "Kharaharapriya": 7,    # Pa
}


# ── Core DSP functions ─────────────────────────────────────────────────────────

def _spectral_tilt(y, sr, tilt_db, cutoff_hz):
    """
    Boost or cut frequencies above cutoff_hz by tilt_db.
    Positive tilt_db = brighter. Negative = warmer/darker.
    """
    N        = len(y)
    Y        = rfft(y)
    freqs    = rfftfreq(N, d=1.0/sr)
    gain_lin = 10 ** (tilt_db / 20.0)

    mask = np.ones(len(freqs))
    mask[freqs > cutoff_hz] = gain_lin
    # Smooth transition around cutoff
    trans = (freqs > cutoff_hz * 0.8) & (freqs <= cutoff_hz)
    if trans.any():
        t = (freqs[trans] - cutoff_hz * 0.8) / (cutoff_hz * 0.2)
        mask[trans] = 1.0 + (gain_lin - 1.0) * t

    Y_out = Y * mask
    return irfft(Y_out, n=N).astype(np.float32)


def _formant_shift(y, sr, ratio):
    """
    Shift spectral envelope (formants) by ratio using cepstral method.
    ratio > 1 = brighter/higher formants
    ratio < 1 = darker/lower formants
    Does not change pitch — only timbre.
    """
    N    = len(y)
    Y    = rfft(y)
    mag  = np.abs(Y)
    ph   = np.angle(Y)

    # Log magnitude → cepstrum → lifter → back
    log_mag  = np.log(mag + 1e-9)
    ceps     = np.fft.irfft(log_mag)

    # Lifter: keep low quefrency (envelope) only
    lifter_len = int(sr * 0.002)   # ~2ms
    lifter      = np.zeros_like(ceps)
    lifter[:lifter_len] = ceps[:lifter_len]

    env_log = np.fft.rfft(lifter).real
    env_log = env_log[:len(mag)]

    # Stretch/compress the envelope in frequency
    old_idx = np.arange(len(env_log))
    new_idx = old_idx / ratio
    new_idx = np.clip(new_idx, 0, len(env_log) - 1)
    env_shifted = np.interp(old_idx, new_idx, env_log)

    # Reconstruct
    residual = log_mag - env_log
    new_log  = env_shifted + residual
    new_mag  = np.exp(new_log)
    Y_out    = new_mag * np.exp(1j * ph)

    return irfft(Y_out, n=N).astype(np.float32)


def _harmonic_emphasis(y, sr, raga_name, boost_db):
    """
    Boost frequency bands around the harmonics of the raga's vadi note.
    This makes the vadi note resonate more strongly in the output.
    """
    vadi_st   = RAGA_VADI_SEMITONE.get(raga_name, 4)
    sa_hz     = 261.63   # C4 as tonic reference
    vadi_hz   = sa_hz * (2 ** (vadi_st / 12.0))

    N         = len(y)
    Y         = rfft(y)
    freqs     = rfftfreq(N, d=1.0/sr)
    boost_lin = 10 ** (boost_db / 20.0)
    mask      = np.ones(len(freqs))

    # Boost first 6 harmonics of the vadi note
    for h in range(1, 7):
        target_hz = vadi_hz * h
        bw        = target_hz * 0.04   # 4% bandwidth around each harmonic
        band      = np.abs(freqs - target_hz) < bw
        # Smooth gaussian-like boost
        dist      = np.abs(freqs - target_hz)
        gaussian  = np.exp(-0.5 * (dist / (bw * 0.5)) ** 2)
        mask     += gaussian * (boost_lin - 1.0)

    Y_out = Y * mask
    return irfft(Y_out, n=N).astype(np.float32)


def _reshape_dynamics(y, attack_sharpness):
    """
    Reshape amplitude envelope.
    attack_sharpness > 1 = punchier transients (Angry, Surprised)
    attack_sharpness < 1 = softer, smoother envelope (Peaceful, Sad)
    """
    # Compute short-time RMS envelope
    frame = 512
    hop   = 256
    n_frames = (len(y) - frame) // hop + 1

    env = np.array([
        np.sqrt(np.mean(y[i*hop : i*hop+frame]**2) + 1e-9)
        for i in range(n_frames)
    ])

    # Apply power curve to envelope
    env_shaped = np.power(env, attack_sharpness)
    ratio      = env_shaped / (env + 1e-9)

    # Interpolate ratio back to sample level
    sample_idx  = np.arange(len(y))
    frame_times = np.arange(n_frames) * hop + frame // 2
    frame_times = np.clip(frame_times, 0, len(y) - 1)
    ratio_full  = np.interp(sample_idx, frame_times, ratio)

    return (y * ratio_full).astype(np.float32)


def _add_vibrato(y, sr, depth_st, rate_hz):
    """
    Apply pitch micro-modulation (vibrato) via sinusoidal delay modulation.
    depth_st  = vibrato depth in semitones
    rate_hz   = vibrato rate in Hz
    """
    if depth_st < 0.01:
        return y

    max_delay_samples = int(sr * 0.025)   # max 25ms delay buffer
    t       = np.arange(len(y)) / sr
    mod     = depth_st * np.sin(2 * np.pi * rate_hz * t)

    # Convert semitone modulation to sample delay
    delay_samples = (max_delay_samples * mod / 2.0).astype(np.float32)
    out     = np.zeros_like(y)

    for i in range(len(y)):
        d     = delay_samples[i]
        idx   = i - d
        i0    = int(np.floor(idx))
        i1    = i0 + 1
        frac  = idx - i0
        if 0 <= i0 < len(y) and 0 <= i1 < len(y):
            out[i] = y[i0] * (1 - frac) + y[i1] * frac
        elif 0 <= i0 < len(y):
            out[i] = y[i0]

    return out.astype(np.float32)


# ── Main enhancement entry point ───────────────────────────────────────────────

def enhance(y_converted, sr, target_emotion, target_raga):
    """
    Apply all signal processing enhancements to the converted audio.

    Parameters
    ----------
    y_converted   : np.ndarray  — output from raga_transformer.convert_song
    sr            : int         — sample rate
    target_emotion: str         — e.g. "Sad"
    target_raga   : str         — e.g. "Bhairavi"

    Returns
    -------
    y_enhanced    : np.ndarray  — enhanced audio (same length as input)
    enhancement_log : dict      — what was applied and at what strength
    """
    params = EMOTION_PARAMS.get(target_emotion, EMOTION_PARAMS["Happy"])
    y      = y_converted.astype(np.float32).copy()

    # Normalise input
    peak = np.max(np.abs(y)) + 1e-9
    y   /= peak

    steps_applied = []

    # 1. Spectral tilt
    try:
        y = _spectral_tilt(y, sr,
                           tilt_db    = params["spectral_tilt_db"],
                           cutoff_hz  = params["tilt_cutoff_hz"])
        steps_applied.append(f"Spectral tilt {params['spectral_tilt_db']:+.1f} dB @ {params['tilt_cutoff_hz']} Hz")
    except Exception as e:
        steps_applied.append(f"Spectral tilt SKIPPED ({e})")

    # 2. Formant shift
    try:
        y = _formant_shift(y, sr, ratio=params["formant_shift_ratio"])
        steps_applied.append(f"Formant shift ×{params['formant_shift_ratio']:.2f}")
    except Exception as e:
        steps_applied.append(f"Formant shift SKIPPED ({e})")

    # 3. Harmonic emphasis on vadi note
    try:
        y = _harmonic_emphasis(y, sr,
                               raga_name = target_raga,
                               boost_db  = params["harmonic_boost_db"])
        steps_applied.append(f"Harmonic emphasis +{params['harmonic_boost_db']:.1f} dB on {target_raga} vadi harmonics")
    except Exception as e:
        steps_applied.append(f"Harmonic emphasis SKIPPED ({e})")

    # 4. Dynamics reshaping
    try:
        y = _reshape_dynamics(y, attack_sharpness=params["attack_sharpness"])
        steps_applied.append(f"Dynamics reshape (attack sharpness ×{params['attack_sharpness']:.1f})")
    except Exception as e:
        steps_applied.append(f"Dynamics reshape SKIPPED ({e})")

    # 5. Vibrato modulation
    try:
        y = _add_vibrato(y, sr,
                         depth_st = params["vibrato_depth_st"],
                         rate_hz  = params["vibrato_rate_hz"])
        steps_applied.append(f"Vibrato {params['vibrato_depth_st']:.2f} st @ {params['vibrato_rate_hz']:.1f} Hz")
    except Exception as e:
        steps_applied.append(f"Vibrato SKIPPED ({e})")

    # Blend enhanced with original converted (preserves raga transformation)
    blend   = params["blend"]
    y_final = blend * y + (1.0 - blend) * (y_converted / peak)

    # Renormalise to original peak
    y_final = y_final / (np.max(np.abs(y_final)) + 1e-9) * peak * 0.95
    y_final = np.clip(y_final, -1.0, 1.0).astype(np.float32)

    enhancement_log = {
        "emotion":            target_emotion,
        "raga":               target_raga,
        "blend_ratio":        f"{int(blend*100)}% enhanced + {int((1-blend)*100)}% base",
        "spectral_tilt_db":   params["spectral_tilt_db"],
        "formant_shift_ratio":params["formant_shift_ratio"],
        "harmonic_boost_db":  params["harmonic_boost_db"],
        "attack_sharpness":   params["attack_sharpness"],
        "vibrato_depth_st":   params["vibrato_depth_st"],
        "vibrato_rate_hz":    params["vibrato_rate_hz"],
        "steps_applied":      steps_applied,
    }

    return y_final, enhancement_log