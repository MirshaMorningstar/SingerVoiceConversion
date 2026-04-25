"""
RAGA TRANSFORMER  v12 — Best of Doc21 + v11
=============================================

What worked in Doc21 (kept):
  - Frame-by-frame swara substitution (shift only detected notes)
  - _make_sad_bgm / _make_happy_bgm (process original BGM, not external audio)
  - _spectral_gate (removes Demucs vocal bleed)
  - _phrase_silences (breathing gaps for sad)
  - Learned shift map from reference JSON profiles

What v11 added (kept):
  - Time-varying envelope (emotion builds gradually, not flat)
  - audio-separator → Demucs → HPSS fallback chain
  - No external audio mixing — only uploaded song used

Fixed from Doc21:
  - `key_shift` removed — no global semitone transpose
  - `tempo` set to 1.0 — no tempo change (per brief)
  - _make_sad_bgm now also has _make_happy_bgm counterpart
  - BGM processing uses original song's BGM (never external files)
"""

import os, json, tempfile
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt, sosfilt, resample_poly
from core.raga_knowledge_base import RAGA_SWARAS, get_note_changes, get_raga_for_emotion

# ── Emotion profiles ──────────────────────────────────────────────────────────
# tempo=1.0 and key_shift=0 — per brief (no global tempo/pitch change)
EMOTION_PROFILES = {
    "Sad":      {"brightness":0.45, "low_mid":0.50, "reverb":0.55,
                 "silence":True,  "meend":0.32, "meend_rate":3.2,
                 "bgm_level":0.55, "bgm_brightness":0.22},
    "Happy":    {"brightness":1.15, "low_mid":0.00, "reverb":0.00,
                 "silence":False, "meend":0.04, "meend_rate":5.5,
                 "bgm_level":0.58, "bgm_brightness":1.15},
    "Peaceful": {"brightness":0.75, "low_mid":0.22, "reverb":0.25,
                 "silence":True,  "meend":0.18, "meend_rate":3.0,
                 "bgm_level":0.35, "bgm_brightness":0.65},
    "Romantic": {"brightness":0.85, "low_mid":0.18, "reverb":0.18,
                 "silence":True,  "meend":0.20, "meend_rate":4.0,
                 "bgm_level":0.32, "bgm_brightness":0.78},
    "Fearful":  {"brightness":0.42, "low_mid":0.38, "reverb":0.38,
                 "silence":True,  "meend":0.40, "meend_rate":5.0,
                 "bgm_level":0.22, "bgm_brightness":0.28},
    "Surprised":{"brightness":1.20, "low_mid":0.00, "reverb":0.00,
                 "silence":False, "meend":0.04, "meend_rate":6.2,
                 "bgm_level":0.52, "bgm_brightness":1.20},
    "Angry":    {"brightness":1.30, "low_mid":0.00, "reverb":0.00,
                 "silence":False, "meend":0.03, "meend_rate":6.8,
                 "bgm_level":0.62, "bgm_brightness":1.35},
    "Disgusted":{"brightness":0.85, "low_mid":0.10, "reverb":0.10,
                 "silence":False, "meend":0.10, "meend_rate":4.5,
                 "bgm_level":0.38, "bgm_brightness":0.82},
}

RAGA_SEMITONES = {
    "Mohanam":        {0,2,4,7,9},
    "Bhairavi":       {0,2,3,5,7,8,10},
    "Dhanyasi":       {0,3,5,7,10},
    "Todi":           {0,1,3,6,7,8,11},
    "Kambhoji":       {0,2,4,5,7,9},
    "Kalyani":        {0,2,4,6,7,9,11},
    "Hamsadhwani":    {0,4,7,11},
    "Kharaharapriya": {0,2,3,5,7,9,10},
}
EMOTION_RAGA = {
    "Happy":"Mohanam",      "Sad":"Bhairavi",
    "Angry":"Dhanyasi",     "Fearful":"Todi",
    "Disgusted":"Kambhoji", "Surprised":"Kalyani",
    "Peaceful":"Hamsadhwani","Romantic":"Kharaharapriya",
}
EMOTION_VALENCE = {
    "Happy":+1,"Surprised":+1,"Peaceful":+1,
    "Sad":-1,"Angry":-1,"Fearful":-1,"Disgusted":-1,"Romantic":0,
}


# ══════════════════════════════════════════════════════════════════════════════
# VOCAL SEPARATION — uploaded song only, no external audio
# ══════════════════════════════════════════════════════════════════════════════

def separate_vocals_demucs(audio_path, sr):
    """
    Separation priority: audio-separator → Demucs → HPSS.
    Returns (vocals, bgm) — both from the uploaded file only.
    """
    # Try audio-separator (no PyTorch dependency — works even with DLL issues)
    try:
        from audio_separator.separator import Separator
        import soundfile as _sf2
        with tempfile.TemporaryDirectory() as _td:
            sep = Separator(output_dir=_td, output_format="WAV",
                           normalization=0.9, mdx_enable_denoise=True)
            sep.load_model("Kim_Vocal_2.onnx")
            files = sep.separate(audio_path)
            vf = next((f for f in files
                       if "Vocals" in f or "vocal" in f.lower()), None)
            if vf:
                v, vsr = _sf2.read(vf, dtype="float32")
                if v.ndim == 2: v = v.mean(axis=1)
                if vsr != sr: v = resample_poly(v, sr, vsr)
                y_full, _ = librosa.load(audio_path, sr=sr, mono=True)
                n = min(len(v), len(y_full))
                bgm = (y_full[:n] - v[:n]).astype(np.float32)
                print("[transformer] Separated via audio-separator")
                return v[:n].astype(np.float32), bgm
    except Exception as e:
        print(f"[transformer] audio-separator failed: {e}")

    # Try Demucs
    try:
        import torch
        import scipy.signal as sps
        from demucs.pretrained import get_model
        from demucs.apply import apply_model

        model = get_model("htdemucs"); model.eval()
        model_sr = model.samplerate

        wav_np, file_sr = sf.read(audio_path, always_2d=True)
        wav_np = wav_np.T
        if wav_np.shape[0] == 1: wav_np = np.repeat(wav_np, 2, axis=0)
        wav_np = wav_np[:2]
        if file_sr != model_sr:
            n_out  = int(wav_np.shape[1] * model_sr / file_sr)
            wav_np = np.stack([sps.resample(wav_np[c], n_out) for c in range(2)])

        wav = torch.from_numpy(wav_np.astype(np.float32))
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / (ref.std() + 1e-8)

        with torch.no_grad():
            sources = apply_model(model, wav.unsqueeze(0), progress=False)[0]

        snames  = list(model.sources)
        vi      = snames.index("vocals")
        vocals  = sources[vi].mean(0).numpy()
        bgm_arr = sum(sources[i].mean(0) for i in range(len(snames)) if i != vi).numpy()

        def _rs(a, f, t):
            return sps.resample(a, int(len(a)*t/f)) if f != t else a

        vocals  = _rs(vocals,  model_sr, sr).astype(np.float32)
        bgm_arr = _rs(bgm_arr, model_sr, sr).astype(np.float32)
        print("[transformer] Separated via Demucs")
        return vocals, bgm_arr
    except Exception as e:
        print(f"[transformer] Demucs failed: {e}")

    # HPSS fallback
    print("[transformer] Using HPSS fallback")
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    D = librosa.stft(y, n_fft=2048, hop_length=512)
    H, P = librosa.decompose.hpss(D, margin=3.0)
    vocals = librosa.istft(H, hop_length=512, length=len(y))
    bgm    = librosa.istft(P, hop_length=512, length=len(y))
    return vocals.astype(np.float32), bgm.astype(np.float32)


def detect_vocal_section(y, sr, duration=60):
    chunk = int(sr * 5)
    n     = len(y) // chunk
    if n == 0: return y[:int(sr*duration)]
    rms   = [float(np.sqrt(np.mean(y[i*chunk:(i+1)*chunk]**2))) for i in range(n)]
    avg   = np.mean(rms)
    vocal_start = 0
    for i in range(len(rms)-1):
        if rms[i] >= avg*0.85 and rms[i+1] >= avg*0.85:
            vocal_start = i; break
    start = vocal_start * chunk
    end   = min(start + int(sr*duration), len(y))
    if (end-start) < int(sr*20):
        start = max(0, len(y)-int(sr*duration)); end = len(y)
    return y[start:end]


# ══════════════════════════════════════════════════════════════════════════════
# VOCAL PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def _spectral_gate(y, sr, threshold_db=-45):
    """Remove Demucs vocal bleed — frames below threshold faded out."""
    hop = int(sr * 0.02)
    n   = len(y) // hop
    out = y.copy()
    thr = 10 ** (threshold_db / 20.0)
    for i in range(n):
        s0, s1 = i*hop, min((i+1)*hop, len(y))
        rms = float(np.sqrt(np.mean(y[s0:s1]**2) + 1e-12))
        if rms < thr:
            out[s0:s1] *= (rms / (thr + 1e-12)) ** 0.5
    return out


def _swara_substitution(y_vocals, sr, tonic, src_set, tgt_set,
                        learned_shift_map=None):
    """
    Frame-by-frame pitch detection → shift ONLY detected notes.
    Raga theory + learned shifts from reference JSON profiles.
    Does NOT globally shift all frequencies — only target notes.
    """
    hop = 512
    pitches, mags = librosa.piptrack(y=y_vocals, sr=sr, hop_length=hop,
                                     fmin=80, fmax=1400, threshold=0.08)
    n      = pitches.shape[1]
    shifts = np.zeros(n)

    for t in range(n):
        col = mags[:, t]
        if col.max() < 1e-4: continue
        freq = pitches[col.argmax(), t]
        if freq < 80 or freq > 1400: continue
        midi  = librosa.hz_to_midi(freq)
        rel   = int(round(midi - tonic)) % 12

        theory_shift = 0.0
        if rel in src_set and rel not in tgt_set:
            nearest = min(tgt_set, key=lambda x: min(abs(x-rel), 12-abs(x-rel)))
            sh = nearest - rel
            if sh > 6:  sh -= 12
            if sh < -6: sh += 12
            theory_shift = float(sh)

        final_shift = theory_shift
        if learned_shift_map and rel in learned_shift_map:
            lsh    = float(learned_shift_map[rel]["shift"])
            weight = float(learned_shift_map[rel]["weight"])
            final_shift = theory_shift*(1-weight*0.6) + lsh*(weight*0.6)

        shifts[t] = final_shift

    # Apply shifts segment by segment
    y_out = np.zeros_like(y_vocals)
    seg_start, prev_sh = 0, shifts[0]
    for t in range(1, n+1):
        curr = shifts[t] if t < n else None
        if curr != prev_sh or t == n:
            s0  = seg_start * hop
            s1  = min(t*hop, len(y_vocals))
            seg = y_vocals[s0:s1]
            if abs(prev_sh) > 0.1 and len(seg) >= 512:
                seg = librosa.effects.pitch_shift(
                    seg, sr=sr, n_steps=float(prev_sh)
                )
            end = min(s0+len(seg), len(y_out))
            y_out[s0:end] = seg[:end-s0]
            seg_start, prev_sh = t, curr
    return y_out


def _meend(y, sr, depth, rate):
    """Sinusoidal pitch modulation — andolita gamaka for sad."""
    if depth < 0.01: return y
    t   = np.arange(len(y)) / sr
    sl  = depth * (2*(t*rate - np.floor(t*rate+0.5)))
    idx = np.clip((np.arange(len(y)) + sl*sr/(rate*8)).astype(int), 0, len(y)-1)
    return y[idx]


def _phrase_silences(y, sr):
    """Add breathing gaps between phrases — key sad quality."""
    hop = int(sr*0.25)
    n   = len(y) // hop
    rms = np.array([float(np.sqrt(np.mean(y[i*hop:min((i+1)*hop,len(y))]**2)+1e-9)) for i in range(n)])
    avg = float(np.mean(rms))
    out = y.copy(); last = -20
    for i in range(2, n-2):
        last += 1
        if (rms[i] < rms[i-1]*0.72 and rms[i] < rms[i+1]*0.72
                and rms[i] < avg*0.78 and last >= 14):
            s0 = i*hop
            fl = min(int(sr*0.65), len(out)-s0)
            out[s0:s0+fl] *= np.linspace(1.0, 0.01, fl)
            sb = s0+fl
            bl = min(int(sr*0.12), len(out)-sb)
            if bl > 0: out[sb:sb+bl] *= np.linspace(0.01, 1.0, bl)
            last = 0
    return out


def _spectral_darken_vocals(y, sr, brightness, low_mid):
    """EQ vocals for target emotion timbre."""
    nyq = sr / 2.0
    if brightness < 1.0:
        cutoff = max(2200, 5500*brightness)
        b, a   = butter(3, min(cutoff/nyq, 0.99), btype='low')
        y_low  = filtfilt(b, a, y)
        b2, a2 = butter(3, min(cutoff/nyq, 0.99), btype='high')
        y      = y_low + filtfilt(b2, a2, y) * brightness
    elif brightness > 1.0:
        b, a = butter(2, min(2500/nyq, 0.99), btype='high')
        y    = y + filtfilt(b, a, y) * (brightness-1.0)
    if low_mid > 0.01:
        b, a = butter(2, [200/nyq, 900/nyq], btype='band')
        y    = y + low_mid * filtfilt(b, a, y)
    return np.clip(y, -0.92, 0.92)


def _add_reverb(y, sr, amount):
    """Multi-tap reverb for sad spacious feel."""
    if amount < 0.01: return y
    out  = y.copy().astype(np.float64)
    taps = [
        (int(sr*0.025), 0.32*amount), (int(sr*0.055), 0.22*amount),
        (int(sr*0.090), 0.16*amount), (int(sr*0.140), 0.10*amount),
    ]
    for delay, gain in taps:
        if delay < len(out): out[delay:] += y[:-delay] * gain
    td = int(sr*0.28)
    if td < len(out):
        tail = np.zeros_like(out); tail[td:] = out[:-td] * 0.38*amount
        for _ in range(3):
            if td < len(tail): tail[td:] += tail[:-td] * 0.15
        out = out + tail
    peak = np.max(np.abs(out))
    if peak > 0.90: out = out * (0.88/peak)
    return out.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# BGM PROCESSING — from original song only
# ══════════════════════════════════════════════════════════════════════════════

def _make_sad_bgm(bgm, sr, brightness, bgm_level):
    """
    Transform uploaded song's own BGM into sad instrumental.
    Heavy low-pass + cello boost + presence cut + reverb.
    No external audio — only the uploaded song's BGM.
    """
    nyq = sr / 2.0
    # Heavy low-pass
    cutoff = max(900, 4500*brightness)
    b, a   = butter(4, min(cutoff/nyq, 0.99), btype='low')
    bgm    = filtfilt(b, a, bgm)
    # Cello range boost (220-580Hz)
    b2, a2 = butter(3, [220/nyq, 580/nyq], btype='band')
    bgm    = bgm + 0.60 * filtfilt(b2, a2, bgm)
    # Sub-bass warmth
    b3, a3 = butter(2, [60/nyq, 120/nyq], btype='band')
    bgm    = bgm + 0.30 * filtfilt(b3, a3, bgm)
    # Presence cut (removes happy brightness)
    b4, a4 = butter(2, [2000/nyq, min(5000/nyq, 0.99)], btype='band')
    bgm    = bgm - 0.45 * filtfilt(b4, a4, bgm)
    # Reverb
    bgm = _add_reverb(bgm, sr, amount=0.65)
    bgm = np.clip(bgm, -0.92, 0.92)
    peak = np.max(np.abs(bgm)) + 1e-9
    return (bgm * (bgm_level / peak)).astype(np.float32)


def _make_happy_bgm(bgm, sr, brightness, bgm_level):
    """
    Transform uploaded song's own BGM into happy instrumental.
    High-shelf boost + presence boost + transient sharpening.
    """
    nyq = sr / 2.0
    # Boost presence (2kHz-8kHz)
    b, a = butter(3, [2000/nyq, min(8000/nyq, 0.99)], btype='band')
    bgm  = bgm + (brightness-1.0) * 0.5 * filtfilt(b, a, bgm)
    # Air boost (>8kHz)
    b2, a2 = butter(2, min(8000/nyq, 0.99), btype='high')
    bgm    = bgm + (brightness-1.0) * 0.3 * filtfilt(b2, a2, bgm)
    # Slight sub-bass reduction (less weight = lighter feel)
    b3, a3 = butter(2, min(120/nyq, 0.99), btype='low')
    bgm    = bgm - 0.20 * filtfilt(b3, a3, bgm)
    bgm = np.clip(bgm, -0.92, 0.92)
    peak = np.max(np.abs(bgm)) + 1e-9
    return (bgm * (bgm_level / peak)).astype(np.float32)


def _process_bgm(bgm, sr, target_emotion, brightness, bgm_level):
    """Route BGM processing based on emotion valence."""
    val = EMOTION_VALENCE.get(target_emotion, 0)
    if val < 0:
        return _make_sad_bgm(bgm, sr, brightness, bgm_level)
    elif val > 0:
        return _make_happy_bgm(bgm, sr, brightness, bgm_level)
    else:
        # Neutral — light processing only
        nyq  = sr/2.0
        b, a = butter(2, min(5000/nyq, 0.99), btype='low')
        bgm  = filtfilt(b, a, bgm)
        peak = np.max(np.abs(bgm)) + 1e-9
        return (bgm * bgm_level / peak).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# TIME-VARYING ENVELOPE (from v11)
# ══════════════════════════════════════════════════════════════════════════════

def _emotion_envelope(n_samples, sr, target_emotion):
    """
    Per-sample strength multiplier that varies over the song.
    Sad: starts at 0.35, builds to 0.92 by 60% through song.
    Happy: starts at 0.5, builds to 1.0 by end.
    Makes emotional shift feel natural, not mechanical.
    """
    t = np.linspace(0, 1, n_samples)
    if EMOTION_VALENCE.get(target_emotion, 0) < 0:
        env = 0.35 + 0.65 * np.clip((t - 0.1) / 0.5, 0, 1)
        env = np.where(t > 0.6, np.minimum(env, 0.92), env)
    elif EMOTION_VALENCE.get(target_emotion, 0) > 0:
        env = 0.50 + 0.50 * np.clip(t / 0.7, 0, 1)
    else:
        env = np.ones(n_samples) * 0.75
    return env.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# REFERENCE PROFILES — JSON only, never audio
# ══════════════════════════════════════════════════════════════════════════════

def _load_ref_profile(target_emotion):
    """12-element pitch weight array from JSON — used for note weighting only."""
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base, "..", "references", target_emotion),
        os.path.join(base, "references", target_emotion),
        os.path.join(os.getcwd(), "references", target_emotion),
    ]
    rdir = next((c for c in candidates if os.path.isdir(c)), None)
    if not rdir: return None
    profiles = []
    for fn in os.listdir(rdir):
        if not fn.endswith(".json"): continue
        try:
            with open(os.path.join(rdir, fn)) as f: d = json.load(f)
            if "profile" in d:
                p = np.array(d["profile"], dtype=np.float32)
                if len(p)==12:
                    p = np.roll(p, -d.get("tonic",0))
                    p /= (p.max()+1e-9)
                    profiles.append(p)
        except Exception: continue
    return np.mean(profiles, axis=0) if profiles else None


def _normalize(y, db=-14.0):
    rms = float(np.sqrt(np.mean(y**2))) + 1e-9
    return np.clip(y * (10**(db/20.0)/rms), -0.95, 0.95)


def _match_length(a, b):
    n = min(len(a), len(b))
    return a[:n], b[:n]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

def convert_song(audio_path, target_emotion, source_raga,
                 tonic, y, sr, output_path):
    """
    Pipeline:
    1. Separate vocals + BGM from uploaded song (audio-separator / Demucs / HPSS)
    2. Load reference JSON profile for note weighting (not audio)
    3. Gate vocals (remove Demucs bleed)
    4. Frame-by-frame swara substitution on vocals
    5. Meend (gamaka) on vocals
    6. Spectral darken/brighten vocals
    7. Reverb on vocals (sad only)
    8. Phrase silences (sad only)
    9. Process BGM from same uploaded song
    10. Mix with time-varying envelope
    11. Normalise
    """
    target_raga = EMOTION_RAGA.get(target_emotion, get_raga_for_emotion(target_emotion))
    src_notes   = {int(s)%12 for s in RAGA_SWARAS.get(source_raga,{}).get("semitones",[]) if isinstance(s,(int,float))}
    tgt_notes   = {int(s)%12 for s in RAGA_SWARAS.get(target_raga,{}).get("semitones",[]) if isinstance(s,(int,float))}

    if y.ndim == 2: y = y.mean(axis=1)
    y = y.astype(np.float32)
    y /= (np.max(np.abs(y)) + 1e-9)

    p = EMOTION_PROFILES.get(target_emotion, EMOTION_PROFILES["Happy"])

    # ── 1. Separate vocals + BGM from uploaded song ────────────────────────
    print(f"[transformer] Separating vocals...")
    vocals, bgm = separate_vocals_demucs(audio_path, sr)

    # Align lengths
    n = min(len(vocals), len(bgm), len(y))
    vocals = vocals[:n]; bgm = bgm[:n]

    # ── 2. Reference JSON profile — note weighting only ────────────────────
    ref_profile = _load_ref_profile(target_emotion)

    # Learned shift map from reference library
    learned_map = {}
    try:
        from core.reference_library import get_shift_map_for_conversion
        _, learned_map = get_shift_map_for_conversion(vocals, sr, target_emotion)
        if learned_map:
            print(f"[transformer] {len(learned_map)} learned shifts for {target_emotion}")
    except Exception: pass

    # ── 3. Detect tonic + gate vocals ─────────────────────────────────────
    chroma = librosa.feature.chroma_cqt(y=vocals, sr=sr)
    tonic  = int(np.argmax(np.bincount(np.argmax(chroma, axis=0), minlength=12)))
    vocals = _spectral_gate(vocals, sr)

    # ── 4. Swara substitution — frame-by-frame ────────────────────────────
    print(f"[transformer] Swara substitution {source_raga}→{target_raga}...")
    vocals_out = _swara_substitution(
        vocals, sr, tonic, src_notes, tgt_notes, learned_map
    )

    # ── 5. Meend ──────────────────────────────────────────────────────────
    vocals_out = _meend(vocals_out, sr, p["meend"], p["meend_rate"])

    # ── 6. Spectral reshape vocals ────────────────────────────────────────
    vocals_out = _spectral_darken_vocals(vocals_out, sr, p["brightness"], p["low_mid"])

    # ── 7. Reverb on vocals ───────────────────────────────────────────────
    vocals_out = _add_reverb(vocals_out, sr, p["reverb"])

    # ── 8. Phrase silences ────────────────────────────────────────────────
    if p["silence"]:
        vocals_out = _phrase_silences(vocals_out, sr)

    # ── 9. Process BGM from same uploaded song ────────────────────────────
    print(f"[transformer] Processing BGM for {target_emotion}...")
    bgm_out = _process_bgm(bgm, sr, target_emotion, p["bgm_brightness"], p["bgm_level"])

    # ── 10. Time-varying mix ──────────────────────────────────────────────
    vocals_out, bgm_out = _match_length(vocals_out, bgm_out)
    n_out = len(vocals_out)

    env = _emotion_envelope(n_out, sr, target_emotion)

    # Normalise vocals to -16 dBFS
    vocals_norm = _normalize(vocals_out, db=-16.0)

    # Apply time-varying emotional strength to BGM processing
    # At env=0.35 (start of sad song): BGM processing is lighter
    # At env=0.92 (mid song): full sad BGM processing
    bgm_raw = bgm[:n_out].astype(np.float32)
    bgm_raw /= (np.max(np.abs(bgm_raw)) + 1e-9)
    bgm_processed = bgm_out * env + bgm_raw * (1.0 - env) * p["bgm_level"]

    out = vocals_norm + bgm_processed
    out = np.clip(out, -0.95, 0.95)

    # ── 11. Normalise ──────────────────────────────────────────────────────
    out = _normalize(out, db=-14.0)

    sf.write(output_path, out, sr)
    print(f"[transformer] Done → {output_path}")

    changes = get_note_changes(source_raga, target_raga)
    return {
        "notes_removed":     changes.get("removed", {}),
        "notes_added":       changes.get("added", {}),
        "notes_shared":      changes.get("shared", {}),
        "pitch_shift_st":    0.0,
        "tempo_factor":      1.0,
        "shift_map":         {},
        "source_raga":       source_raga,
        "target_raga":       target_raga,
        "separation_method": "audio-separator/demucs/hpss",
        "ref_profiles_used": 1 if ref_profile is not None else 0,
    }


def detect_vocal_section(y, sr, duration=60):
    return y[:int(duration * sr)]


def _sep_demucs(audio_path, sr):
    v, _ = separate_vocals_demucs(audio_path, sr)
    return v 