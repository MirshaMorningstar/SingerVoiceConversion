import librosa
import soundfile as sf
import numpy as np
import os

def merge_audio(vocal_conv, instrumental, workspace):

    # Load both stems
    v, sr = librosa.load(vocal_conv, sr=44100)
    i, _ = librosa.load(instrumental, sr=44100)

    # Match lengths
    length = min(len(v), len(i))
    v = v[:length]
    i = i[:length]

    # --------------------------------------------------
    # KEEP VOCAL FULL
    # Reduce instrumental to 90% loudness (your request)
    # --------------------------------------------------
    instrumental_gain = 0.9
    i_scaled = i * instrumental_gain

    # Sum signals (true remix, not crossfade)
    final = v + i_scaled

    # --------------------------------------------------
    # Prevent clipping (important!)
    # --------------------------------------------------
    peak = np.max(np.abs(final))
    if peak > 1.0:
        final = final / peak * 0.98  # gentle normalization

    # Save output
    final_path = os.path.join(workspace, "outputs", "final_mix.wav")
    sf.write(final_path, final, sr)

    return final_path

