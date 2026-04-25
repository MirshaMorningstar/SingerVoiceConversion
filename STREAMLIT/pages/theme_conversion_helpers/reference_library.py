"""
REFERENCE LIBRARY
=================
Learns real pitch shift patterns from actual Tamil songs.

How it works:
  1. User uploads a sad Tamil song as reference
  2. We extract its pitch histogram (which notes are dominant, how strong)
  3. We compare against a happy song's pitch histogram
  4. The difference = real-world note shift map
  5. This shift map is saved permanently and used during conversion

The more reference songs added, the more accurate the emotion conversion.
This is the transfer-learning approach — learning from real songs
not just raga theory.
"""

import os
import json
import numpy as np
import librosa

REFS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "references")


def _ensure_dirs():
    for emotion in ["Sad","Happy","Angry","Fearful","Peaceful","Romantic","Surprised","Disgusted"]:
        os.makedirs(os.path.join(REFS_DIR, emotion), exist_ok=True)


def extract_pitch_profile(y, sr):
    """
    Extract a 12-bin pitch class histogram from audio.
    Returns normalized array of shape (12,) — one value per semitone.
    Also returns dominant notes (those above 0.25 threshold).
    """
    # Use CQT chroma — more accurate than STFT for pitch
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=48, n_octaves=6)

    # Weight by energy — louder frames count more
    rms    = librosa.feature.rms(y=y, hop_length=512)[0]
    rms    = rms[:chroma.shape[1]]
    weight = rms / (rms.sum() + 1e-9)

    # Weighted histogram
    profile = np.dot(chroma, weight)
    profile = profile / (profile.max() + 1e-9)

    # Dominant notes — above 25% of max
    dominant = [int(i) for i in np.where(profile > 0.25)[0]]

    return {
        "profile":   profile.tolist(),
        "dominant":  dominant,
        "tonic":     int(np.argmax(profile)),
    }


def save_reference(y, sr, song_name, emotion):
    """
    Save a reference song's pitch profile permanently.
    Called once when user uploads a reference song.
    """
    _ensure_dirs()
    profile_data = extract_pitch_profile(y, sr)
    profile_data["song_name"] = song_name
    profile_data["emotion"]   = emotion
    profile_data["sr"]        = sr

    # Save as JSON — no audio stored, just the pitch profile
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in song_name)
    out_path  = os.path.join(REFS_DIR, emotion, f"{safe_name}.json")
    with open(out_path, "w") as f:
        json.dump(profile_data, f, indent=2)

    return out_path


def load_references(emotion):
    """Load all saved reference profiles for a given emotion."""
    _ensure_dirs()
    refs  = []
    edir  = os.path.join(REFS_DIR, emotion)
    for fname in os.listdir(edir):
        if fname.endswith(".json"):
            with open(os.path.join(edir, fname)) as f:
                refs.append(json.load(f))
    return refs


def list_all_references():
    """Returns dict of emotion → list of song names."""
    _ensure_dirs()
    result = {}
    for emotion in os.listdir(REFS_DIR):
        edir = os.path.join(REFS_DIR, emotion)
        if os.path.isdir(edir):
            songs = []
            for fname in os.listdir(edir):
                if fname.endswith(".json"):
                    with open(os.path.join(edir, fname)) as f:
                        data = json.load(f)
                        songs.append(data.get("song_name", fname))
            if songs:
                result[emotion] = songs
    return result


def delete_reference(emotion, song_name):
    """Delete a saved reference."""
    edir     = os.path.join(REFS_DIR, emotion)
    safe     = "".join(c if c.isalnum() or c in "-_" else "_" for c in song_name)
    fpath    = os.path.join(edir, f"{safe}.json")
    if os.path.exists(fpath):
        os.remove(fpath)
        return True
    return False


def compute_learned_shift_map(source_profile, target_emotion):
    """
    Core function — computes note shift map by comparing:
      - source song's actual pitch profile
      - average pitch profile of all reference songs for target emotion

    Returns dict: {semitone: shift_amount} for notes that need shifting.
    Shift amount is in semitones (float).

    This is how it works:
      - For each semitone in source, find the closest matching semitone
        in the target emotion's average profile
      - If a note is strong in source but weak in target → needs to shift
        toward the nearest strong note in target
      - Shift magnitude is weighted by how dominant the note is
    """
    refs = load_references(target_emotion)
    if not refs:
        return {}  # No references — fall back to raga theory only

    # Average all reference profiles for this emotion
    avg_profile = np.zeros(12)
    for ref in refs:
        p = np.array(ref["profile"])
        # Align to same tonic before averaging
        tonic_shift = ref["tonic"]
        p_aligned   = np.roll(p, -tonic_shift)
        avg_profile += p_aligned
    avg_profile /= len(refs)
    avg_profile /= (avg_profile.max() + 1e-9)

    # Align source to its tonic too
    src_p     = np.array(source_profile["profile"])
    src_tonic = source_profile["tonic"]
    src_aligned = np.roll(src_p, -src_tonic)

    shift_map = {}
    tgt_dominant = set(int(i) for i in np.where(avg_profile > 0.20)[0])
    src_dominant = set(int(i) for i in np.where(src_aligned > 0.20)[0])

    for note in src_dominant:
        if note not in tgt_dominant:
            # This note is strong in source but weak in target — needs to shift
            if not tgt_dominant:
                continue
            nearest = min(
                tgt_dominant,
                key=lambda x: min(abs(x - note), 12 - abs(x - note))
            )
            sh = nearest - note
            if sh > 6:  sh -= 12
            if sh < -6: sh += 12

            # Weight: stronger source note = stronger shift
            weight = float(src_aligned[note])
            shift_map[note] = {"shift": float(sh), "weight": weight}

    return shift_map


def get_shift_map_for_conversion(source_y, source_sr, target_emotion):
    """
    Full pipeline: given source audio and target emotion,
    returns the learned shift map combining raga theory + reference data.
    """
    source_profile = extract_pitch_profile(source_y, source_sr)
    shift_map      = compute_learned_shift_map(source_profile, target_emotion)
    return source_profile, shift_map