import os
import json
import numpy as np
import librosa

REFS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "references")

NOTE_NAMES = ["Sa","R1","R2/G1","R3/G2","G3","M1","M2","Pa","D1","D2/N1","D3/N2","N3"]

def _ensure_dirs():
    for emotion in ["Sad","Happy","Angry","Fearful","Peaceful","Romantic","Surprised","Disgusted"]:
        os.makedirs(os.path.join(REFS_DIR, emotion), exist_ok=True)


def extract_pitch_profile(y, sr):
    chroma  = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=48, n_octaves=6)
    rms     = librosa.feature.rms(y=y, hop_length=512)[0]
    rms     = rms[:chroma.shape[1]]
    weight  = rms / (rms.sum() + 1e-9)
    profile = np.dot(chroma, weight)
    profile = profile / (profile.max() + 1e-9)
    dominant = [int(i) for i in np.where(profile > 0.20)[0]]
    tonic    = int(np.argmax(profile))
    return {"profile": profile.tolist(), "dominant": dominant, "tonic": tonic}


def save_reference(y, sr, song_name, emotion):
    _ensure_dirs()
    data = extract_pitch_profile(y, sr)
    data["song_name"] = song_name
    data["emotion"]   = emotion
    data["sr"]        = sr
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in song_name)
    path = os.path.join(REFS_DIR, emotion, f"{safe}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def load_references(emotion):
    _ensure_dirs()
    refs = []
    edir = os.path.join(REFS_DIR, emotion)
    for fname in os.listdir(edir):
        if fname.endswith(".json"):
            with open(os.path.join(edir, fname)) as f:
                refs.append(json.load(f))
    return refs


def list_all_references():
    _ensure_dirs()
    result = {}
    for emotion in os.listdir(REFS_DIR):
        edir = os.path.join(REFS_DIR, emotion)
        if os.path.isdir(edir):
            songs = []
            for fname in os.listdir(edir):
                if fname.endswith(".json"):
                    with open(os.path.join(edir, fname)) as f:
                        d = json.load(f)
                        songs.append(d.get("song_name", fname))
            if songs:
                result[emotion] = songs
    return result


def delete_reference(emotion, song_name):
    edir = os.path.join(REFS_DIR, emotion)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in song_name)
    path = os.path.join(edir, f"{safe}.json")
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def compute_learned_shift_map(source_profile, target_emotion):
    """
    Key fix: instead of comparing strong vs weak notes,
    we directly map EACH source note to its nearest note
    in the target emotion profile ranked by strength.
    This always produces shifts regardless of overlap.
    """
    refs = load_references(target_emotion)
    print(f"[RefLib] {len(refs)} references loaded for {target_emotion}")
    if not refs:
        return {}

    # Build averaged target profile
    avg = np.zeros(12)
    for ref in refs:
        avg += np.array(ref["profile"])
    avg /= len(refs)
    avg /= (avg.max() + 1e-9)

    src = np.array(source_profile["profile"])
    src /= (src.max() + 1e-9)

    # Target notes ranked by strength
    tgt_ranked = sorted(range(12), key=lambda i: avg[i], reverse=True)
    tgt_top    = set(tgt_ranked[:6])   # top 6 notes in target emotion

    # Source notes ranked by strength
    src_ranked = sorted(range(12), key=lambda i: src[i], reverse=True)
    src_top    = set(src_ranked[:6])   # top 6 notes in source

    print(f"[RefLib] Source top notes:  {[NOTE_NAMES[i] for i in sorted(src_top)]}")
    print(f"[RefLib] Target top notes:  {[NOTE_NAMES[i] for i in sorted(tgt_top)]}")

    shift_map = {}
    # For every strong source note NOT in target top — shift it
    for note in src_top:
        if note not in tgt_top:
            nearest = min(tgt_top, key=lambda x: min(abs(x-note), 12-abs(x-note)))
            sh = nearest - note
            if sh > 6:  sh -= 12
            if sh < -6: sh += 12
            weight = float(src[note])
            shift_map[note] = {"shift": float(sh), "weight": weight}
            print(f"[RefLib] {NOTE_NAMES[note]}({note}) → {NOTE_NAMES[nearest]}({nearest}) shift={sh:+d} weight={weight:.2f}")

    print(f"[RefLib] {len(shift_map)} learned shifts ready")
    return shift_map


def get_shift_map_for_conversion(source_y, source_sr, target_emotion):
    profile   = extract_pitch_profile(source_y, source_sr)
    shift_map = compute_learned_shift_map(profile, target_emotion)
    return profile, shift_map
