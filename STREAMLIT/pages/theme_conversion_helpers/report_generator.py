import pandas as pd
from core.raga_knowledge_base import (
    EMOTION_RAGA_MAP, RAGA_SWARAS, SEMITONE_TO_SWARA,
    SWARA_TO_SEMITONE, build_comparison_table, get_note_changes
)

def _raga_semitone_set(raga_name):
    info = RAGA_SWARAS[raga_name]
    if "semitones" in info:
        return set(info["semitones"])
    result = set()
    for sw in info.get("swaras", []):
        if sw in SWARA_TO_SEMITONE:
            result.add(SWARA_TO_SEMITONE[sw])
    return result

def get_emotion_raga_table():
    rows = build_comparison_table()
    df = pd.DataFrame(rows)
    front = ["Emotion","Raga","Melakarta","Tempo","Description"]
    front = [c for c in front if c in df.columns]
    rest  = [c for c in df.columns if c not in front]
    return df[front + rest]

def get_note_change_table(source_emotion, target_emotion):
    src_raga = EMOTION_RAGA_MAP[source_emotion]
    tgt_raga = EMOTION_RAGA_MAP[target_emotion]
    src_set  = _raga_semitone_set(src_raga)
    tgt_set  = _raga_semitone_set(tgt_raga)
    rows = []
    for s in range(12):
        name   = SEMITONE_TO_SWARA[s]
        in_src = s in src_set
        in_tgt = s in tgt_set
        if in_src and in_tgt:       status = "Kept"
        elif in_src and not in_tgt: status = "Removed"
        elif not in_src and in_tgt: status = "Added"
        else:                       status = "Not used"
        rows.append({
            "Semitone": s,
            "Swara":    name,
            "In " + src_raga + " (" + source_emotion + ")": "Yes" if in_src else "No",
            "In " + tgt_raga + " (" + target_emotion + ")": "Yes" if in_tgt else "No",
            "Change":   status,
        })
    return pd.DataFrame(rows)

def get_song_analysis_row(filename, duration, tempo, dominant_swaras,
    detected_raga, detected_emotion, match_score, target_emotion,
    target_raga, pitch_shift, tempo_factor, notes_removed, notes_added):
    return {
        "File":             filename,
        "Duration (s)":     duration,
        "Tempo (BPM)":      round(tempo, 1),
        "Dominant Swaras":  ", ".join(dominant_swaras),
        "Detected Raga":    detected_raga,
        "Detected Emotion": detected_emotion,
        "Match Score":      str(round(match_score, 2)),
        "Target Emotion":   target_emotion,
        "Target Raga":      target_raga,
        "Pitch Shift (st)": pitch_shift,
        "Tempo Scale":      str(tempo_factor),
        "Notes Removed":    ", ".join(notes_removed.values()) or "None",
        "Notes Added":      ", ".join(notes_added.values()) or "None",
    }

def build_batch_df(rows):
    return pd.DataFrame(rows)