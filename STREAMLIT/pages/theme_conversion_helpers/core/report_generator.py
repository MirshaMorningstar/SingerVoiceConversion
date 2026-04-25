"""
REPORT GENERATOR — 60-Parameter Edition
========================================
Covers every dimension of raga/swara/emotion change:
  - Raga identity & theory
  - All 12 swaras (presence in source & target)
  - Note-level changes (added / removed / shifted / kept)
  - Semitone shift per note
  - Arohanam / Avarohanam presence
  - Tempo scaling
  - Raga type, melakarta, character
  - Emotion-to-emotion comparison for all 8 × 8 pairs
  - Per-song batch analysis row (13 columns)
"""

import pandas as pd
import numpy as np
from core.raga_knowledge_base import (
    EMOTION_RAGA_MAP, RAGA_SWARAS, SEMITONE_TO_SWARA,
    SWARA_TO_SEMITONE, build_comparison_table, get_note_changes
)

# ── Helpers ────────────────────────────────────────────────────────────────────

ALL_EMOTIONS = ["Happy", "Sad", "Angry", "Fearful",
                "Disgusted", "Surprised", "Peaceful", "Romantic"]

ALL_SWARAS = ["Sa", "R1", "R2", "G2", "G3", "M1", "M2",
              "Pa", "D1", "D2", "N2", "N3"]

SEMITONE_LABELS = {
    0: "Sa",  1: "R1",  2: "R2/G1", 3: "R3/G2",
    4: "G3",  5: "M1",  6: "M2",    7: "Pa",
    8: "D1",  9: "D2/N1", 10: "D3/N2", 11: "N3",
}

RAGA_TYPE = {
    "Mohanam":        "Pentatonic (Audava-Audava)",
    "Bhairavi":       "Heptatonic (Sampoorna)",
    "Dhanyasi":       "Pentatonic (Audava-Audava)",
    "Todi":           "Heptatonic (Sampoorna)",
    "Kambhoji":       "Shadava-Sampoorna (6-7)",
    "Kalyani":        "Heptatonic (Sampoorna)",
    "Hamsadhwani":    "Pentatonic (Audava-Audava)",
    "Kharaharapriya": "Heptatonic (Sampoorna)",
}

RAGA_SCALE_TYPE = {
    "Mohanam":        "Major Pentatonic",
    "Bhairavi":       "Minor / Phrygian",
    "Dhanyasi":       "Minor Pentatonic",
    "Todi":           "Augmented / Lydian-b2",
    "Kambhoji":       "Mixolydian variant",
    "Kalyani":        "Lydian (raised 4th)",
    "Hamsadhwani":    "Major Pentatonic (no Ma, Dha)",
    "Kharaharapriya": "Natural Minor (Dorian variant)",
}

RAGA_GAMAKA = {
    "Mohanam":        "Kampita on Ga, Dha",
    "Bhairavi":       "Andolita on Ga, Ni; Kampita on Ma",
    "Dhanyasi":       "Andolita on Ga; oscillation on Ni",
    "Todi":           "Kampita on Ri, Ga; Andolita on Dha",
    "Kambhoji":       "Kampita on Ma, Ni",
    "Kalyani":        "Kampita on Ma2, Ga",
    "Hamsadhwani":    "Kampita on Ga, Ni",
    "Kharaharapriya": "Andolita on Ga2, Ni2",
}

RAGA_PREDOMINANT_NOTE = {
    "Mohanam":        "Ga3 (E)",
    "Bhairavi":       "G2 / Ma1 (Eb / F)",
    "Dhanyasi":       "G2 (Eb)",
    "Todi":           "G2 / M2 (Eb / F#)",
    "Kambhoji":       "Ma1 / Pa (F / G)",
    "Kalyani":        "M2 / Ga3 (F# / E)",
    "Hamsadhwani":    "Ga3 / Ni3 (E / B)",
    "Kharaharapriya": "Ni2 / Ga2 (Bb / Eb)",
}

RAGA_VADI_SAMVADI = {
    "Mohanam":        ("Ga3", "Dha2"),
    "Bhairavi":       ("Ma1", "Sa"),
    "Dhanyasi":       ("Ga2", "Ni2"),
    "Todi":           ("Dha1", "Ga2"),
    "Kambhoji":       ("Pa",   "Sa"),
    "Kalyani":        ("Ma2",  "Sa"),
    "Hamsadhwani":    ("Ga3",  "Ni3"),
    "Kharaharapriya": ("Pa",   "Sa"),
}

RAGA_TIME_OF_DAY = {
    "Mohanam":        "Evening / Dusk",
    "Bhairavi":       "Morning (Sunrise)",
    "Dhanyasi":       "Night",
    "Todi":           "Morning (early)",
    "Kambhoji":       "Evening",
    "Kalyani":        "Evening",
    "Hamsadhwani":    "Any time / Devotional",
    "Kharaharapriya": "Night",
}

RAGA_PARENT_MELAKARTA = {
    "Mohanam":        "Harikambhoji (28)",
    "Bhairavi":       "Natabhairavi (20)",
    "Dhanyasi":       "Natabhairavi (20)",
    "Todi":           "Shubhapantuvarali (45)",
    "Kambhoji":       "Harikambhoji (28)",
    "Kalyani":        "Mechakalyani (65)",
    "Hamsadhwani":    "Sankarabharanam (29)",
    "Kharaharapriya": "Kharaharapriya (22)",
}

TEMPO_DESCRIPTION = {
    "Happy":    "Fast (Druta laya)",
    "Sad":      "Slow (Vilamba laya)",
    "Angry":    "Very Fast (Ati-druta)",
    "Fearful":  "Irregular / Tense",
    "Disgusted":"Medium-Slow",
    "Surprised":"Sudden accelerations",
    "Peaceful": "Very Slow (Ati-vilamba)",
    "Romantic": "Medium (Madhyama laya)",
}

EMOTIONAL_INTENSITY = {
    "Happy":    8,
    "Sad":      6,
    "Angry":    10,
    "Fearful":  7,
    "Disgusted":5,
    "Surprised":9,
    "Peaceful": 3,
    "Romantic": 6,
}


def _semitone_set(raga_name):
    info = RAGA_SWARAS[raga_name]
    if "semitones" in info:
        return set(int(s) for s in info["semitones"] if isinstance(s, (int, float)))
    result = set()
    for sw in info.get("swaras", []):
        if sw in SWARA_TO_SEMITONE:
            result.add(SWARA_TO_SEMITONE[sw])
    return result


def _arohanam_set(raga_name):
    info = RAGA_SWARAS.get(raga_name, {})
    aro = info.get("arohanam", "")
    notes = [n.strip() for n in aro.replace("–", " ").split() if n.strip()]
    result = set()
    for n in notes:
        if n in SWARA_TO_SEMITONE:
            result.add(SWARA_TO_SEMITONE[n])
    return result


def _avarohanam_set(raga_name):
    info = RAGA_SWARAS.get(raga_name, {})
    ava = info.get("avarohanam", "")
    notes = [n.strip() for n in ava.replace("–", " ").split() if n.strip()]
    result = set()
    for n in notes:
        if n in SWARA_TO_SEMITONE:
            result.add(SWARA_TO_SEMITONE[n])
    return result


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 1 — MASTER RAGA–EMOTION TABLE  (one row per emotion, ~35 columns)
# ══════════════════════════════════════════════════════════════════════════════

def get_emotion_raga_table():
    """
    35 columns per emotion row:
    Identity (4) + Theory (8) + Swara presence (12) +
    Arohanam count + Avarohanam count + Gamaka + Vadi/Samvadi +
    Scale type + Time + Intensity + Tempo desc + Parent melakarta
    """
    rows = []
    for emotion in ALL_EMOTIONS:
        raga   = EMOTION_RAGA_MAP[emotion]
        info   = RAGA_SWARAS[raga]
        s_set  = _semitone_set(raga)
        aro    = _arohanam_set(raga)
        ava    = _avarohanam_set(raga)
        vadi, samvadi = RAGA_VADI_SAMVADI.get(raga, ("–", "–"))

        row = {
            # ── IDENTITY (4 columns) ──────────────────────────────────────
            "Emotion":              emotion,
            "Raga":                 raga,
            "Melakarta":            info.get("melakarta", "–"),
            "Parent Melakarta No.": RAGA_PARENT_MELAKARTA.get(raga, "–"),

            # ── THEORY (8 columns) ───────────────────────────────────────
            "Raga Type":            RAGA_TYPE.get(raga, "–"),
            "Scale Equivalent":     RAGA_SCALE_TYPE.get(raga, "–"),
            "No. of Swaras":        len(s_set),
            "Arohanam Note Count":  len(aro) if aro else len(s_set),
            "Avarohanam Note Count":len(ava) if ava else len(s_set),
            "Arohanam":             info.get("arohanam", "–"),
            "Avarohanam":           info.get("avarohanam", "–"),
            "Tempo Description":    TEMPO_DESCRIPTION.get(emotion, "–"),

            # ── CHARACTER (6 columns) ────────────────────────────────────
            "Vadi (King Note)":     vadi,
            "Samvadi (Queen Note)": samvadi,
            "Predominant Note":     RAGA_PREDOMINANT_NOTE.get(raga, "–"),
            "Gamaka Style":         RAGA_GAMAKA.get(raga, "–"),
            "Time of Day":          RAGA_TIME_OF_DAY.get(raga, "–"),
            "Emotional Intensity":  f"{EMOTIONAL_INTENSITY.get(emotion, 5)}/10",

            # ── SWARA PRESENCE (12 columns, one per semitone) ────────────
            "Sa (0)":    "✓" if 0  in s_set else "–",
            "R1 (1)":    "✓" if 1  in s_set else "–",
            "R2/G1 (2)": "✓" if 2  in s_set else "–",
            "R3/G2 (3)": "✓" if 3  in s_set else "–",
            "G3 (4)":    "✓" if 4  in s_set else "–",
            "M1 (5)":    "✓" if 5  in s_set else "–",
            "M2 (6)":    "✓" if 6  in s_set else "–",
            "Pa (7)":    "✓" if 7  in s_set else "–",
            "D1 (8)":    "✓" if 8  in s_set else "–",
            "D2/N1 (9)": "✓" if 9  in s_set else "–",
            "D3/N2 (10)":"✓" if 10 in s_set else "–",
            "N3 (11)":   "✓" if 11 in s_set else "–",

            # ── EXAMPLES (1 column) ──────────────────────────────────────
            "Tamil Song Examples":  ", ".join(info.get("tamil_songs", [])[:3]),
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 2 — NOTE CHANGE TABLE  (one row per semitone, ~20 columns)
# ══════════════════════════════════════════════════════════════════════════════

def get_note_change_table(source_emotion, target_emotion):
    """
    20 columns per semitone row:
    Semitone + Swara name + In-source + In-target +
    Change type + Semitone shift + Direction + Shift magnitude +
    In source arohanam + In source avarohanam +
    In target arohanam + In target avarohanam +
    Role in source raga + Role in target raga +
    Frequency (Hz approx, Sa=261.6) + Western note +
    Gamaka in source + Gamaka in target +
    Emotional role + Transformation rule
    """
    src_raga = EMOTION_RAGA_MAP[source_emotion]
    tgt_raga = EMOTION_RAGA_MAP[target_emotion]
    src_set  = _semitone_set(src_raga)
    tgt_set  = _semitone_set(tgt_raga)
    src_aro  = _arohanam_set(src_raga)
    src_ava  = _avarohanam_set(src_raga)
    tgt_aro  = _arohanam_set(tgt_raga)
    tgt_ava  = _avarohanam_set(tgt_raga)

    src_info = RAGA_SWARAS[src_raga]
    tgt_info = RAGA_SWARAS[tgt_raga]

    # Vadi/Samvadi for source and target
    src_vadi, src_samvadi = RAGA_VADI_SAMVADI.get(src_raga, ("–","–"))
    tgt_vadi, tgt_samvadi = RAGA_VADI_SAMVADI.get(tgt_raga, ("–","–"))

    # Western note names (Sa = C4 = 261.6 Hz)
    WESTERN = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]
    SA_FREQ  = 261.63

    rows = []
    for s in range(12):
        swara    = SEMITONE_LABELS[s]
        in_src   = s in src_set
        in_tgt   = s in tgt_set

        # Change type
        if in_src and in_tgt:
            change = "Kept"
            shift_st = 0
        elif in_src and not in_tgt:
            change = "Removed"
            # Nearest note in target
            if tgt_set:
                nearest = min(tgt_set, key=lambda x: min(abs(x-s), 12-abs(x-s)))
                sh = nearest - s
                if sh > 6:  sh -= 12
                if sh < -6: sh += 12
                shift_st = sh
            else:
                shift_st = 0
        elif not in_src and in_tgt:
            change   = "Added"
            shift_st = 0
        else:
            change   = "Not used"
            shift_st = 0

        direction = "Up" if shift_st > 0 else ("Down" if shift_st < 0 else "No shift")
        magnitude = abs(shift_st)

        # Role determination
        def _role(st, vadi, samvadi, raga_set):
            nm = SEMITONE_LABELS[st]
            if nm == vadi:              return "Vadi (King)"
            if nm == samvadi:           return "Samvadi (Queen)"
            if st == 0:                 return "Sa (Tonic)"
            if st == 7:                 return "Pa (Dominant)"
            if st in raga_set:          return "Anga swara"
            return "–"

        freq = round(SA_FREQ * (2 ** (s / 12)), 2)

        row = {
            # ── IDENTIFICATION (3) ───────────────────────────────────────
            "Semitone":           s,
            "Swara Name":         swara,
            "Western Note (C=Sa)":WESTERN[s],

            # ── PRESENCE (4) ────────────────────────────────────────────
            f"In {src_raga} ({source_emotion})": "Yes" if in_src else "No",
            f"In {tgt_raga} ({target_emotion})": "Yes" if in_tgt else "No",
            "In Source Arohanam":  "Yes" if s in src_aro else ("N/A" if not in_src else "No"),
            "In Source Avarohanam":"Yes" if s in src_ava else ("N/A" if not in_src else "No"),

            # ── TARGET AROHANAM/AVAROHANAM (2) ──────────────────────────
            "In Target Arohanam":  "Yes" if s in tgt_aro else ("N/A" if not in_tgt else "No"),
            "In Target Avarohanam":"Yes" if s in tgt_ava else ("N/A" if not in_tgt else "No"),

            # ── CHANGE (4) ──────────────────────────────────────────────
            "Change Type":         change,
            "Semitone Shift":      f"{shift_st:+d}" if change == "Removed" else "0",
            "Direction":           direction if change == "Removed" else "–",
            "Shift Magnitude (st)":magnitude if change == "Removed" else 0,

            # ── ROLE (2) ────────────────────────────────────────────────
            "Role in Source Raga": _role(s, src_vadi, src_samvadi, src_set),
            "Role in Target Raga": _role(s, tgt_vadi, tgt_samvadi, tgt_set),

            # ── PHYSICS & STYLE (3) ─────────────────────────────────────
            "Frequency Hz (approx)":freq,
            "Gamaka in Source":     RAGA_GAMAKA.get(src_raga, "–") if in_src else "–",
            "Gamaka in Target":     RAGA_GAMAKA.get(tgt_raga, "–") if in_tgt else "–",

            # ── TRANSFORMATION RULE (1) ──────────────────────────────────
            "Transformation Rule":  (
                f"Shift {swara} by {shift_st:+d} st toward nearest {tgt_raga} note"
                if change == "Removed" else
                f"Introduce {swara} — new in {tgt_raga}" if change == "Added" else
                f"Retain {swara} — common to both ragas" if change == "Kept" else
                f"{swara} absent in both ragas — skip"
            ),
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 3 — ALL-PAIRS CHANGE MATRIX  (8 × 8 = 64 rows, compact)
# ══════════════════════════════════════════════════════════════════════════════

def get_all_pairs_change_matrix():
    """
    One row per emotion pair (8×8=64). Columns:
    Source emotion, Target emotion, Source raga, Target raga,
    Notes in source, Notes in target, Shared notes, Notes removed,
    Notes added, Notes removed names, Notes added names, Shared names,
    Avg shift (st), Max shift (st), Tempo change direction,
    Same/different raga type, Melakarta match
    """
    rows = []
    for src_emo in ALL_EMOTIONS:
        for tgt_emo in ALL_EMOTIONS:
            src_raga = EMOTION_RAGA_MAP[src_emo]
            tgt_raga = EMOTION_RAGA_MAP[tgt_emo]
            src_set  = _semitone_set(src_raga)
            tgt_set  = _semitone_set(tgt_raga)

            removed = src_set - tgt_set
            added   = tgt_set - src_set
            shared  = src_set & tgt_set

            # Average shift for removed notes
            shifts = []
            for note in removed:
                if tgt_set:
                    near = min(tgt_set, key=lambda x: min(abs(x-note), 12-abs(x-note)))
                    sh   = near - note
                    if sh > 6:  sh -= 12
                    if sh < -6: sh += 12
                    shifts.append(sh)

            avg_shift = round(np.mean(shifts), 2) if shifts else 0
            max_shift = int(max(shifts, key=abs)) if shifts else 0

            src_tempo = RAGA_SWARAS[src_raga].get("tempo_factor", 1.0)
            tgt_tempo = RAGA_SWARAS[tgt_raga].get("tempo_factor", 1.0)
            if tgt_tempo > src_tempo:   tempo_dir = "Faster"
            elif tgt_tempo < src_tempo: tempo_dir = "Slower"
            else:                       tempo_dir = "Same"

            row = {
                "Source Emotion":       src_emo,
                "Target Emotion":       tgt_emo,
                "Source Raga":          src_raga,
                "Target Raga":          tgt_raga,
                "Notes in Source":      len(src_set),
                "Notes in Target":      len(tgt_set),
                "Shared Notes Count":   len(shared),
                "Notes Removed Count":  len(removed),
                "Notes Added Count":    len(added),
                "Notes Removed":        ", ".join(SEMITONE_LABELS[n] for n in sorted(removed)) or "None",
                "Notes Added":          ", ".join(SEMITONE_LABELS[n] for n in sorted(added))   or "None",
                "Shared Notes":         ", ".join(SEMITONE_LABELS[n] for n in sorted(shared))  or "None",
                "Avg Semitone Shift":   avg_shift,
                "Max Semitone Shift":   max_shift,
                "Tempo Direction":      tempo_dir,
                "Same Raga Type":       "Yes" if RAGA_TYPE.get(src_raga)==RAGA_TYPE.get(tgt_raga) else "No",
                "Same Melakarta":       "Yes" if RAGA_PARENT_MELAKARTA.get(src_raga)==RAGA_PARENT_MELAKARTA.get(tgt_raga) else "No",
            }
            rows.append(row)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 4 — PER-SONG BATCH ROW  (13 columns)
# ══════════════════════════════════════════════════════════════════════════════

def get_song_analysis_row(filename, duration, tempo, dominant_swaras,
    detected_raga, detected_emotion, match_score, target_emotion,
    target_raga, pitch_shift, tempo_factor, notes_removed, notes_added):

    src_set = _semitone_set(detected_raga)
    tgt_set = _semitone_set(target_raga)
    shared  = src_set & tgt_set
    removed = src_set - tgt_set
    added   = tgt_set - src_set

    return {
        # ── FILE INFO ────────────────────────────────────────────────────
        "File":                  filename,
        "Duration (s)":          round(duration, 1),
        "Tempo (BPM)":           round(tempo, 1),

        # ── SOURCE ───────────────────────────────────────────────────────
        "Detected Raga":         detected_raga,
        "Detected Emotion":      detected_emotion,
        "Raga Match Score":      str(round(match_score, 3)),
        "Dominant Swaras":       ", ".join(dominant_swaras),
        "Source Note Count":     len(src_set),

        # ── TARGET ───────────────────────────────────────────────────────
        "Target Emotion":        target_emotion,
        "Target Raga":           target_raga,
        "Target Note Count":     len(tgt_set),

        # ── CHANGES ──────────────────────────────────────────────────────
        "Notes Removed":         ", ".join(SEMITONE_LABELS.get(int(k), k)
                                           for k in notes_removed.keys()) or "None",
        "Notes Added":           ", ".join(SEMITONE_LABELS.get(int(k), k)
                                           for k in notes_added.keys()) or "None",
        "Shared Notes Count":    len(shared),
        "Notes Removed Count":   len(removed),
        "Notes Added Count":     len(added),

        # ── TRANSFORMATION ───────────────────────────────────────────────
        "Pitch Shift (st)":      pitch_shift,
        "Tempo Scale Factor":    str(tempo_factor),
        "Tempo Direction":       "Faster" if float(tempo_factor) > 1 else
                                 ("Slower" if float(tempo_factor) < 1 else "Same"),

        # ── RAGA THEORY ──────────────────────────────────────────────────
        "Source Raga Type":      RAGA_TYPE.get(detected_raga, "–"),
        "Target Raga Type":      RAGA_TYPE.get(target_raga, "–"),
        "Source Scale":          RAGA_SCALE_TYPE.get(detected_raga, "–"),
        "Target Scale":          RAGA_SCALE_TYPE.get(target_raga, "–"),
        "Source Melakarta":      RAGA_PARENT_MELAKARTA.get(detected_raga, "–"),
        "Target Melakarta":      RAGA_PARENT_MELAKARTA.get(target_raga, "–"),
        "Source Gamaka":         RAGA_GAMAKA.get(detected_raga, "–"),
        "Target Gamaka":         RAGA_GAMAKA.get(target_raga, "–"),
        "Source Vadi Note":      RAGA_VADI_SAMVADI.get(detected_raga, ("–","–"))[0],
        "Target Vadi Note":      RAGA_VADI_SAMVADI.get(target_raga, ("–","–"))[0],
        "Emotional Intensity Src":f"{EMOTIONAL_INTENSITY.get(detected_emotion,5)}/10",
        "Emotional Intensity Tgt":f"{EMOTIONAL_INTENSITY.get(target_emotion,5)}/10",
    }


def build_batch_df(rows):
    return pd.DataFrame(rows)