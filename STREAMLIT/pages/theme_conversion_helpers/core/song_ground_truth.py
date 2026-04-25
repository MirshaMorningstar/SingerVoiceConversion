"""
SONG GROUND TRUTH DATABASE
Cleaned + bug-fixed version
"""

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────

SONG_GROUND_TRUTH = {

    "poneepo": {
        "song": "Po Nee Po",
        "film": "3 (Moonu)",
        "emotion": "Sad",
        "raga": "Kharaharapriya",
        "tonic": "C",

        "semitone_profile": [
            0.283,0.000,0.133,0.221,0.000,
            0.083,0.004,0.092,0.042,0.008,
            0.133,0.000
        ],

        "dominant_swaras": ["S","G2","R2","N2","M1","P"],

        "sequence": [
            "G2","S","N2","N2","S","G2","G2","R2","G2","G2","G2","R2",
            "G2","S","S","S","G2","R2","S","S","R2","S","N2","P","P","S"
        ],

        "arohanam": "S R2 G2 M1 P D1 N2 S",
        "avarohanam": "S N2 D1 P M1 G2 R2 S"
    }
}


# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────

import re
import os
import numpy as np


NOTE_NAMES = {
    0:"S",1:"R1",2:"R2",3:"G2",4:"G3",5:"M1",
    6:"M2",7:"P",8:"D1",9:"D2",10:"N2",11:"N3"
}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _clean(text):
    return re.sub(r'[^a-z0-9]', '', text.lower())


def _safe_normalize(arr):
    arr = np.array(arr, dtype=np.float32)
    max_val = np.max(arr) if np.max(arr) > 0 else 1.0
    return arr / max_val


# ─────────────────────────────────────────────
# LOOKUP
# ─────────────────────────────────────────────

def lookup_song(filename):
    fn = _clean(os.path.splitext(os.path.basename(filename))[0])

    for key, data in SONG_GROUND_TRUTH.items():
        if key in fn or fn in key:
            return data

    return None


# ─────────────────────────────────────────────
# SIMILARITY (FIXED)
# ─────────────────────────────────────────────

def compute_similarity(extracted_profile, ground_truth_profile):

    ep = _safe_normalize(extracted_profile)
    gp = _safe_normalize(ground_truth_profile)

    cos_sim = float(
        np.dot(ep, gp) /
        (np.linalg.norm(ep) * np.linalg.norm(gp) + 1e-9)
    )

    threshold = 0.15

    ep_notes = set(np.where(ep > threshold)[0])
    gp_notes = set(np.where(gp > threshold)[0])

    correct = ep_notes & gp_notes
    false_pos = ep_notes - gp_notes
    false_neg = gp_notes - ep_notes

    precision = len(correct) / max(len(ep_notes), 1)
    recall = len(correct) / max(len(gp_notes), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    # ✅ NEW: PER-NOTE TABLE (THIS FIXES YOUR ERROR)
    per_note = []
    for i in range(12):
        per_note.append({
            "Note": NOTE_NAMES[i],
            "Extracted": round(float(ep[i]), 3),
            "Ground Truth": round(float(gp[i]), 3),
            "Match": "Yes" if (i in correct) else "No"
        })

    return {
        "cosine_similarity": round(cos_sim, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),

        "correct_notes": [NOTE_NAMES[n] for n in correct],
        "false_positives": [NOTE_NAMES[n] for n in false_pos],
        "false_negatives": [NOTE_NAMES[n] for n in false_neg],

        # ✅ REQUIRED KEY
        "per_note": per_note
    }


# ─────────────────────────────────────────────
# S1 vs S2 COMPARISON (FULL FIX)
# ─────────────────────────────────────────────

def compare_s1_s2(s1_profile, s2_profile, s1_emotion, s2_emotion):

    p1 = _safe_normalize(s1_profile)
    p2 = _safe_normalize(s2_profile)

    threshold = 0.15

    n1 = set(np.where(p1 > threshold)[0])
    n2 = set(np.where(p2 > threshold)[0])

    cos_sim = float(
        np.dot(p1, p2) /
        (np.linalg.norm(p1) * np.linalg.norm(p2) + 1e-9)
    )

    # NOTE CHANGES
    changes = []
    for i in range(12):
        diff = float(p2[i] - p1[i])
        if abs(diff) > 0.05:
            changes.append({
                "Note": NOTE_NAMES[i],
                "S1 Weight": round(float(p1[i]), 3),
                "S2 Weight": round(float(p2[i]), 3),
                "Change": round(diff, 3),
                "Direction": "Increased" if diff > 0 else "Decreased",
            })

    changes.sort(key=lambda x: abs(x["Change"]), reverse=True)

    return {
        "s1_emotion": s1_emotion,
        "s2_emotion": s2_emotion,

        # REQUIRED KEY
        "cosine_distance": round(1 - cos_sim, 4),

        # REQUIRED KEYS
        "notes_gained": [NOTE_NAMES[n] for n in (n2 - n1)],
        "notes_lost": [NOTE_NAMES[n] for n in (n1 - n2)],
        "notes_shared": [NOTE_NAMES[n] for n in (n1 & n2)],

        # REQUIRED KEY
        "note_changes": changes
    }