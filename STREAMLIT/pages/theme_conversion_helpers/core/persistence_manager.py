"""
PERSISTENCE MANAGER
===================
Saves converted songs + metadata to disk permanently.
Survives Streamlit refreshes, restarts, and long gaps.

Storage layout:
  saved_conversions/
    {timestamp}_{songname}_{emotion}/
      original.wav        ← original uploaded audio
      converted.wav       ← converted output
      metadata.json       ← all report, log, row fields (JSON-serialisable)

On app start, call load_all_conversions() to restore session state.
"""

import os
import json
import time
import shutil
import numpy as np

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_conversions")


def _ensure_save_dir():
    os.makedirs(SAVE_DIR, exist_ok=True)


def _safe_name(s):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s))


def _serialise(obj):
    """Recursively make obj JSON-serialisable."""
    if isinstance(obj, dict):
        return {
            str(k): _serialise(v) for k, v in obj.items()
            if not (isinstance(k, str) and k.startswith("_"))
        }
    if isinstance(obj, (list, tuple)):
        return [_serialise(i) for i in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def save_conversion(original_bytes, original_name,
                    converted_bytes, report, log, row,
                    target_emotion, target_raga):
    """
    Persist one conversion to disk.
    Returns the folder path created.
    """
    _ensure_save_dir()

    ts     = int(time.time())
    folder = f"{ts}_{_safe_name(original_name)}_{_safe_name(target_emotion)}"
    path   = os.path.join(SAVE_DIR, folder)
    os.makedirs(path, exist_ok=True)

    # Audio files
    with open(os.path.join(path, "original.wav"), "wb") as f:
        f.write(original_bytes)
    with open(os.path.join(path, "converted.wav"), "wb") as f:
        f.write(converted_bytes)

    # Metadata — strip numpy arrays before serialising
    meta = {
        "original_name":  original_name,
        "target_emotion": target_emotion,
        "target_raga":    target_raga,
        "timestamp":      ts,
        "report":         _serialise(report),
        "log":            _serialise(log),
        "row":            _serialise(row),
    }
    with open(os.path.join(path, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return path


def load_all_conversions():
    """
    Load every saved conversion from disk.
    Returns a list of result dicts compatible with session_state.results.
    """
    _ensure_save_dir()
    results = []

    folders = sorted(os.listdir(SAVE_DIR))   # chronological order
    for folder in folders:
        fpath = os.path.join(SAVE_DIR, folder)
        if not os.path.isdir(fpath):
            continue

        meta_path  = os.path.join(fpath, "metadata.json")
        orig_path  = os.path.join(fpath, "original.wav")
        conv_path  = os.path.join(fpath, "converted.wav")

        if not all(os.path.exists(p) for p in [meta_path, orig_path, conv_path]):
            continue

        try:
            with open(meta_path) as f:
                meta = json.load(f)
            with open(orig_path, "rb") as f:
                original_bytes = f.read()
            with open(conv_path, "rb") as f:
                converted_bytes = f.read()

            results.append({
                "original_bytes":  original_bytes,
                "original_name":   meta["original_name"],
                "converted_bytes": converted_bytes,
                "report":          meta["report"],
                "log":             meta["log"],
                "row":             meta["row"],
                "target_emotion":  meta["target_emotion"],
                "target_raga":     meta["target_raga"],
                "timestamp":       meta.get("timestamp", 0),
                "folder":          fpath,
            })
        except Exception:
            continue   # skip corrupted entries silently

    return results


def delete_conversion(folder_path):
    """Delete a saved conversion folder from disk."""
    if os.path.isdir(folder_path):
        shutil.rmtree(folder_path)