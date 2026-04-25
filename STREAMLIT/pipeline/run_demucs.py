import os
from pipeline.extract_vocals import separate_single_file

def run_demucs(source_path, workspace):
    vocal = os.path.join(workspace, "intermediate", "vocals.wav")
    instrumental = os.path.join(workspace, "intermediate", "non_vocals.wav")

    os.makedirs(os.path.dirname(vocal), exist_ok=True)

    separate_single_file(source_path, vocal, instrumental)

    return vocal, instrumental
