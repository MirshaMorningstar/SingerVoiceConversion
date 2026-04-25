import os
import torch
import soundfile as sf
import numpy as np
import librosa

from demucs import pretrained
from demucs.apply import apply_model


# ------------------ Audio helpers ------------------ #
def separate_single_file(input_path, vocal_out, non_vocal_out):
    """
    Streamlit wrapper call for one file instead of folder batch.
    """
    model_name = "htdemucs_ft"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = pretrained.get_model(model_name)
    model.to(device)
    model.eval()

    separate_song(
        model=model,
        device=device,
        input_path=input_path,
        vocal_out_path=vocal_out,
        non_vocal_out_path=non_vocal_out,
    )


def load_audio(path, target_sr=44100):
    """
    Load an audio file as float32 tensor [channels, samples], resampled to target_sr.
    """
    audio, sr = librosa.load(path, sr=target_sr, mono=False)
    if audio.ndim == 1:
        audio = np.expand_dims(audio, 0)  # [1, samples]
    return torch.from_numpy(audio).float(), target_sr


def save_audio(path, audio, sr):
    """
    Save tensor [channels, samples] to file (float32 PCM WAV).
    """
    audio_np = audio.detach().cpu().numpy()
    if audio_np.ndim == 2 and audio_np.shape[0] == 1:
        audio_np = audio_np[0]
    sf.write(path, audio_np.T if audio_np.ndim == 2 else audio_np, sr)


def highpass_filter(audio, sr, cutoff=80.0):
    """
    Simple high‑pass via first‑order IIR to remove low‑frequency rumble.
    audio: [channels, samples] tensor.
    """
    import scipy.signal as sps

    audio_np = audio.cpu().numpy()
    nyq = 0.5 * sr
    norm_cutoff = cutoff / nyq
    b, a = sps.butter(1, norm_cutoff, btype="high", analog=False)
    filtered = sps.lfilter(b, a, audio_np, axis=-1)
    return torch.from_numpy(filtered).to(audio.device)


def loudness_normalize(audio, target_db=-16.0):
    """
    Normalize audio to a target RMS level (approx. LUFS‑like).
    audio: [channels, samples].
    """
    eps = 1e-9
    rms = torch.sqrt(torch.mean(audio ** 2) + eps)
    current_db = 20.0 * torch.log10(rms + eps)
    gain_db = target_db - current_db
    gain = 10.0 ** (gain_db / 20.0)
    return audio * gain


# ------------------ Separation core ------------------ #

def separate_song(
    model,
    device,
    input_path: str,
    vocal_out_path: str,
    non_vocal_out_path: str,
):
    """
    Run Demucs on a single song, save vocals and merged non‑vocals.
    """
    print(f"\n=== Processing: {os.path.basename(input_path)} ===")
    print(f"Using device: {device}")

    # Load and prepare audio
    mix, sr = load_audio(input_path, target_sr=44100)
    mix = mix.to(device)
    mix_batch = mix.unsqueeze(0)  # [1, channels, samples]

    with torch.no_grad():
        print("Running Demucs separation...")
        # sources: [num_sources, channels, samples]
        sources = apply_model(
            model,
            mix_batch,
            shifts=1,
            split=True,
            overlap=0.25,
            progress=True
        )[0]

    source_names = model.sources
    print(f"Model sources: {source_names}")

    # ---- Extract vocals ---- #
    try:
        vocal_index = source_names.index("vocals")
    except ValueError:
        raise RuntimeError(
            f"Model does not provide a 'vocals' stem: {source_names}"
        )

    vocals = sources[vocal_index]  # [channels, samples]

    # ---- Merge all non‑vocal instruments (drums, bass, other, etc.) ---- #
    non_vocals_list = [
        sources[i] for i, name in enumerate(source_names) if name != "vocals"
    ]
    if len(non_vocals_list) == 0:
        raise RuntimeError("No non‑vocal stems found in model output.")

    # Sum instruments; this can include 10+ instrument types implicitly
    non_vocals = torch.stack(non_vocals_list, dim=0).sum(dim=0)

    # Post‑processing
    vocals = highpass_filter(vocals, sr, cutoff=80.0)
    vocals = loudness_normalize(vocals, target_db=-18.0)

    non_vocals = highpass_filter(non_vocals, sr, cutoff=40.0)
    non_vocals = loudness_normalize(non_vocals, target_db=-18.0)

    # Ensure output directories exist
    os.makedirs(os.path.dirname(vocal_out_path), exist_ok=True)
    os.makedirs(os.path.dirname(non_vocal_out_path), exist_ok=True)

    # Save
    save_audio(vocal_out_path, vocals, sr)
    save_audio(non_vocal_out_path, non_vocals, sr)

    print(f"Saved vocals to     : {vocal_out_path}")
    print(f"Saved non‑vocals to : {non_vocal_out_path}")
    print("=== Finished ===")


# ------------------ Batch over folder ------------------ #

def main():
    # Root project directory (edit if needed)
    project_root = r"D:\RagaVoiceStudio\Putham_Pudhu_Kaalai"

    inputs_dir = os.path.join(project_root, "Inputs")
    vocals_dir = os.path.join(project_root, "Vocals")
    non_vocals_dir = os.path.join(project_root, "Non_Vocals")

    # Choose the best Demucs model: fine‑tuned Hybrid Transformer Demucs
    # More accurate than plain htdemucs, but slower. [web:39][web:81][web:161]
    model_name = "htdemucs_ft"

    # Auto‑select device; change to "cpu" if you want to force CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Global device selection: {device}")
    print(f"Using Demucs model: {model_name}")

    # Load model once and reuse
    print("Loading pretrained Demucs model (this happens only once)...")
    model = pretrained.get_model(model_name)
    model.to(device)
    model.eval()

    # List audio files in Inputs/
    supported_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    files = [
        f for f in os.listdir(inputs_dir)
        if os.path.splitext(f)[1].lower() in supported_exts
    ]

    if not files:
        print(f"No audio files found in {inputs_dir}")
        return

    print(f"Found {len(files)} file(s) to process in {inputs_dir}")

    for fname in files:
        in_path = os.path.join(inputs_dir, fname)
        base, _ = os.path.splitext(fname)

        vocal_out = os.path.join(vocals_dir, base + "_vocals.wav")
        non_vocal_out = os.path.join(non_vocals_dir, base + "_non_vocals.wav")

        separate_song(
            model=model,
            device=device,
            input_path=in_path,
            vocal_out_path=vocal_out,
            non_vocal_out_path=non_vocal_out,
        )

    print("\nAll songs processed.")


if __name__ == "__main__":
    main()
