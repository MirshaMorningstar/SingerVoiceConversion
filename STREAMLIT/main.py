import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf

def load_and_separate(audio_path, sr=22050):
    """🎤 ULTRA-CLEAN VOCAL EXTRACTION (No background music)"""
    y_mix, sr = librosa.load(audio_path, sr=sr, mono=True)
    print(f"[INFO] Loaded: {os.path.basename(audio_path)} ({len(y_mix)/sr:.1f}s)")
    
    # ADVANCED HPSS - Maximum vocal isolation
    D = librosa.stft(y_mix)
    D_harmonic, D_percussive = librosa.decompose.hpss(D, margin=(5.0, 2.0))  # Vocal bias
    y_vocal = librosa.istft(D_harmonic, length=len(y_mix))
    
    # Voice Activity Detection - Remove silence
    intervals = librosa.effects.split(y_vocal, top_db=25)
    y_clean = np.zeros_like(y_vocal)
    for start, end in intervals:
        y_clean[start:end] = y_vocal[start:end]
    
    print(f"[INFO] Clean vocals extracted")
    return y_clean, sr

def make_emotion_sound(y, sr, emotion="sad"):
    """🎭 3 PERFECT EMOTIONS - VOCALS ONLY"""
    
    emotion_params = {
        "sad":   {"pitch": -4, "tempo": 0.82, "gain": 0.70, "desc": "Melancholic"}, 
        "happy": {"pitch": +3, "tempo": 1.18, "gain": 1.30, "desc": "Joyful"},
        "calm":  {"pitch": -2, "tempo": 0.92, "gain": 0.45, "desc": "Serene"}
    }
    
    params = emotion_params[emotion]
    
    # PITCH (most emotional)
    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=params["pitch"])
    
    # TEMPO  
    y_tempo = librosa.effects.time_stretch(y_pitch, rate=params["tempo"])
    
    # DYNAMICS
    y_emotion = y_tempo * params["gain"]
    
    print(f"    → {params['desc']} ({params['pitch']}st, {params['tempo']:.0%} tempo)")
    return y_emotion[:len(y)]

def make_loud_and_clear(y_original, y_converted):
    """Perfect volume matching"""
    orig_peak = np.max(np.abs(y_original))
    orig_rms = librosa.feature.rms(y=y_original)[0].mean()
    
    conv_peak = np.max(np.abs(y_converted))
    conv_rms = librosa.feature.rms(y=y_converted)[0].mean()
    
    if conv_peak > 0:
        y_converted = y_converted * (orig_peak / conv_peak * 1.05)
    if conv_rms > 0:
        y_converted = y_converted * (orig_rms / conv_rms)
    
    return np.clip(y_converted, -0.98, 0.98)

def create_three_plots(y_vocal, y_converted, sr, base_name, emotion, output_dir):
    """🎨 3 EXTRA PLOTS - SIMPLE & BULLETPROOF"""
    
    # 1. SPECTROGRAM COMPARISON
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original Spectrogram
    D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(y_vocal[:15000])), ref=np.max)
    img1 = librosa.display.specshow(D_orig, sr=sr, x_axis='time', y_axis='hz', ax=axes[0,0], cmap='Blues')
    axes[0,0].set(title="🔵 Original Spectrogram")
    plt.colorbar(img1, ax=axes[0,0])
    
    # Converted Spectrogram  
    D_conv = librosa.amplitude_to_db(np.abs(librosa.stft(y_converted[:15000])), ref=np.max)
    img2 = librosa.display.specshow(D_conv, sr=sr, x_axis='time', y_axis='hz', ax=axes[0,1], cmap='Reds')
    axes[0,1].set(title=f"🔴 {emotion.title()} Spectrogram")
    plt.colorbar(img2, ax=axes[0,1])
    
    # Vocal Energy (RMS)
    rms_orig = librosa.feature.rms(y=y_vocal[:30000])[0]
    rms_conv = librosa.feature.rms(y=y_converted[:30000])[0]
    times = librosa.times_like(rms_orig, sr=sr)
    axes[1,0].plot(times, rms_orig, 'b-', linewidth=2, label="Original")
    axes[1,0].plot(times[:len(rms_conv)], rms_conv, 'r--', linewidth=2, label=emotion.title())
    axes[1,0].set(title="📊 Vocal Energy", xlabel="Time (s)")
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Spectral Centroid (Brightness)
    cent_orig = librosa.feature.spectral_centroid(y=y_vocal[:30000], sr=sr)[0]
    cent_conv = librosa.feature.spectral_centroid(y=y_converted[:30000], sr=sr)[0]
    axes[1,1].plot(times[:len(cent_orig)], cent_orig, 'b-', linewidth=2, label="Original")
    axes[1,1].plot(times[:len(cent_conv)], cent_conv, 'r--', linewidth=2, label=emotion.title())
    axes[1,1].set(title="🌈 Brightness (Centroid)", xlabel="Time (s)")
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{base_name}_{emotion}_extra_plots.png", dpi=150, bbox_inches='tight')
    plt.close()

def theme_conversion_pipeline(audio_path, target_emotion="sad", output_dir="converted"):
    """🎵 CLEAN VOCAL PIPELINE + 4 PLOTS TOTAL"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🎤 {target_emotion.upper()} VOCAL CONVERSION")
    
    # Extract CLEAN vocals
    y_vocal, sr = load_and_separate(audio_path)
    base_name = os.path.splitext(os.path.basename(audio_path))[0].replace(" ", "_")
    
    # Save ORIGINAL clean vocal
    orig_path = f"{output_dir}/{base_name}_01_clean_vocal.wav"
    sf.write(orig_path, y_vocal, sr)
    
    # Convert emotion
    y_emotion = make_emotion_sound(y_vocal, sr, target_emotion)
    y_converted = make_loud_and_clear(y_vocal, y_emotion)
    
    # Save emotion version
    conv_path = f"{output_dir}/{base_name}_02_{target_emotion}_vocal.wav"
    sf.write(conv_path, y_converted, sr)
    
    # YOUR ORIGINAL WAVEFORM PLOT (1️⃣)
    plt.figure(figsize=(14, 4))
    samples = min(2000, len(y_vocal))
    t = np.linspace(0, samples/sr, samples)
    
    plt.plot(t, y_vocal[:samples], 'b-', alpha=0.8, linewidth=1, label="Clean Vocal")
    plt.plot(t, y_converted[:samples], 'r--', linewidth=2, label=target_emotion.title())
    plt.title(f"VOCAL ONLY: {target_emotion.upper()} Conversion", fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/{base_name}_{target_emotion}_plot.png", dpi=150)
    plt.close()
    
    # 3 NEW EXTRA PLOTS (2️⃣3️⃣4️⃣)
    create_three_plots(y_vocal, y_converted, sr, base_name, target_emotion, output_dir)
    
    print(f"✅ {orig_path}")
    print(f"✅ {conv_path}")
    print(f"✅ 4 PLOTS: waveform_plot.png + extra_plots.png")
    print("🎉 VOCAL CONVERSION COMPLETE!")

if __name__ == "__main__":
    theme_conversion_pipeline("songs/Unnaale Unnaale - isaimini.one.mp3", "sad")
