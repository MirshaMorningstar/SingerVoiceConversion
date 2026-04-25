import os
import io
import json
import warnings
import tempfile

import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import find_peaks
from collections import Counter

import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Swara Extraction Pipeline",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  h1, h2, h3 { font-family: 'Playfair Display', serif; }

  .main-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #c8602a 0%, #e8a060 50%, #c8602a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
  }

  .subtitle {
    color: #888;
    font-size: 1rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 2rem;
  }

  .section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    color: #c8602a;
    border-bottom: 2px solid #c8602a22;
    padding-bottom: 0.4rem;
    margin-top: 2rem;
    margin-bottom: 1rem;
  }

  .swara-pill {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    margin: 0.25rem;
    border-radius: 999px;
    font-weight: 500;
    font-size: 1.0rem;
    font-family: 'DM Sans', sans-serif;
  }

  .sequence-box {
    background: #1a1a2e;
    border: 1px solid #333;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.95rem;
  }

  .info-card {
    background: #f9f5f0;
    border-left: 4px solid #c8602a;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.95rem;
  }

  .stProgress > div > div { background-color: #c8602a; }

  div[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif;
    color: #c8602a;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS (unchanged from notebook)
# ─────────────────────────────────────────────────────────────────────────────
TARGET_SR = 22050

SWARA_CENTS = {
    'Sa': 0, 'Ri1': 100, 'Ri2': 200, 'Ri3': 300,
    'Ga1': 200, 'Ga2': 300, 'Ga3': 400,
    'Ma1': 500, 'Ma2': 600, 'Pa': 700,
    'Da1': 800, 'Da2': 900, 'Da3': 1000,
    'Ni1': 900, 'Ni2': 1000, 'Ni3': 1100, "Sa'": 1200,
}
BOUNDARY_CENTS = 35

SWARA_TABLE = {
    'Sa': (0, 'Sa'), 'Ri1': (90, 'Ri'), 'Ri2': (204, 'Ri'), 'Ri3': (294, 'Ri'),
    'Ga1': (90, 'Ga'), 'Ga2': (204, 'Ga'), 'Ga3': (294, 'Ga'),
    'Ma1': (498, 'Ma'), 'Ma2': (590, 'Ma'), 'Pa': (702, 'Pa'),
    'Da1': (792, 'Da'), 'Da2': (906, 'Da'), 'Da3': (996, 'Da'),
    'Ni1': (792, 'Ni'), 'Ni2': (906, 'Ni'), 'Ni3': (1008, 'Ni'),
    "Sa'": (1200, 'Sa'), "Ri1'": (1290, 'Ri'), "Ri2'": (1404, 'Ri'),
    "Pa_low": (-498, 'Pa'), "Sa_low": (-1200, 'Sa'), "Ni3_low": (-192, 'Ni'),
}

SWARA_COLOR_MAP = {
    'Sa': '#2196F3', 'Ri1': '#9C27B0', 'Ri2': '#9C27B0', 'Ri3': '#9C27B0',
    'Ga1': '#E91E63', 'Ga2': '#E91E63', 'Ga3': '#E91E63',
    'Ma1': '#FF9800', 'Ma2': '#FF5722', 'Pa': '#4CAF50',
    'Da1': '#00BCD4', 'Da2': '#00BCD4', 'Da3': '#00BCD4',
    'Ni1': '#8BC34A', 'Ni2': '#8BC34A', 'Ni3': '#8BC34A',
    "Sa'": '#2196F3', 'transit': '#BBBBBB', 'unvoiced': None,
}

MIN_PRESENCE_PERCENT = 2.0
TOLERANCE_CENTS = 55

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE FUNCTIONS (exact logic from notebook, no changes)
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(file_obj, target_sr=TARGET_SR):
    audio, sr = librosa.load(file_obj, sr=target_sr, mono=True)
    audio = np.clip(audio, -1.0, 1.0)
    return audio, sr


def plot_waveform(audio, sr, title="Waveform", max_display_seconds=60):
    duration = len(audio) / sr
    display_samples = min(len(audio), int(max_display_seconds * sr))
    audio_display = audio[:display_samples]
    time_axis = np.linspace(0, display_samples / sr, num=display_samples)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    fig.suptitle(f"{title}  |  Duration: {duration:.1f}s  |  SR: {sr} Hz", fontsize=13)

    axes[0].plot(time_axis, audio_display, color='steelblue', linewidth=0.4, alpha=0.8)
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"Waveform (first {max_display_seconds}s shown)" if duration > max_display_seconds else "Full waveform")
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].axhline(0, color='gray', linewidth=0.5, linestyle='--')
    axes[0].grid(True, alpha=0.3)

    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=audio_display, frame_length=frame_length, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    axes[1].fill_between(rms_times, rms, color='darkorange', alpha=0.7)
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("RMS Energy")
    axes[1].set_title("Energy envelope")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def build_pitch_histogram(audio, sr, fmin=60, fmax=1200, bins_per_octave=120):
    hop_length = 512
    n_octaves = int(np.log2(fmax / fmin)) + 1
    n_bins = bins_per_octave * n_octaves
    cqt = librosa.cqt(audio, sr=sr, hop_length=hop_length, fmin=fmin,
                      n_bins=n_bins, bins_per_octave=bins_per_octave)
    cqt_mag = np.abs(cqt)
    pitch_histogram = np.sum(cqt_mag, axis=1)
    freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
    return pitch_histogram, freqs


def detect_tonic(pitch_histogram, freqs, n_candidates=5):
    hist_norm = pitch_histogram / pitch_histogram.max()
    peaks, properties = find_peaks(hist_norm, prominence=0.05, distance=5)
    if len(peaks) == 0:
        raise ValueError("No peaks found in pitch histogram.")
    prominences = properties['prominences']
    sorted_idx = np.argsort(prominences)[::-1]
    top_peaks = peaks[sorted_idx[:n_candidates]]
    top_freqs = freqs[top_peaks]
    top_proms = prominences[sorted_idx[:n_candidates]]
    return top_freqs, top_peaks, hist_norm


def plot_tonic_histogram(hist_norm, freqs, candidate_peaks, candidate_freqs):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(freqs, hist_norm, color='steelblue', linewidth=0.8, alpha=0.8)
    ax.fill_between(freqs, hist_norm, alpha=0.3, color='steelblue')
    colors = ['red', 'orange', 'green', 'purple', 'brown']
    for i, (peak_bin, freq) in enumerate(zip(candidate_peaks, candidate_freqs)):
        midi = librosa.hz_to_midi(freq)
        note = librosa.midi_to_note(int(round(midi)))
        label = f"#{i+1}: {freq:.1f} Hz ({note})"
        ax.axvline(x=freq, color=colors[i % len(colors)],
                   linewidth=1.5, linestyle='--', alpha=0.8, label=label)
    ax.set_xscale('log')
    ax.set_xlabel("Frequency (Hz) — log scale")
    ax.set_ylabel("Normalised energy")
    ax.set_title("Pitch histogram — tonic candidates marked")
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(80, 800)
    plt.tight_layout()
    return fig


def build_folded_histogram(audio, sr, fmin=80, fmax=800, bins_per_octave=120):
    hop_length = 512
    n_octaves = int(np.log2(fmax / fmin)) + 1
    n_bins = bins_per_octave * n_octaves
    cqt = librosa.cqt(audio, sr=sr, hop_length=hop_length, fmin=fmin,
                      n_bins=n_bins, bins_per_octave=bins_per_octave)
    cqt_mag = np.abs(cqt)
    full_hist = np.sum(cqt_mag, axis=1)
    folded = np.zeros(bins_per_octave)
    for i in range(n_octaves):
        start = i * bins_per_octave
        end = start + bins_per_octave
        if end <= len(full_hist):
            folded += full_hist[start:end]
    cent_axis = np.linspace(0, 1200, bins_per_octave, endpoint=False)
    folded = folded / folded.max()
    return folded, cent_axis, full_hist, fmin, bins_per_octave


def detect_tonic_folded(folded_hist, cent_axis, fmin=80, bins_per_octave=120, n_candidates=5):
    peaks, properties = find_peaks(folded_hist, prominence=0.03, distance=4)
    if len(peaks) == 0:
        peaks = np.array([np.argmax(folded_hist)])
        properties = {'prominences': np.array([1.0])}
    prominences = properties['prominences']
    sorted_idx = np.argsort(prominences)[::-1]
    top_peaks = peaks[sorted_idx[:n_candidates]]
    results = []
    for peak_bin in top_peaks:
        cent_offset = cent_axis[peak_bin]
        ref_hz = fmin * (2 ** (cent_offset / 1200))
        while ref_hz < 100:
            ref_hz *= 2
        while ref_hz > 350:
            ref_hz /= 2
        results.append((ref_hz, cent_offset, prominences[sorted_idx[list(top_peaks).index(peak_bin)]]))
    return results


def estimate_tonic_from_pyin(audio, sr, fmin=80, fmax=500):
    f0, voiced_flag, voiced_prob = librosa.pyin(audio, fmin=fmin, fmax=fmax,
                                                 sr=sr, hop_length=512)
    voiced_f0 = f0[voiced_flag & (voiced_prob > 0.7)]
    voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]
    if len(voiced_f0) == 0:
        return None
    hist, bin_edges = np.histogram(voiced_f0, bins=200)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return voiced_f0, hist, bin_centers


def plot_tonic_combined(folded_hist, cent_axis, tonic_results, pyin_result):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(cent_axis, folded_hist, color='steelblue', linewidth=1.0)
    axes[0].fill_between(cent_axis, folded_hist, alpha=0.3, color='steelblue')
    axes[0].set_title("Octave-folded pitch histogram (0–1200 cents within one octave)")
    axes[0].set_xlabel("Cents within octave")
    axes[0].set_ylabel("Normalised energy")
    axes[0].grid(True, alpha=0.3)
    if tonic_results:
        best_cents = tonic_results[0][1]
        axes[0].axvline(x=best_cents, color='red', linewidth=2, linestyle='--',
                        label=f"Best Sa candidate: {tonic_results[0][0]:.1f} Hz")
        axes[0].legend()
    if pyin_result:
        voiced_f0, hist, bin_centers = pyin_result
        axes[1].bar(bin_centers, hist, width=(bin_centers[1] - bin_centers[0]),
                    color='darkorange', alpha=0.7)
        axes[1].set_title("pYIN pitch histogram — lower register (80–350 Hz)")
        axes[1].set_xlabel("Frequency (Hz)")
        axes[1].set_ylabel("Frame count")
        axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def hz_to_cents(f0_hz, tonic_hz):
    with np.errstate(divide='ignore', invalid='ignore'):
        cents = 1200.0 * np.log2(f0_hz / tonic_hz)
    return cents


def assign_swara(cents_from_sa):
    if np.isnan(cents_from_sa):
        return 'unvoiced', np.nan
    c = cents_from_sa % 1200
    best_label = 'transit'
    best_dist = BOUNDARY_CENTS
    for name, swara_c in SWARA_CENTS.items():
        dist = abs(c - swara_c)
        if name == 'Sa':
            dist = min(dist, abs(c - 1200))
        if dist < best_dist:
            best_dist = dist
            best_label = name
    return best_label, best_dist if best_label != 'transit' else np.nan


def resolve_swara_conflicts(swaras, counts):
    groups = {
        'Ri': ['Ri1', 'Ri2', 'Ri3'], 'Ga': ['Ga1', 'Ga2', 'Ga3'],
        'Da': ['Da1', 'Da2', 'Da3'], 'Ni': ['Ni1', 'Ni2', 'Ni3'],
        'Ma': ['Ma1', 'Ma2'],
    }
    result = set(swaras)
    for group_name, variants in groups.items():
        present = [v for v in variants if v in result]
        if len(present) > 1:
            best = max(present, key=lambda v: counts.get(v, 0))
            for v in present:
                if v != best:
                    result.discard(v)
    return list(result)


def get_dominant_swaras(labels, min_percent=MIN_PRESENCE_PERCENT):
    voiced_labels = [l for l in labels if l not in ('unvoiced', 'transit')]
    total_voiced = len(voiced_labels)
    if total_voiced == 0:
        return []
    counts = Counter(voiced_labels)
    dominant = []
    for swara, count in counts.items():
        pct = (count / total_voiced) * 100
        if pct >= min_percent and swara != "Sa'":
            dominant.append(swara)
    dominant = resolve_swara_conflicts(dominant, counts)
    return sorted(dominant)


def get_base_swara(label):
    if label in ['Sa', 'Pa', 'unvoiced']:
        return label
    return ''.join([c for c in label if not c.isdigit()])


def plot_f0_curve(time_arr, f0_cents, tonic_hz, title="F0 pitch curve", max_seconds=60):
    display_swaras = {
        'Sa': 0, 'Ri1/Ga1': 112, 'Ri2/Ga2': 204, 'Ga3': 386,
        'Ma1': 498, 'Ma2': 590, 'Pa': 702,
        'Da1/Ni1': 814, 'Da2/Ni2': 906, 'Ni3': 1088, "Sa'": 1200
    }
    step = time_arr[1] - time_arr[0]
    n = min(len(time_arr), int(max_seconds / step))
    t = time_arr[:n]
    c = f0_cents[:n]

    fig, ax = plt.subplots(figsize=(16, 7))
    for name, cents in display_swaras.items():
        ax.axhline(y=cents, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.text(0.5, cents + 8, name, fontsize=7, color='gray', alpha=0.8)
        ax.axhline(y=cents - 1200, color='gray', linewidth=0.4, linestyle=':', alpha=0.3)
    ax.scatter(t, c, s=0.3, color='steelblue', alpha=0.6, label='F0 (cents)')
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Cents relative to Sa (0 = Sa)")
    ax.set_title(f"{title}  |  Sa = {tonic_hz:.1f} Hz  |  First {max_seconds}s shown")
    ax.set_ylim(-800, 1400)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_swara_sequence(time_arr, labels, f0_cents, tonic_hz,
                        title="Swara sequence", max_seconds=60):
    step = time_arr[1] - time_arr[0]
    n = min(len(time_arr), int(max_seconds / step))

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle(f"{title}  |  Sa = {tonic_hz:.1f} Hz  |  First {max_seconds}s", fontsize=12)

    for i in range(n):
        if labels[i] == 'unvoiced' or np.isnan(f0_cents[i]):
            continue
        color = SWARA_COLOR_MAP.get(labels[i], '#999999')
        if color:
            axes[0].scatter(time_arr[i], f0_cents[i], color=color, s=1.5, alpha=0.8)

    for name, (cents, _) in SWARA_TABLE.items():
        if name == "Sa'":
            continue
        axes[0].axhline(y=cents, color='gray', linewidth=0.4, linestyle='--', alpha=0.4)
        axes[0].text(0.2, cents + 6, name, fontsize=6.5, color='gray', alpha=0.7)

    axes[0].set_ylabel("Cents relative to Sa")
    axes[0].set_ylim(-300, 1350)
    axes[0].set_title("F0 pitch curve — colored by swara")
    axes[0].grid(True, alpha=0.15)

    swara_order = ['Sa', 'Ri1', 'Ri2', 'Ri3', 'Ga1', 'Ga2', 'Ga3',
                   'Ma1', 'Ma2', 'Pa', 'Da1', 'Da2', 'Da3', 'Ni1', 'Ni2', 'Ni3', "Sa'"]
    swara_y = {s: i for i, s in enumerate(swara_order)}

    for i in range(n):
        if labels[i] in ('unvoiced', 'transit'):
            continue
        y = swara_y.get(labels[i], -1)
        color = SWARA_COLOR_MAP.get(labels[i], '#999999')
        axes[1].bar(time_arr[i], 0.8, bottom=y, width=step, color=color, alpha=0.7)

    axes[1].set_yticks(range(len(swara_order)))
    axes[1].set_yticklabels(swara_order, fontsize=8)
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_title("Swara labels over time")
    axes[1].grid(True, alpha=0.15, axis='x')

    patches = [mpatches.Patch(color=c, label=s)
               for s, c in SWARA_COLOR_MAP.items()
               if c and s not in ('transit',)]
    axes[0].legend(handles=patches, fontsize=7, loc='upper right', ncol=4)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def reset_state():
    for key in ['audio', 'sr', 'song_name', 'tonic_hz', 'tonic_note',
                'f0_hz', 'f0_cents', 'time_arr', 'labels_fixed',
                'labels_corrected', 'dominant_swaras', 'pipeline_done']:
        st.session_state.pop(key, None)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎵 Swara Extraction Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Carnatic Music Analysis · Pitch Detection · Swara Labelling</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    uploaded = st.file_uploader("Upload Audio File", type=["mp3", "wav", "ogg", "flac", "m4a"])
    confidence_threshold = st.slider("CREPE Confidence Threshold", 0.3, 0.9, 0.6, 0.05)
    max_display_seconds = st.slider("Plot Window (seconds)", 20, 120, 60, 10)
    tonic_override = st.number_input("Override Tonic (Hz, 0 = auto)", min_value=0.0, max_value=500.0, value=0.0, step=1.0)

    if uploaded and st.button("🔄 Reset & Re-upload"):
        reset_state()
        st.rerun()

    st.markdown("---")
    st.markdown("**Pipeline Steps**")
    st.markdown("""
    1. Load Audio  
    2. Waveform Plot  
    3. Tonic Detection  
    4. Pitch Extraction (pYIN)  
    5. Cents Conversion  
    6. Swara Assignment  
    7. Distinct Swaras  
    8. First 20 Swara Sequence  
    """)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────
if uploaded is None:
    st.info("👈 Upload an audio file from the sidebar to begin analysis.")
    st.stop()

# ─── STEP 1: LOAD AUDIO ───
if 'audio' not in st.session_state:
    st.markdown('<div class="section-header">Step 1 · Loading Audio</div>', unsafe_allow_html=True)
    with st.spinner("Loading audio..."):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[-1])
        tmp.write(uploaded.read())
        tmp.close()
        audio, sr = load_audio(tmp.name)
        os.unlink(tmp.name)

        st.session_state['audio'] = audio
        st.session_state['sr'] = sr
        st.session_state['song_name'] = os.path.splitext(uploaded.name)[0]

audio = st.session_state['audio']
sr = st.session_state['sr']
song_name = st.session_state['song_name']

duration = len(audio) / sr
c1, c2, c3 = st.columns(3)
c1.metric("Duration", f"{duration:.1f}s")
c2.metric("Sample Rate", f"{sr} Hz")
c3.metric("Samples", f"{len(audio):,}")

# ─── STEP 2: WAVEFORM PLOT ───
st.markdown('<div class="section-header">Step 2 · Waveform & Energy Envelope</div>', unsafe_allow_html=True)
with st.spinner("Plotting waveform..."):
    fig_wave = plot_waveform(audio, sr, title=song_name, max_display_seconds=max_display_seconds)
    st.pyplot(fig_wave)
    plt.close(fig_wave)

# ─── STEP 3: TONIC DETECTION ───
st.markdown('<div class="section-header">Step 3 · Tonic (Sa) Detection</div>', unsafe_allow_html=True)

if 'tonic_hz' not in st.session_state:
    with st.spinner("Building pitch histogram..."):
        pitch_hist, freq_axis = build_pitch_histogram(audio, sr)
        tonic_candidates, candidate_bins, hist_norm = detect_tonic(pitch_hist, freq_axis)

    st.markdown("**Pitch Histogram — Top Tonic Candidates**")
    fig_hist = plot_tonic_histogram(hist_norm, freq_axis, candidate_bins, tonic_candidates)
    st.pyplot(fig_hist)
    plt.close(fig_hist)

    with st.spinner("Building octave-folded histogram & running pYIN..."):
        folded_hist, cent_axis, full_hist, fmin_used, bpo = build_folded_histogram(audio, sr)
        tonic_results = detect_tonic_folded(folded_hist, cent_axis)
        pyin_result = estimate_tonic_from_pyin(audio, sr, fmin=80, fmax=350)

    st.markdown("**Octave-Folded Histogram & pYIN Comparison**")
    fig_combined = plot_tonic_combined(folded_hist, cent_axis, tonic_results, pyin_result)
    st.pyplot(fig_combined)
    plt.close(fig_combined)

    # Determine tonic
    if tonic_override > 0:
        TONIC_HZ = float(tonic_override)
    elif pyin_result:
        voiced_f0, hist, bin_centers = pyin_result
        TONIC_HZ = float(bin_centers[np.argmax(hist)])
    else:
        TONIC_HZ = tonic_results[0][0]

    midi_val = librosa.hz_to_midi(TONIC_HZ)
    note_name = librosa.midi_to_note(int(round(midi_val)))

    st.session_state['tonic_hz'] = TONIC_HZ
    st.session_state['tonic_note'] = note_name
    st.session_state['folded_hist'] = folded_hist
    st.session_state['cent_axis'] = cent_axis
    st.session_state['tonic_results'] = tonic_results
    st.session_state['pyin_result'] = pyin_result
    st.session_state['pitch_hist'] = pitch_hist
    st.session_state['freq_axis'] = freq_axis
    st.session_state['hist_norm'] = hist_norm
    st.session_state['candidate_bins'] = candidate_bins
    st.session_state['tonic_candidates'] = tonic_candidates

else:
    TONIC_HZ = st.session_state['tonic_hz']
    note_name = st.session_state['tonic_note']

    # Re-draw plots from cached data
    fig_hist = plot_tonic_histogram(
        st.session_state['hist_norm'], st.session_state['freq_axis'],
        st.session_state['candidate_bins'], st.session_state['tonic_candidates']
    )
    st.pyplot(fig_hist)
    plt.close(fig_hist)

    fig_combined = plot_tonic_combined(
        st.session_state['folded_hist'], st.session_state['cent_axis'],
        st.session_state['tonic_results'], st.session_state['pyin_result']
    )
    st.pyplot(fig_combined)
    plt.close(fig_combined)

    if tonic_override > 0:
        TONIC_HZ = float(tonic_override)
        midi_val = librosa.hz_to_midi(TONIC_HZ)
        note_name = librosa.midi_to_note(int(round(midi_val)))
        st.session_state['tonic_hz'] = TONIC_HZ
        st.session_state['tonic_note'] = note_name

ta, tb, tc = st.columns(3)
ta.metric("Tonic (Sa)", f"{TONIC_HZ:.2f} Hz")
tb.metric("MIDI Note", f"{librosa.hz_to_midi(TONIC_HZ):.1f}")
tc.metric("Western Note", note_name)

# ─── STEP 4: PITCH EXTRACTION ───
st.markdown('<div class="section-header">Step 4 · Pitch Extraction (pYIN)</div>', unsafe_allow_html=True)

if 'f0_hz' not in st.session_state:
    with st.spinner("Extracting F0 with pYIN (this may take a minute)..."):
        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'), sr=sr, hop_length=512
        )
        time_arr = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=512)
        f0_filtered = f0.copy()
        f0_filtered[voiced_prob < confidence_threshold] = np.nan
        f0_cents = hz_to_cents(f0_filtered, TONIC_HZ)

        st.session_state['f0_hz'] = f0_filtered
        st.session_state['f0_cents'] = f0_cents
        st.session_state['time_arr'] = time_arr

f0_filtered = st.session_state['f0_hz']
f0_cents = st.session_state['f0_cents']
time_arr = st.session_state['time_arr']

voiced_frames = np.sum(~np.isnan(f0_filtered))
voiced_pct = voiced_frames / len(f0_filtered) * 100
p1, p2, p3 = st.columns(3)
p1.metric("Total Frames", f"{len(f0_filtered):,}")
p2.metric("Voiced Frames", f"{voiced_frames:,} ({voiced_pct:.1f}%)")
p3.metric("F0 Range", f"{np.nanmin(f0_filtered):.1f}–{np.nanmax(f0_filtered):.1f} Hz")

# ─── STEP 5: F0 CENTS PLOT ───
st.markdown('<div class="section-header">Step 5 · F0 Pitch Curve (Cents)</div>', unsafe_allow_html=True)
with st.spinner("Plotting F0 curve..."):
    fig_f0 = plot_f0_curve(time_arr, f0_cents, TONIC_HZ, title=song_name, max_seconds=max_display_seconds)
    st.pyplot(fig_f0)
    plt.close(fig_f0)

# ─── STEP 6: SWARA ASSIGNMENT ───
st.markdown('<div class="section-header">Step 6 · Swara Assignment</div>', unsafe_allow_html=True)

if 'labels_fixed' not in st.session_state:
    with st.spinner("Assigning swaras to each frame..."):
        labels_fixed = []
        for cents_val in f0_cents:
            if np.isnan(cents_val):
                labels_fixed.append('unvoiced')
            else:
                lbl, _ = assign_swara(cents_val)
                labels_fixed.append(lbl)
        labels_fixed = np.array(labels_fixed)

        dominant_swaras = get_dominant_swaras(labels_fixed)
        dominant_map = {}
        for sw in dominant_swaras:
            base = get_base_swara(sw)
            dominant_map[base] = sw

        labels_corrected = []
        for lbl in labels_fixed:
            if lbl == 'unvoiced':
                labels_corrected.append(lbl)
                continue
            base = get_base_swara(lbl)
            if base in dominant_map:
                labels_corrected.append(dominant_map[base])
            else:
                labels_corrected.append(lbl)
        labels_corrected = np.array(labels_corrected)

        st.session_state['labels_fixed'] = labels_fixed
        st.session_state['labels_corrected'] = labels_corrected
        st.session_state['dominant_swaras'] = dominant_swaras

labels_fixed = st.session_state['labels_fixed']
labels_corrected = st.session_state['labels_corrected']
dominant_swaras = st.session_state['dominant_swaras']

with st.spinner("Plotting swara sequence..."):
    fig_swara = plot_swara_sequence(
        time_arr, labels_fixed, f0_cents, TONIC_HZ,
        title=song_name, max_seconds=max_display_seconds
    )
    st.pyplot(fig_swara)
    plt.close(fig_swara)

# ─── STEP 7: DISTINCT SWARAS ───
st.markdown('<div class="section-header">Step 7 · Distinct Swaras Detected</div>', unsafe_allow_html=True)

voiced_labels = [l for l in labels_corrected if l not in ('unvoiced', 'transit')]
total_voiced = len(voiced_labels)
counts = Counter(voiced_labels)

pills_html = ""
for sw in sorted(dominant_swaras):
    color = SWARA_COLOR_MAP.get(sw, '#888888')
    pct = counts.get(sw, 0) / total_voiced * 100 if total_voiced > 0 else 0
    pills_html += (
        f'<span class="swara-pill" style="background:{color}22; '
        f'border: 2px solid {color}; color:{color};">'
        f'{sw} <small style="opacity:0.7">({pct:.1f}%)</small></span>'
    )

st.markdown(pills_html, unsafe_allow_html=True)

# Swara frequency table
st.markdown("**Swara Presence (voiced frames only)**")
swara_rows = []
for sw in sorted(dominant_swaras):
    cnt = counts.get(sw, 0)
    pct = cnt / total_voiced * 100 if total_voiced > 0 else 0
    swara_rows.append({"Swara": sw, "Frames": cnt, "Presence (%)": f"{pct:.2f}%"})

if swara_rows:
    import pandas as pd
    df = pd.DataFrame(swara_rows).sort_values("Frames", ascending=False).reset_index(drop=True)
    st.dataframe(df, use_container_width=True, hide_index=True)

# ─── STEP 8: FIRST 20 SWARAS ───
st.markdown('<div class="section-header">Step 8 · Sequence of First 20 Swaras</div>', unsafe_allow_html=True)

st.markdown("""
The first 20 distinct swara labels (excluding silent/unvoiced frames, showing consecutive swara changes):
""")

# Build sequence of distinct consecutive swaras (not just raw labels)
sequence_labels = []
prev = None
for lbl in labels_corrected:
    if lbl == 'unvoiced':
        prev = None  # reset so next voiced is counted fresh
        continue
    if lbl == 'transit':
        continue
    if lbl != prev:
        sequence_labels.append(lbl)
        prev = lbl
    if len(sequence_labels) >= 20:
        break

st.markdown("**First 20 distinct swara transitions:**")
seq_pills = ""
for i, sw in enumerate(sequence_labels, 1):
    color = SWARA_COLOR_MAP.get(sw, '#888888')
    seq_pills += (
        f'<span class="swara-pill" style="background:{color}33; '
        f'border: 2px solid {color}; color:{color}; font-size:1.1rem;">'
        f'<span style="opacity:0.5; font-size:0.7rem;">{i}.</span> {sw}</span>'
    )
st.markdown(seq_pills + "<br><br>", unsafe_allow_html=True)

# Also show raw first-20 frame labels (as notebook does: labels_corrected[:20])
st.markdown("**Raw first 20 frame labels** (direct slice, as in notebook output):")
raw_20 = list(labels_corrected[:20])
raw_pills = ""
for lbl in raw_20:
    color = SWARA_COLOR_MAP.get(lbl, '#888888') or '#888888'
    raw_pills += (
        f'<span class="swara-pill" style="background:{color}22; '
        f'border: 1px solid {color}55; color:{color}; font-size:0.85rem;">{lbl}</span>'
    )
st.markdown(raw_pills, unsafe_allow_html=True)

# ─── FOOTER ───
st.markdown("---")
st.markdown(
    '<center style="color:#aaa; font-size:0.85rem;">'
    'Swara Extraction Pipeline · Built with librosa + pYIN · Carnatic Music Analysis'
    '</center>',
    unsafe_allow_html=True
)
