# Acoustic Feature Analysis Breakdown

Based on the investigation of [mega_extractor.py](file:///d:/RagaVoiceStudio/mega_extractor.py) and the generated CSV, the 317-dimensional Identity Profile Vector is constructed from a comprehensive set of temporal, spectral, and voice-quality attributes. The breakdown is as follows:

### 1. Pitch & Vibrato Features (15 Features)
Captures the fundamental frequency (F0) and how the singer modulates their pitch over time (essential for capturing unique Ragas and Gamakas).
- **F0 Statistics (10):** mean, std, min, max, range, 5th & 95th percentiles, Interquartile Range (IQR), skewness, kurtosis.
- **Vibrato (3):** `vibrato_rate`, `vibrato_depth`, `f0_modulation_variance`.
- **Voicing (2):** `voiced_ratio`, `pitch_stability`.

### 2. Formant Features (32 Features)
Formants (F1, F2, F3) represent the resonant frequencies of the singer's vocal tract, fundamentally dictating vowel articulation and voice timbre.
- **F1, F2, F3 Statistics (30):** Full 10 statistical metrics (mean, std, min, max, etc.) for each of the first three formants.
- **Distances (2):** `F1_F2_distance_mean`, `F2_F3_distance_mean` (determines vowel spacing/warmth).

### 3. Voice Quality / Acoustic Perturbation (3 Features)
Extracted via Praat, these measure the micro-instabilities in the vocal folds (breathiness, roughness).
- **`jitter_mean`**: Frequency micro-variations.
- **`shimmer_mean`**: Amplitude micro-variations.
- **`HNR_mean`**: Harmonics-to-Noise Ratio (measures voice clarity vs. breathiness).

### 4. Spectral Features (110 Features)
Captures the overall shape, brightness, and texture of the voice spectrum.
- **Global Spectral Shape (40):** 10 statistical metrics each for Spectral Centroid (brightness), Bandwidth (spread), Rolloff (high-frequency presence), and Flatness (tonality vs. noise).
- **Spectral Contrast (70):** Evaluates peak-to-valley energy differences across 7 distinct frequency bands, yielding 10 statistical metrics per band.

### 5. MFCC Features (156 Features)
Mel-Frequency Cepstral Coefficients (MFCCs) compactly represent the short-term power spectrum of the voice based on human auditory perception. These strongly dictate singer identity in machine learning models.
- **Static Coefficients (130):** 10 statistical metrics for 13 MFCC bands (`mfcc1` through `mfcc13`).
- **Dynamic Coefficients (26):** Deltas (`delta_mean`) and Delta-deltas (`delta2_mean`) for each of the 13 MFCCs, capturing the speech rate and transition speeds.

**Total Features: 15 + 32 + 3 + 110 + 156 = 316 + 1 (Singer Label) = 317 Features.**

This highly granular array is perfectly formulated for the $N \times N$ matrix transformation since it comprehensively spaces out all physical and stylistic traits of the singer's voice.
