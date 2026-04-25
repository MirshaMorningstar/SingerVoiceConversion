# Singer Voice Conversion (SVC) Refinement Strategy

Based on your professor's feedback, the Seed-VC model relies on a diffusion-based transformer approach that doesn't completely overwrite the source voice characteristics (leaving ~30% leakage) and doesn't enforce the strict frequency/raga profiles unique to legendary singers like ARR, Ilayaraja, Mano, and SPB.

To address this, we will implement a robust **Post-Processing Acoustic Pipeline** that completely de-identifies the source voice and morphs it into the target acoustic space using linear algebra (Matrix operations) and signal processing (Smoothing).

## User Review Required
> [!IMPORTANT]
> Please review this methodology. This plan proposes using **Covariance Matching (Whitening & Coloring transforms)** as the $N \times N$ matrix operation to shift the identity vector of the Seed-VC output to precisely match the target singer's identity vector. 
> For re-synthesis, it assumes we can map those 317 features back to audio, or we will apply the transformations directly on the World Vocoder parameters (Harmonic spectral envelope, F0, Aperiodicity) using the learned $N \times N$ transformation matrices. Do you approve this mathematical approach?

## Proposed Changes

We will build a custom Python pipeline that sits *after* the Seed-VC output generation.

### 1. Vector Space Transformation ($N \times N$ Matrix Operations)
Seed-VC Output -> 317-D Feature Space -> Mathematical Transformation -> Target 317-D Feature Space.
*   **Whitening Transform**: Convert the Seed-VC output features into an identity covariance space. This mathematically "scrubs" or destroys the remaining 30% of the original singer's voice.
*   **Coloring Transform**: Multiply the whitened vectors by the $N \times N$ Cholesky decomposition matrix of the Target Singer's Covariance Matrix. This forces the audio features to strictly inhabit the multidimensional space of the target singer (ARR, Mano, etc.).

### 2. Frequency and Raga Smoothing
Singers sing in specific tessituras (ranges) with unique vibrato styles.
*   **F0 Histogram Matching**: Extract the Fundamental Frequency (F0) from the Seed-VC output. Map its distribution to exactly match the target singer's natural frequency distribution (preventing them from singing "out of character").
*   **Smoothing Operations**: Apply a Savitzky-Golay filter or a moving average filter to the F0 contour and Energy envelopes. This removes the "abrupt" diffusion artifacts and interpolates the pitch to respect traditional Carnatic/Raga glides (Gamakas).

### 3. Pipeline Implementation

#### [NEW] `refinement_pipeline.py`
This will be the main script executing the refinement.
*   Loads `ALL_SINGER_PROFILES.csv` to calculate target means ($\mu$) and $317 \times 317$ Covariance matrices ($\Sigma$).
*   Extracts features from the files in `dataset/converted/`.
*   Applies the Matrix projections (Whitening + Coloring).
*   Applies smoothing algorithms on the pitch contour.
*   Re-synthesizes the audio using a robust vocoder like `pyworld` or `librosa` matching the refined features.

## Verification Plan

### Automated Tests
*   Run the feature extraction on the *refined* audio and calculate the Cosine Similarity with the `ALL_SINGER_PROFILES.csv`. The similarity should ideally jump from 70% to >95%.
*   Verify that the $317 \times 317$ cross-correlation matrix of the refined output matches the target singer's matrix with high statistical significance.

### Manual Verification
*   Listen to the output to verify:
    1.  The 30% source leakage is completely gone.
    2.  The abrupt diffusion artifacts are smoothed out.
    3.  The frequency profile sounds natural and restricted to the legendary singer's known style.
