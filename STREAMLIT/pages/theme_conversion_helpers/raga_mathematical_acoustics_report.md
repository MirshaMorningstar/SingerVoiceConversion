# Mathematical & Acoustic Specifications: AI Raga Voice Changer



## 1. Acoustic DSP Parameters (Feature Extraction)
The following mathematical constants and parameters are used within the `librosa` audio processing pipeline to extract pitches and swaras accurately.

| Parameter | Mathematical Value / Formula | Purpose |
| :--- | :--- | :--- |
| **Sampling Rate ($f_s$)** | $16,000$ Hz | Downsampled from 44.1kHz to isolate vocal frequencies while optimizing Chroma-STFT processing speed. |
| **Nyquist Frequency ($f_N$)** | $16,000 / 2 = 8,000$ Hz | The maximum analyzable acoustic frequency in the signal. |
| **STFT Window Size ($N_{FFT}$)** | $4096$ samples | Determines the frequency resolution (approx. $\Delta f = 16000 / 4096 \approx 3.9$ Hz). Highly accurate for pitch detection. |
| **Hop Length ($H$)** | $2048$ samples | Overlap size for STFT frames. Dictates time resolution ($\Delta t = 2048 / 16000 = 128$ ms). |
| **Note Frame Rate** | $1024$ samples ($64$ ms) | Used specifically for the fast frame-by-frame Swara timeline generator. |
| **Chroma Frequency Bins** | $12$ bins | Each bin represents one semitone (pitch class) in an octave. |

---

## 2. Mathematical Swara to Frequency Mapping (12-TET)
In Carnatic music theory applied to Western DSP, the 12 semitones are mapped using the 12-Tone Equal Temperament (12-TET) mathematical formula: 
$ f(n) = f_0 \times 2^{(n/12)} $
where $n$ is the semitone distance from the root note (Tonic / Shadjam $f_0$).

*(Assuming a standard 3-Shruti Tonic $f_0 = 246.94$ Hz for mapping illustration)*

| $n$ (Semitone) | Carnatic Swara | Math Ratio ($2^{n/12}$) | Approx. Frequency (Hz) |
| :---: | :--- | :---: | :---: |
| 0 | Sa (Shadjam) | $1.000$ | $246.94$ |
| 1 | Ri1 / Shuddha Rishabham | $1.059$ | $261.63$ |
| 2 | Ri2 / Chatushruti Rishabham | $1.122$ | $277.18$ |
| 3 | Ga2 / Sadharana Gandharam | $1.189$ | $293.66$ |
| 4 | Ga3 / Antara Gandharam | $1.260$ | $311.13$ |
| 5 | Ma1 / Shuddha Madhyamam | $1.335$ | $329.63$ |
| 6 | Ma2 / Prati Madhyamam | $1.414$ | $349.23$ |
| 7 | Pa (Panchamam) | $1.498$ | $369.99$ |
| 8 | Da1 / Shuddha Dhaivatam | $1.587$ | $392.00$ |
| 9 | Da2 / Chatushruti Dhaivatam | $1.682$ | $415.30$ |
| 10 | Ni2 / Kaisika Nishadam | $1.782$ | $440.00$ |
| 11 | Ni3 / Kakali Nishadam | $1.888$ | $466.16$ |

---

## 3. Raga Sequence Chromagram Matrix (Template Vectors)
For Emotion classification, the system uses a mathematical One-Hot Encoded Chromagram Matrix. Each row represents a Raga reference Vector $\mathbf{R} = [r_0, r_1, \dots, r_{11}]$. 
A `1` indicates the swara (frequency bin) is active.

| Emotion | Raga | 0 (Sa) | 1 | 2 | 3 | 4 | 5 | 6 | 7 (Pa) | 8 | 9 | 10 | 11 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Happy** | Mohanam | **1** | 0 | 0 | 0 | **1** | 0 | 0 | **1** | 0 | **1** | 0 | **1** |
| **Sad** | Bhairavi | **1** | 0 | **1** | **1** | 0 | **1** | 0 | **1** | **1** | 0 | **1** | 0 |
| **Angry** | Dhanyasi | **1** | 0 | **1** | **1** | 0 | 0 | 0 | **1** | **1** | 0 | 0 | 0 |
| **Fearful** | Todi | **1** | **1** | 0 | **1** | 0 | **1** | 0 | **1** | **1** | 0 | 0 | **1** |
| **Disgusted** | Kambhoji | **1** | 0 | **1** | 0 | **1** | **1** | 0 | **1** | 0 | **1** | **1** | 0 |
| **Surprised** | Kalyani | **1** | 0 | **1** | 0 | **1** | 0 | **1** | **1** | 0 | **1** | 0 | **1** |
| **Peaceful** | Hamsadhwani | **1** | 0 | 0 | 0 | **1** | 0 | 0 | **1** | 0 | 0 | 0 | **1** |
| **Romantic** | Kharaharapriya | **1** | 0 | **1** | **1** | 0 | **1** | 0 | **1** | 0 | **1** | **1** | 0 |

---

## 4. Evaluation Algorithm Mathematical Formulae

### A. Emotion Classification via Cosine Similarity
The system identifies the emotion by finding the closest match between the extracted song Chroma Vector ($\mathbf{C}$) and the Raga Template Vector ($\mathbf{R}$) using the Cosine Similarity metric:
$$ \text{Similarity Score} = \frac{\mathbf{C} \cdot \mathbf{R}}{||\mathbf{C}|| \times ||\mathbf{R}||} = \frac{\sum_{i=0}^{11} (c_i \times r_i)}{\sqrt{\sum c_i^2} \times \sqrt{\sum r_i^2}} $$

### B. Pitch Shifting (Raga Conversion) Frame Operation
When converting a swara (e.g., $S_{old}$ to $S_{new}$), the algorithmic pitch shift ($P_S$) in semi-tones is calculated as:
$$ P_S = S_{new} - S_{old} $$

The Phase Vocoder algorithm applies this by scaling the frequency $f$ by a shift factor $\alpha$:
$$ \alpha = 2^{\frac{P_S}{12}} $$
$$ f_{new} = f \times \alpha $$

### C. Ground Truth Evaluation Metrics
When comparing the algorithm's detected swaras versus the manual human transcriptions (ground truth):

- **True Positives ($TP$)**: Notes in both Ground Truth and System Extraction.
- **False Positives ($FP$)**: Notes extracted by System but NOT in Ground Truth (Hallucinated notes).
- **False Negatives ($FN$)**: Notes in Ground Truth missed by System.

**Precision**: The accuracy of the extracted notes.
$$ \text{Precision} = \frac{TP}{TP + FP} $$

**Recall**: How much of the actual song was correctly captured.
$$ \text{Recall} = \frac{TP}{TP + FN} $$

**F1-score**: The harmonic mean of Precision and Recall.
$$ F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$

---
## 5. Temporal Dynamic Constants
To map the behavioral aspects of an emotion, the overall audio reconstruction timeline $T$ is multiplied by an **Emotion Tempo Scaling Factor ($\gamma$)**:
$$ Tempo_{new} (BPM) = Tempo_{original} \times \gamma $$

| Target Emotion | Tempo Factor ($\gamma$) | Mathematical Effect on Signal Timeline |
| :--- | :---: | :--- |
| Happy | $1.00$ | $T_{new} = T_{old}$ (No change) |
| Sad | $0.85$ | $T_{new} = 1.17 \times T_{old}$ (Stretched / Slower by 15%) |
| Angry | $1.10$ | $T_{new} = 0.90 \times T_{old}$ (Compressed / Faster by 10%) |
| Fearful | $0.90$ | $T_{new} = 1.11 \times T_{old}$ (Stretched / Slower by 10%) |
| Surprised | $1.05$ | $T_{new} = 0.95 \times T_{old}$ (Compressed / Faster by 5%) |
| Peaceful | $0.88$ | $T_{new} = 1.13 \times T_{old}$ (Stretched / Slower by 12%) |
| Romantic | $0.93$ | $T_{new} = 1.07 \times T_{old}$ (Stretched / Slower by 7%) |
| Disgusted | $0.95$ | $T_{new} = 1.05 \times T_{old}$ (Stretched / Slower by 5%) |
