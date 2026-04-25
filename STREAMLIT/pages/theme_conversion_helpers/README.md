# AI Raga Voice Changer 🎵
### Final Year Project — Tamil Song Emotion Converter

---

## Project Overview

This system converts Tamil songs between **8 emotional states** by analysing and transforming
the underlying **raga** (melodic framework) at the **swara (note) level**.

Unlike simple pitch-shift or tempo-change tools, this system:
1. **Identifies** which raga the input song belongs to
2. **Maps** that raga to an emotion
3. **Transforms only the notes that differ** between the source and target raga
4. **Produces a musically valid output** in the target raga

---

## The 8 Emotions & Their Ragas

| Emotion    | Raga           | Key Characteristic                          |
|------------|----------------|---------------------------------------------|
| Happy      | Mohanam        | Pentatonic – bright, celebratory            |
| Sad        | Bhairavi       | Flat 3rd/6th/7th – melancholic, grief       |
| Angry      | Dhanyasi       | Minor pentatonic – fierce, intense          |
| Fearful    | Todi           | Augmented 4th – eerie, anxious              |
| Disgusted  | Kambhoji       | Ni-absent ascent – heavy, unresolved        |
| Surprised  | Kalyani        | Raised 4th (Ma2) – bright, wonder           |
| Peaceful   | Hamsadhwani    | 4-note – serene, devotional                 |
| Romantic   | Kharaharapriya | Natural minor + Da1 – intimate, longing     |

---

## Technical Pipeline

```
INPUT SONG (MP3/WAV)
        │
        ▼
[1] FEATURE EXTRACTION (librosa)
    • Load audio → mono waveform
    • Chroma CQT → 12-dim pitch class energy vector
    • Beat tracking → tempo (BPM)
    • Argmax of chroma → tonic (Sa) estimation
        │
        ▼
[2] RAGA IDENTIFICATION
    • Normalise chroma vector to tonic
    • Cosine similarity vs each raga's binary swara vector
    • Rank ragas by similarity score → detect source raga & emotion
        │
        ▼
[3] NOTE CHANGE ANALYSIS
    • Compare source raga swaras vs target raga swaras
    • Identify: notes to remove, notes to add, notes shared
    • Compute semitone shift to nearest target note
        │
        ▼
[4] AUDIO TRANSFORMATION
    • Per-frame pitch tracking (piptrack)
    • Dominant shift calculation
    • librosa.effects.pitch_shift → note-level shift
    • librosa.effects.time_stretch → tempo scaling
        │
        ▼
OUTPUT SONG (WAV) + ANALYSIS TABLE
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the app
```bash
streamlit run app.py
```

### 3. Use the app
- **Convert Song tab**: Upload any Tamil MP3/WAV → select emotion → convert & play
- **Raga–Emotion Table tab**: Master comparison table of all 8 ragas
- **Batch Results tab**: All converted songs in one downloadable table
- **Note Change Comparison tab**: See exactly which notes change between any two emotions

---

## Project Structure

```
ai_raga_voice_changer/
├── app.py                          ← Streamlit UI (run this)
├── requirements.txt
├── README.md
└── core/
    ├── __init__.py
    ├── raga_knowledge_base.py      ← Raga definitions, swara mappings, 8-emotion table
    ├── raga_extractor.py           ← Audio feature extraction & raga identification
    ├── raga_transformer.py         ← Note-level pitch transformation
    └── report_generator.py         ← Academic tables & batch reports
```

---

## Answering Your Mam's Questions

**Q: "What is the extraction mechanism?"**
→ Chroma CQT (Constant-Q Transform) extracts the 12 pitch-class energy profile of the song.
   The tonic (Sa) is estimated as the most energetic pitch class.
   The chroma vector is compared to raga templates via cosine similarity.

**Q: "How is it telling that the raga is getting converted?"**
→ The system uses cosine similarity between the extracted chroma vector and
   binary swara presence vectors for each of the 8 ragas. The highest-scoring raga is selected.

**Q: "Which part of the note is getting changed?"**
→ Only notes that exist in the source raga but NOT in the target raga are shifted.
   Each such note is moved to the nearest note that belongs to the target raga.
   The Note Change Comparison tab shows this for every emotion pair.

**Q: "Comparison table — which note is getting changed across 8 emotions?"**
→ See the Raga–Emotion Table tab and Note Change Comparison tab in the app.

**Q: "What is used to get extracted from song to raga?"**
→ librosa's `chroma_cqt` function extracts the pitch class energy.
   Beat tracking gives tempo. `piptrack` gives frame-by-frame pitch.

---

*Built with librosa, soundfile, numpy, pandas, streamlit*
