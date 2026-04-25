The Demucs model you are using (htdemucs_ft) separates each song into four stems: vocals, drums, bass, and other.
​

Main instrument groups
Vocals

Lead and many background vocals.

Spoken parts, rap, shouts, ad‑libs are usually grouped here.
​

Drums

Kick, snare, toms, hi‑hats, cymbals, percussion loops, claps, electronic drum hits.
​

Reverb and ambience tightly tied to the drum kit usually follow this stem.
​

Bass

Electric bass guitar, synth bass, acoustic/contrabass playing the low‑end line.
​

Sub‑bass and many 808‑style low elements if they behave like a bass line.
​

Other (all remaining instruments)
This is where the rest of the band/orchestra goes, including many more than 10 instruments:

Harmonic instruments: piano, keyboards, organ, electric/acoustic guitars, harp.
​

Melodic/lead instruments: flute, strings (violin, viola, cello), brass (trumpet, trombone, sax), synth leads, pads.
​

FX and texture layers: risers, whooshes, ambient pads, sound‑design elements that are not clearly drums/bass/vocals.
​

In your script, “non‑vocals” = drums + bass + other, so all these instrumental elements (piano, flute, strings, guitars, synths, drums, beat drops, etc.) are grouped together in the Non_Vocals output while the singer’s track is isolated in Vocals

RUNNING_INSTRUCTIONS:
D:\SVC\venv\Scripts\activate

###########################################################################
1. Pitch & Register Features
-----------------------------
What they capture:

Natural singing range

Upper and lower pitch limits

Register usage distribution

Pitch stability

Why important:

Some singers predominantly operate in low–mid register.

Others frequently access high register.

Pitch percentiles reflect vocal capability, not just melody.

These features quantify vocal range constraints and tendencies.

🔵 2. Vibrato & F0 Modulation
-----------------------------
What they capture:

Vibrato rate (Hz)

Vibrato depth

Micro pitch fluctuations

F0 skewness and kurtosis

Why important:

Vibrato is a strong biometric marker.

Modulation patterns differ across singers.

It reflects neuromuscular vocal control.

This captures expressive identity patterns.

🔵 3. Formant Features (Vocal Tract Signature)
----------------------------------------------------
What they capture:

F1, F2, F3 distributions

Formant spacing

Formant distances

Why important:

Formants reflect vocal tract shape.

Vocal tract structure is physiologically constrained.

Strongly singer-specific and song-independent.

This is one of the most important identity blocks.

🔵 4. Voice Quality Measures
----------------------------------------------------
Includes:

Jitter

Shimmer

HNR

What they capture:

Roughness

Breathiness

Hoarseness

Harmonic strength

Why important:

Two singers may sing the same pitch.

But their glottal stability differs.

These are intrinsic vocal fold behaviors.

This captures voice texture.

🔵 5. Spectral Envelope Statistics
----------------------------------------------------
Includes:

Spectral centroid

Bandwidth

Rolloff

Flatness

Spectral contrast

Statistical moments (mean, std, skew, kurtosis)

What they capture:

Brightness vs darkness

Energy spread

Harmonic structure

Timbre shape

Why important:

Timbre defines identity.

Spectral tilt differs between singers.

These features represent global tonal color.

🔵 6. MFCC Features (Timbre Fingerprint)
----------------------------------------------------
13 MFCCs × statistical descriptors + delta terms.

What they capture:

Perceptual spectral shape

Fine timbral structure

Dynamic spectral evolution

Why important:

MFCCs are standard in speaker recognition.

They compactly represent vocal tract + spectral identity.

Delta terms capture dynamic articulation patterns.

This provides high-resolution timbre representation.

Why Statistical Aggregation?

We did not store second-by-second features.

We extracted frame-level features and computed:

Mean

Std

Percentiles

Skewness

Kurtosis

Range

Reason:

Singer identity = distributional property
Song expression = temporal sequence

We model identity, not melody.