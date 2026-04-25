# Novel SVC Architecture Diagram

Below is the Mermaid code for the architecture diagram. You can paste this code into [Mermaid Live Editor](https://mermaid.live/) to export it as an image, or view it directly in VS Code if you have a Mermaid extension installed.

```mermaid
graph TD
    %% Styling
    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef prep fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef enc fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;
    classDef diff fill:#fCE4EC,stroke:#880E4F,stroke-width:3px;
    classDef post fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;
    classDef output fill:#e1f5fe,stroke:#01579b,stroke-width:2px;

    %% Step 1: Inputs
    Source[Source Mixed Song]:::input
    Target[Target Reference Singer]:::input

    %% Step 2: Preprocessing
    subgraph Preprocessing["Multi-Band Audio Isolator"]
        Source --> Demucs[Source Separation U-Net]:::prep
        Demucs --> Instrumental[Background Instrumental]:::prep
        Demucs --> PureVocal[Pure Source Vocal]:::prep
    end

    %% Step 3: Feature Extraction
    subgraph Feature_Extraction["Disentangled Representation Extractors"]
        PureVocal --> SSL[Self-Supervised Linguistic Encoder]:::enc
        PureVocal --> F0[Prosodic F0 Contour Extractor]:::enc
        
        Target --> TimbreEnc[Zero-Shot Timbre Encoder]:::enc
    end

    %% Step 4: The Novel Backbone
    subgraph Diffusion_Backbone["Novel Diffusion Transformer (TC-DiT) Backbone"]
        Noise[Latent Gaussian Noise N0,I]:::diff
        SSL --> DiT[Timbre-Conditioned Diffusion Transformer]:::diff
        F0 --> DiT
        TimbreEnc --> DiT
        Noise --> DiT
        DiT -->|Iterative Denoising| LatentMel[Denoised Latent Spectrogram]:::diff
    end

    %% Step 5: Vocoding
    subgraph Vocoder_Synthesis["Neural Vocoder"]
        LatentMel --> Vocoder[HiFi-GAN / WaveFormer Vocoder]:::prep
        Vocoder --> RawSynth[Raw Synthesized Vocal 75% Clarity]:::prep
    end

    %% Step 6: Post Processing Identity Alignment
    subgraph Post_Processing["N x N Identity Post-Processing Pipeline"]
        RawSynth --> FeatureExtractor[Acoustic Feature Extractor<br/>317 Features]:::post
        Target --> FeatureExtractor
        FeatureExtractor --> Matrix[N x N Similarity Matrix Evaluation]:::post
        Matrix -->|Residual Artifact Correction| Refine[Timbre Latent Alignment Filter]:::post
        Refine --> Crisp[100% Crisp Mapped Vocal]:::post
    end

    %% Step 7: Final Output
    Crisp --> Mixer[Audio Signal Mixer]:::output
    Instrumental --> Mixer
    Mixer --> FinalOutput[Final Converted Song]:::output
```
