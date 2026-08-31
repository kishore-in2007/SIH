# 🎙️ ShieldVoice (SIH26104) — Hackathon Presentation Slide
## Slide Title: Channel Robustness: Open Limitation & Engineering Roadmap

---

### 1. The Known Limitation & Empirical Evidence
* **Observation**: Current state-of-the-art anti-spoofing models (Wav2Vec2, AASIST, RawNet2) achieve **<4% EER** on standard benchmarks (ASVspoof 2019/2021) but exhibit elevated false-positive rates on live laptop microphones and raw telephony channels.
* **Empirical Evidence**:
  * Uncompressed Studio Dataset Speech $\rightarrow$ **Logit Margin $\approx 1.0$** (Pass)
  * Neural Clones & Telephony Scams $\rightarrow$ **Logit Margin $\approx 12.0 - 13.8$** (Detected)
  * Live Low-SNR Laptop Microphones $\rightarrow$ **Logit Margin $\approx 6.5 - 7.5$** (Domain Shift)

---

### 2. Root Cause Analysis
* **Acoustic Confounding in Research Benchmarks**:
  * In standard datasets, $100\%$ of bonafide human recordings originate from high-end studio condenser microphones in anechoic chambers (e.g., VCTK).
  * Consequently, the neural network conflates **acoustic degradation** (room reverberation, thermal noise, mic clipping) with **synthesis artifacts**.

---

### 3. Production Roadmap: 5-Point Resolution Strategy

```
[Clean Dataset] ───► [Dual-Class Channel Augmentation] ───► [Layer Unfreezing] ───► [Uncertainty Abstention]
 (ASVspoof / MLAAD)     (MUSAN Noise + RIR + G.711/Opus)      (Top 4 Transformer Layers)    (Adaptive Verification)
```

1. **Dual-Class Channel Augmentation Pipeline**:
   * Augment **both bonafide and spoof audio identically** using:
     * **Real Room Impulse Responses (RIR)**: OpenSLR RIR & BUT ReverbDB.
     * **Additive Environmental Noise**: MUSAN & DEMAND at varied SNR (5dB to 25dB).
     * **Lossy Codec Round-trips**: G.711 $\mu$-law/a-law, AMR-NB, and Opus (8–24 kbps) via FFmpeg.
   * *Critical Principle*: Augmenting both classes forces the network to learn that **channel degradation $\neq$ spoof**.
2. **Layer-Unfrozen Fine-Tuning**:
   * Unfreeze the top 4–6 transformer encoder layers alongside the Graph Attention head on speaker-disjoint splits.
3. **Dedicated Target-Channel Validation Gate**:
   * Curate an untouched 100-sample target-channel validation set (real laptop mics, smartphones, landlines) as a strict out-of-distribution evaluation gate.
4. **Temperature-Scaled Calibration & Safe Abstention**:
   * Implement temperature scaling and embedding manifold distance estimation.
   * Ambiguous or out-of-distribution inputs route to **"Adaptive Secondary Verification"** rather than hard rejection.
5. **Statistical Rigor Harness**:
   * Benchmark with Bootstrap Confidence Intervals (95% CI) across cross-dataset splits.

---

### 4. Operational Demonstration Posture
* **Validated Cases 1–3**: Benchmark replay pipeline for high-precision spoof detection.
* **Live Ingestion Chamber**: Captures liveness, acoustic energy continuity, and contextual fraud signals with safe uncertainty abstention.
