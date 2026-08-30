# ShieldVoice — SIH26104
## Build Spec v3 — Merged (working prototype + PROVE audit + dataset/pretrained-model sourcing)
Problem Statement: AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks (SIH26104, Software/Miscellaneous, AICTE)

---

## 0. HOW TO USE THIS DOCUMENT
This supersedes v2, the original PDF proposal, and the standalone PROVE audit. It keeps what's
already built and validated, adopts the PROVE audit's genuinely correct fixes, and adds the
dataset-sourcing and pretrained-model decisions made since v2. Feed this to Antigravity phase
by phase — do not start a phase until the previous one's exit criteria pass.

**What changed since v2:** Section 4 (datasets) now has verified download links (official +
Kaggle mirrors). Section 5 Phase A is updated to fine-tune via a pretrained SSL front-end
(Wav2Vec2-XLS-R) rather than treating AASIST-L as fixed/closed — this is the current
state-of-the-art approach in the literature, not a scope increase.

---

## 1. STATUS — WHAT'S ALREADY BUILT AND VALIDATED
Do not rebuild any of this from scratch. Extend it.

| Component | File | Status |
|---|---|---|
| Stage 0/1 DSP gate (silence/energy, spectral flatness, ZCR) | `detector.py` | Working |
| Stage 2 neural classifier — **real pretrained AASIST-L** (clovaai/aasist), not a placeholder | `detector.py` | Working — genuine human speech scores 0.3–10%, synthetic TTS scores 95–100% |
| Contextual Fraud Fusion (number-series mismatch, new-beneficiary+high-value, urgency language, first-time caller) | `context_fusion.py` | Working — additive-only boost, never launders a high audio score down |
| File-upload REST API (`POST /analyze`) | `main.py` | Working |
| Real-time WebSocket streaming (`/ws/stream`), 1s hop over 4.04s rolling window, 2-hop sustain debounce before auto-block | `streaming_server.py` | Working — ~180–200ms per-hop inference latency measured |
| Live ASR + automatic context extraction from transcript (zero manual flags) | `streaming_server.py` | Working — openai-whisper tiny.en, ~270–450ms per 3s batch; fixed two false-positive bugs (birth-year-as-amount, profession-mention-as-claimed-identity) |
| Browser demo with live risk gauge + "System Heard" transcript | `demo_stream.html` | Working |
| CLI live-call simulator | `stream_client_demo.py` | Working |

**Known open gaps (see Section 2/5 for what to do about each):**
- Genuine-speech testing has been TTS-in-disguise twice now (looped clip, then TTS labeled "conversational") — real mic-recorded natural conversation still not tested.
- Real archival human speech (Wikimedia Commons Spoken Wikipedia) showed risk spikes up to 98–99% on two of three clips, attributed to OGG/Vorbis compression + reverb artifacts — not yet root-caused.
- Indian-accent robustness — the #1 risk named in the original PDF — still empirically untested.
- Model detects TTS/VC-style synthesis artifacts; has not been tested against real-time streaming voice conversion (Item A below) — genuinely different attack signature.

---

## 2. CORRECTIONS FROM THE PROVE AUDIT — WHAT TO KEEP, DROP, OR FIX

| # | PROVE audit finding | Verdict | Action |
|---|---|---|---|
| A | Threat model conflates offline TTS/VC cloning with real-time streaming voice conversion (RVC-style); ASVspoof-trained classifiers generalize poorly to the latter | **Correct and important — the single best catch in that document** | Do NOT attempt to build streaming-VC detection from scratch (research-level problem, not hackathon-buildable). Document explicitly as a named limitation with a roadmap answer. Costs nothing, meaningfully raises credibility. |
| B | Baseline model should be "LCNN or MobileNetV3-Small... per the PDF's stated stack" | **Superseded — see Section 5 Phase A** | Not training a from-scratch CNN. Keeping AASIST-L as the backend, adding a pretrained SSL front-end (Wav2Vec2-XLS-R) — stronger than either the original PROVE suggestion or a frozen AASIST-L alone, and still fine-tuning, not training from zero. |
| C | Fine-tune on Indian-accent augmented data, re-measure EER | **Correct, and the actual fix for our #1 open gap** | Do this — see Section 5 Phase A. |
| D | Split into two evaluated tracks: offline TTS/VC vs. real-time streaming VC | Correct in spirit, wrong in scope | Document the distinction (per A) rather than building two separate eval tracks under deadline pressure. |
| E | Opt-in-only speaker enrollment, no silent voiceprinting | **Correct privacy fix, cheap to state, low cost to build** | Adopt as a stated design principle even if enrollment itself isn't built for the demo. |
| F | Single runtime: ONNX Runtime Mobile only, drop TFLite/CoreML sprawl | Correct scope discipline | Adopt for any future edge-deployment work; not yet relevant since current build is server-side. |
| G | Full Android `.aar` + iOS Swift Package SDKs, versioned, with reference apps | **Overscoped for hackathon time** | Defer past the finale. A working WebSocket + REST API already demonstrates the PS's "APIs and SDKs" requirement adequately for a hackathon judge. |
| H | Adversarial eval vs. RVC/so-vits-svc-generated samples | Valuable, not core-path | "If time permits" only — do not let this block demo-hardening. |
| I | Real dataset sourcing: ASVspoof 2019/2021/5, Kathbath, IndicVoices, MLAAD | **Correct and directly usable** | See Section 4 — verified links, including Kaggle mirrors, added. |
| J | Speaker-disjoint splits, no-fabricated-numbers discipline, decision log format | **Correct engineering discipline, cheap to adopt** | Adopt documentation habits in Section 6. |
| K | Latency claim "<20–30ms" in original PDF has no measurement plan | Already resolved | We have real measured numbers (180–200ms AASIST, 270–450ms ASR) — use these, not the original aspirational figure. |

---

## 3. REQUIREMENT TRACEABILITY

| PS Requirement | Status | Note |
|---|---|---|
| Real-time / near-real-time stream analysis | ✅ built & validated | WebSocket streaming, 1s hop, measured latency |
| Acoustic/spectral artifact detection | ✅ built & validated | Real AASIST-L, not placeholder |
| Dynamic risk score, not binary | ✅ built & validated | 0–100 fused score |
| Contextual/behavioral fraud signals | ✅ built & validated | Auto-extracted from live ASR transcript, zero manual input |
| Alerts + secondary verification workflow | ✅ built | Sustained-band debounce → action mapping; browser UI shows live badge |
| Privacy-preserving / edge-capable | ⚠️ partial | Currently server-side; opt-in-enrollment principle stated (item E), edge deployment is roadmap |
| Multilingual / Indian accent robustness | ⚠️ claimed, not yet evidenced | Real gap — Section 5 Phase A |
| Real-time streaming VC (not just TTS/VC) detection | ⚠️ named limitation, not solved | Documented per item A, not attempted live |
| APIs/SDKs for banking/telecom/enterprise integration | ✅ sufficiently covered for hackathon | REST + WebSocket API with OpenAPI docs; native SDKs deferred (item G) |

---

## 4. DATASETS & PRETRAINED MODELS — VERIFIED SOURCES

### 4.1 Accent fine-tuning data (Phase A — do this now)
- **AI4Bharat Kathbath** — 1,684 hrs labeled ASR speech, 12 Indian languages, CC BY-SA 4.0.
  `https://github.com/AI4Bharat/Kathbath` · mirror: `https://aikosh.indiaai.gov.in/home/datasets/details/kathbath.html`
- **AI4Bharat IndicVoices** — 23.7K hrs spontaneous/conversational speech, 22 languages, CC BY 4.0.
  `https://huggingface.co/datasets/ai4bharat/IndicVoices`
- **IndicVoices-R** — cleaned TTS-research variant, useful if synthesizing accented clones for augmentation.
  `https://aikosh.indiaai.gov.in/home/datasets/details/indicvoices_r.html`
- Check CC BY / CC BY-SA terms per subset before any public release of derived clone data — keep internal-only during the competition unless license + team confirm otherwise.

### 4.2 ASVspoof (official host, no registration needed — post-challenge open release)
- **2019 (LA+PA)**, DOI 10.7488/ds/2555, official: `https://datashare.ed.ac.uk/handle/10283/3336`
  Kaggle mirrors (verify against official file counts/checksums before use; still cite the DOI, not Kaggle, in the report):
  `https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset` (LA+PA, most-used)
  `https://www.kaggle.com/datasets/anishsarkar22/asvpoof-2019-dataset-la` (LA only)
- **2021 LA**: `https://zenodo.org/record/4837263` · keys: `https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz`
- **2021 PA**: `https://zenodo.org/record/4834716` · keys: `https://www.asvspoof.org/asvspoof2021/PA-keys-full.tar.gz`
- **2021 DF**: `https://zenodo.org/record/4835108` · keys: `https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz`
- Open Data Commons Attribution License (ODC-BY) — attribution required, no registration required (post-challenge release).
- Official EER/min-tDCF scoring scripts: `https://github.com/asvspoof-challenge/2021`
- **Note:** ASVspoof 2019 train/dev is the only sanctioned training partition; 2021 LA/DF/PA are eval-only (no separate 2021 training set) — matches the fine-tuning plan in Section 5.

### 4.3 Pretrained models (fine-tune, do not train from scratch)
- **AASIST-L** (already in use) — `https://github.com/clovaai/aasist`
- **Wav2Vec2-XLS-R** (SSL front-end, pretrained on 128 languages incl. Indian languages — directly helps accent robustness) — `https://huggingface.co/facebook/wav2vec2-xls-r-300m`
- **Reference architecture combining both** (Wav2Vec2-XLS-R + AASIST back-end) — most-cited current approach in the field; freeze most SSL layers, fine-tune last 2-3 + AASIST head. Reference implementation: `https://github.com/TakHemlata/SSL_Anti-spoofing`
- **Directly usable pretrained checkpoint** (fine-tuned on ASVspoof 2019 LA, EER 0.63%) to start from instead of zero: `https://github.com/xieyuankun/ADD-W2V2-LCNN-19LA0.6`

### 4.4 Deferred (post-hackathon research track, not this build)
- MLAAD, In-the-Wild, WaveFake, FakeAVCeleb — cross-dataset generalization testing
- MUSAN/RIRS_NOISES — noise/reverb augmentation, useful for root-causing the archival-audio false-positive spike (Section 1) if time allows
  `https://www.openslr.org/17/` (MUSAN) · `https://www.openslr.org/28/` (RIRS_NOISES)

---

## 5. EXECUTION PLAN — hackathon-realistic phases

### Phase A — Close the accent-validation gap (highest priority open item)
**Goal:** A real, measured answer on Indian-accent robustness — the single most-repeated open risk across every round of testing so far.
1. Get real mic-recorded natural conversation (not TTS, not scripted, not looped) from multiple speakers/accents — still not done despite being asked for twice. This is the one thing that cannot be substituted with more synthetic data.
2. Pull bonafide clips from Kathbath/IndicVoices (Section 4.1) for additional accent diversity.
3. Add a Wav2Vec2-XLS-R SSL front-end ahead of AASIST-L (Section 4.3); fine-tune only the last 2-3 transformer layers + AASIST head on Indian-accent bonafide + existing spoof data — cheaper and more effective than retraining AASIST-L alone, and realistic on one Colab/Kaggle GPU session.
4. At minimum, run zero-shot evaluation with the current model and report real numbers even if fine-tuning doesn't finish in time — do not claim robustness without a measured result.
5. Root-cause the archival-audio false-positive spike (98–99% risk on real human Wikipedia clips) — isolate whether it's OGG/Vorbis compression, reverb, or narration style before assuming it's a codec-robustness gap. Use MUSAN/RIRS_NOISES (Section 4.4) if isolating compression vs. reverb needs controlled augmentation.
**Exit criteria:** a real results table — genuine accented speech samples, measured risk scores, honestly reported (including any failures or if fine-tuning didn't complete).

### Phase B — Demo hardening
**Goal:** The live demo works twice in a row without manual intervention.
1. Stress-test the browser demo (`demo_stream.html`) end-to-end on the actual presentation hardware.
2. Prepare an offline/network-failure fallback per the original "offline-resilient" claim.
3. Rehearse both scenarios: fraud call → auto-block, benign call → stays green — using real (not TTS-labeled-as-genuine) audio for the benign case.
**Exit criteria:** two consecutive clean live runs.

### Phase C — Documentation discipline
1. `docs/eval_report.md` — every number that appears in the pitch deck must be in this file first. No exceptions.
2. `docs/decision_log.md` — Decision / Reason / Alternative rejected / Trade-off, one line each. Use item A (streaming-VC limitation) as the first entry.
3. Limitations section in the submission doc: name the TTS/VC-vs-streaming-VC gap (item A) and the accent-testing status (Phase A result) honestly.
4. `docs/dataset_card.md` — record which datasets/checkpoints were used (Section 4), license terms, and internal-vs-public status.
**Exit criteria:** all files exist and are internally consistent with the demo's actual numbers.

### Phase D — Deferred / roadmap-only (name, do not build)
- Native Android `.aar` / iOS Swift Package SDKs
- Full ASVspoof 2019/2021/5 cross-dataset EER benchmarking beyond what Phase A needs
- Adversarial robustness eval against RVC/so-vits-svc
- Opt-in speaker enrollment implementation (state the principle, don't build the flow)
- Edge/on-device quantized deployment

---

## 6. OPEN QUESTIONS — CANNOT BE AUTO-RESOLVED
1. Who can provide real mic-recorded conversational samples, and from how many different speakers/accents, before the deadline?
2. Is GPU compute available (Colab/Kaggle, 400GB Drive confirmed available) for the Phase A fine-tuning run, and how much time can be spent on it without threatening demo-hardening time?
3. Confirm: is any dataset derived from Kathbath/IndicVoices staying internal-only, or is public release intended? (Changes what's permissible under their license terms.)

---

## 7. DECISION LOG

| Decision | Reason | Alternative rejected | Trade-off |
|---|---|---|---|
| Keep AASIST-L as the back-end; add a pretrained Wav2Vec2-XLS-R SSL front-end and fine-tune | Already validated and working; SSL front-end is current state-of-the-art for accent/language robustness, directly addresses the #1 open gap | Train LCNN/MobileNetV3-Small from scratch (original PROVE suggestion); keep AASIST-L frozen with no accent adaptation | Slightly more fine-tuning complexity than a frozen model, but directly targets the accent-robustness gap with proven literature support |
| Document the TTS/VC vs. streaming-VC detection gap rather than attempting to solve it | Genuinely research-level problem; not buildable in hackathon time | Attempt frame-to-frame phase-discontinuity detection live | Smaller technical claim, but an honest and defensible one |
| Defer native SDK packaging past the finale | REST/WebSocket API already satisfies the PS's API/SDK requirement for demo purposes | Build full Android `.aar` + iOS SPM packages now | Less polished integration story, but demo-hardening time preserved |
| Prioritize real accent-testing data over more synthetic/TTS test variety | Accent robustness is the most-repeated unresolved risk across every prior test round | Generate more TTS "conversational" samples (already tried twice, doesn't answer the real question) | Slower to get results (needs real recordings), but the only test that actually answers the question |
| Use ASVspoof 2019 official/Kaggle-mirror + 2021 official Zenodo links, no registration needed | Post-challenge open release confirmed (ODC-BY license); Kaggle mirrors accelerate access but must be checksum-verified against official source | Wait on ASVspoof registration form (unnecessary — access is already open) | None significant; mirrors carry a small verification step before trusted use |
