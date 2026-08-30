# ShieldVoice (SIH26104) — AI-Powered Voice Cloning & Impersonation Attack Detector

ShieldVoice is a real-time deepfake voice detection and anti-spoofing system built for **Smart India Hackathon (SIH26104)**. It combines self-supervised multilingual acoustic representations (**Wav2Vec2-XLS-R 300M**) with **AASIST (Graph Attention Network)** back-ends to provide robust detection across diverse accents (including Indian regional accents) and audio compression formats.

---

## Key Features

1. **State-of-the-Art Architecture**:
   - **Front-End**: `facebook/wav2vec2-xls-r-300m` (pretrained on 128 languages for cross-lingual & accent robustness).
   - **Back-End**: Graph Attention Network (GAT) capturing spectro-temporal synthesis artifacts.
2. **Kaggle & ASVspoof Dataset Integration**:
   - `awsaf49/asvpoof-2019-dataset`
   - `birdy654/deep-voice-deepfake-voice-recognition`
   - `mohammedabdeldayem/avsspoof-2021`
   - `trapka/mlaadthe-multi-languagaudioanti-spoofing-dataset`
3. **Google Colab GPU Accelerated**:
   - Automatic Mixed Precision (`torch.cuda.amp`) for 2x faster training on T4, V100, and A100 GPUs.
   - Built-in `colab-ssh` bridge for direct PowerShell/Terminal CLI access.
4. **Persistent Google Drive Checkpointing**:
   - Automatically mounts and saves `best_model.pt` and `training_history.json` to `/content/drive/MyDrive/ShieldVoice_Models/`.
5. **Real-Time Inference Engine**:
   - Fast sub-200ms latency inference yielding a 0–100% deepfake risk score and threat level (`SAFE`, `SUSPICIOUS`, `CRITICAL`).

---

## Project Structure

```text
├── colab_training_pipeline.ipynb   # 1-Click Google Colab GPU training notebook
├── train.py                        # Multi-dataset GPU training script with mixed-precision (FP16) & EER calculation
├── evaluate.py                     # Benchmark evaluation & cross-dataset EER testing
├── inference.py                    # Real-time / file-based audio deepfake inference engine
├── config.py                       # Hyperparameters, sample rate (16kHz), window size, audio augmentation
├── models/
│   ├── __init__.py
│   ├── wav2vec2_aasist.py          # Pretrained Wav2Vec2-XLS-R + AASIST Graph Attention architecture
│   └── aasist.py                   # SincNet + Residual GAT standalone anti-spoofing model
├── dataset/
│   ├── __init__.py
│   └── dataset_loader.py           # Unified dataset parser for ASVspoof 2019/2021, Deep Voice, MLAAD
├── requirements.txt                # Python dependencies
└── README.md                       # Documentation & instructions
```

---

## Quickstart: Training on Google Colab (GPU)

### Step 1: Open Notebook in Google Colab
1. Upload [`colab_training_pipeline.ipynb`](file:///C:/Users/sakth/Downloads/SIH/colab_training_pipeline.ipynb) to Google Colab.
2. Select GPU Runtime: **Runtime** → **Change runtime type** → select **T4 GPU** (or A100).

### Step 2: Run Dataset Download & Training
Run the notebook cells in sequence. The notebook will:
- Mount your Google Drive at `/content/drive/MyDrive/ShieldVoice_Models/`.
- Download all 4 Kaggle datasets automatically using `kagglehub`.
- Train the model using mixed precision on the Colab GPU.
- Compute validation Equal Error Rate (EER) and save the best weights to Google Drive.

---

## Running from Local CLI or Remote SSH

### 1. Environment Setup
```bash
git clone https://github.com/kishore-in2007/SIH.git
cd SIH
pip install -r requirements.txt
```

### 2. Start GPU Training
```bash
python train.py \
    --kagglehub_download \
    --model_type wav2vec2_aasist \
    --ssl_model facebook/wav2vec2-xls-r-300m \
    --epochs 25 \
    --batch_size 16 \
    --lr 1e-4 \
    --ssl_lr 1e-5 \
    --device cuda \
    --save_dir ./saved_models \
    --drive_save_dir /content/drive/MyDrive/ShieldVoice_Models
```

### 3. Evaluate Model on Benchmark Dataset
```bash
python evaluate.py \
    --checkpoint ./saved_models/best_model.pt \
    --dataset_path /path/to/eval_dataset \
    --device cuda
```

### 4. Run Inference on Audio File
```bash
python inference.py \
    --audio sample_recording.wav \
    --checkpoint ./saved_models/best_model.pt
```

**Sample Output:**
```text
=============================================
      SHIELDVOICE FRAUD RISK ASSESSMENT      
=============================================
 Input File:       sample_recording.wav
 Result:           SPOOF / DEEPFAKE
 Threat Level:     CRITICAL
 Deepfake Risk:    96.42%
 Human Confidence: 3.58%
 Inference Time:   142.6 ms
=============================================
```

---

## Evaluation Metrics

The pipeline calculates:
- **Equal Error Rate (EER)**: Threshold where False Acceptance Rate (FAR) equals False Rejection Rate (FRR) — the standard metric for ASVspoof challenges.
- **Accuracy, Precision, Recall, F1-Score**
- **ROC-AUC (Area Under Receiver Operating Characteristic Curve)**
- **Confusion Matrix**
