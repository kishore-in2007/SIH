"""
ShieldVoice (SIH26104) - Configuration & Hyperparameters
"""
import os
from dataclasses import dataclass

@dataclass
class AudioConfig:
    sample_rate: int = 16000          # Standard 16 kHz sampling rate
    num_samples: int = 64600          # ~4.0375 seconds (matching AASIST-L / ASVspoof standard)
    n_fft: int = 512
    hop_length: int = 160
    n_mels: int = 80
    normalize_audio: bool = True

@dataclass
class ModelConfig:
    # Options: 'wav2vec2_aasist', 'wav2vec2_linear', 'aasist'
    model_type: str = "wav2vec2_aasist"
    # Pretrained SSL model (Wav2Vec2-XLS-R 300M or Base)
    ssl_model_name: str = "facebook/wav2vec2-xls-r-300m"
    fallback_ssl_model_name: str = "facebook/wav2vec2-base"
    freeze_ssl_layers: int = 18       # Freeze bottom transformer layers (fine-tune top 6 layers + head)
    num_classes: int = 2              # 0: Bonafide (Real), 1: Spoof (Deepfake)
    dropout: float = 0.2
    hidden_dim: int = 256

@dataclass
class TrainingConfig:
    batch_size: int = 16              # Batch size per GPU step
    learning_rate: float = 1e-4       # Learning rate for classifier head
    ssl_learning_rate: float = 1e-5   # Lower learning rate for fine-tuning SSL layers
    weight_decay: float = 1e-4
    epochs: int = 25
    warmup_epochs: int = 2
    gradient_accumulation_steps: int = 2
    use_amp: bool = True              # Automatic Mixed Precision for 2x faster GPU training
    num_workers: int = 2
    seed: int = 42
    save_dir: str = "./saved_models"
    drive_save_dir: str = "/content/drive/MyDrive/ShieldVoice_Models"

@dataclass
class DatasetConfig:
    # Supported Kaggle datasets
    asvspoof_2019_name: str = "awsaf49/asvpoof-2019-dataset"
    deep_voice_name: str = "birdy654/deep-voice-deepfake-voice-recognition"
    asvspoof_2021_name: str = "mohammedabdeldayem/avsspoof-2021"
    mlaad_name: str = "trapka/mlaadthe-multi-languagaudioanti-spoofing-dataset"
    
    val_split: float = 0.15
    test_split: float = 0.15
