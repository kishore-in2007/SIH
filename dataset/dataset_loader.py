"""
Unified Audio Anti-Spoofing Dataset Loader for ShieldVoice
Supports ASVspoof 2019, ASVspoof 2021, Deep Voice, MLAAD, and custom folder layouts.
Standardizes all samples to 16 kHz single-channel raw waveforms (64,600 samples ~ 4.04s).
"""
import os
import glob
import random
import numpy as np
import soundfile as sf
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict

TARGET_SAMPLE_RATE = 16000
TARGET_NUM_SAMPLES = 64600  # 4.0375 seconds at 16kHz (AASIST/ASVspoof standard)

def load_and_standardize_audio(file_path: str, target_sr: int = TARGET_SAMPLE_RATE, max_samples: int = TARGET_NUM_SAMPLES, is_train: bool = True) -> np.ndarray:
    """
    Loads an audio file (.flac, .wav, .mp3, .ogg), resamples to 16kHz, converts to mono,
    and crops or circular-pads to max_samples.
    """
    try:
        try:
            audio, sr = sf.read(file_path, dtype='float32')
        except Exception:
            audio, sr = librosa.load(file_path, sr=target_sr, mono=True)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        length = len(audio)
        if length < max_samples:
            repeat_factor = int(np.ceil(max_samples / max(length, 1)))
            audio = np.tile(audio, repeat_factor)[:max_samples]
        elif length > max_samples:
            if is_train:
                max_offset = length - max_samples
                offset = random.randint(0, max_offset)
                audio = audio[offset:offset + max_samples]
            else:
                offset = (length - max_samples) // 2
                audio = audio[offset:offset + max_samples]

        max_val = np.max(np.abs(audio))
        if max_val > 1e-6:
            audio = audio / max_val

        return audio.astype(np.float32)

    except Exception:
        return np.zeros(max_samples, dtype=np.float32)

class AudioSpoofDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], target_sr: int = TARGET_SAMPLE_RATE, num_samples: int = TARGET_NUM_SAMPLES, is_train: bool = True):
        self.samples = samples
        self.target_sr = target_sr
        self.num_samples = num_samples
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        waveform = load_and_standardize_audio(
            file_path=file_path,
            target_sr=self.target_sr,
            max_samples=self.num_samples,
            is_train=self.is_train
        )
        
        if self.is_train:
            if random.random() < 0.3:
                noise = np.random.normal(0, 0.005, waveform.shape).astype(np.float32)
                waveform = waveform + noise
            if random.random() < 0.3:
                scale = random.uniform(0.7, 1.2)
                waveform = waveform * scale

        tensor_waveform = torch.from_numpy(waveform)
        tensor_label = torch.tensor(label, dtype=torch.long)
        return tensor_waveform, tensor_label

def parse_asvspoof_protocols(root_dir: str) -> List[Tuple[str, int]]:
    """
    Finds all protocol text files and maps them directly to audio files in the tree.
    """
    samples = []
    protocol_files = glob.glob(os.path.join(root_dir, "**", "*.txt"), recursive=True)
    
    # Build fast filename lookup dictionary for all audio files in root_dir
    audio_map = {}
    for ext in ['*.flac', '*.wav', '*.mp3', '*.ogg']:
        for path in glob.glob(os.path.join(root_dir, "**", ext), recursive=True):
            base_key = os.path.splitext(os.path.basename(path))[0]
            audio_map[base_key] = path

    for pfile in protocol_files:
        try:
            with open(pfile, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        audio_id = parts[1]
                        key = parts[4].lower()
                        if key in ['bonafide', 'spoof']:
                            label = 0 if key == 'bonafide' else 1
                            if audio_id in audio_map:
                                samples.append((audio_map[audio_id], label))
        except Exception:
            continue

    return samples

def scan_directory_by_folder_names(root_dir: str) -> List[Tuple[str, int]]:
    samples = []
    supported_exts = ('.wav', '.flac', '.mp3', '.ogg', '.m4a')
    
    real_keywords = {'real', 'bonafide', 'genuine', 'human', 'original'}
    spoof_keywords = {'fake', 'spoof', 'clone', 'tts', 'vc', 'deepfake', 'synthetic', 'generated'}

    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(supported_exts):
                full_path = os.path.join(root, f)
                path_lower = full_path.lower()
                
                is_spoof = any(kw in path_lower for kw in spoof_keywords)
                is_real = any(kw in path_lower for kw in real_keywords)

                if is_spoof and not is_real:
                    samples.append((full_path, 1))
                elif is_real and not is_spoof:
                    samples.append((full_path, 0))
    return samples

def build_dataset_from_kagglehub(dataset_paths_dict: Dict[str, str], max_samples_per_class: int = 15000) -> List[Tuple[str, int]]:
    """
    Scans and indexes downloaded Kaggle datasets into a balanced list of samples.
    """
    all_samples = []
    
    print("--- Indexing Datasets ---")
    for name, path in dataset_paths_dict.items():
        if not path or not os.path.exists(path):
            continue
            
        print(f"Scanning dataset: {name} in {path}")
        ds_samples = parse_asvspoof_protocols(path)
        if not ds_samples:
            ds_samples = scan_directory_by_folder_names(path)

        all_samples.extend(ds_samples)
        print(f"  Indexed {len(ds_samples)} samples from {name}")

    unique_samples = list({s[0]: s for s in all_samples}.values())
    
    # Class balancing
    bonafide = [s for s in unique_samples if s[1] == 0]
    spoof = [s for s in unique_samples if s[1] == 1]
    
    random.seed(42)
    random.shuffle(bonafide)
    random.shuffle(spoof)

    # Balance classes so model learns robust decision boundaries
    if len(bonafide) > 0 and len(spoof) > 0:
        target_count = min(max(len(bonafide), len(spoof)), max_samples_per_class)
        # If one class is smaller, upsample or cap proportionally
        sampled_bonafide = bonafide[:target_count]
        sampled_spoof = spoof[:target_count]
        final_samples = sampled_bonafide + sampled_spoof
    else:
        final_samples = unique_samples[:max_samples_per_class * 2]

    random.shuffle(final_samples)
    real_count = sum(1 for s in final_samples if s[1] == 0)
    spoof_count = sum(1 for s in final_samples if s[1] == 1)
    print(f"\nFinal Balanced Training Set: {len(final_samples)} (Real/Bonafide: {real_count}, Spoof/Deepfake: {spoof_count})")
    
    return final_samples

def get_dataloaders(samples: List[Tuple[str, int]], batch_size: int = 16, val_split: float = 0.15, test_split: float = 0.15, num_workers: int = 2):
    n_total = len(samples)
    if n_total == 0:
        raise ValueError("No audio samples found to create DataLoader.")

    n_val = int(n_total * val_split)
    n_test = int(n_total * test_split)
    n_train = n_total - n_val - n_test

    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]

    print(f"Dataset split -> Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    train_ds = AudioSpoofDataset(train_samples, is_train=True)
    val_ds = AudioSpoofDataset(val_samples, is_train=False)
    test_ds = AudioSpoofDataset(test_samples, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
