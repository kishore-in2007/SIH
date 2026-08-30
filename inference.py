"""
ShieldVoice (SIH26104) - Real-time & File-based Deepfake Voice Inference Engine
Outputs spoof probability (0-100%), classification label, and latency profiling.
"""
import os
import time
import argparse
import numpy as np
import torch
import soundfile as sf
import librosa
from typing import Dict, Union

from models import Wav2Vec2AASIST, Wav2Vec2Classifier, AASIST
from dataset.dataset_loader import load_and_standardize_audio

class DeepfakeVoiceDetector:
    def __init__(self, checkpoint_path: str = None, model_type: str = "wav2vec2_aasist", ssl_model: str = "facebook/wav2vec2-xls-r-300m", device: str = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading weights from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            cfg = checkpoint.get("config", {})
            model_type = cfg.get("model_type", model_type)
            ssl_model = cfg.get("ssl_model", ssl_model)

            self.model = self._build_model(model_type, ssl_model)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            print(f"Initializing baseline model architecture: {model_type}")
            self.model = self._build_model(model_type, ssl_model)

        self.model = self.model.to(self.device)
        self.model.eval()

    def _build_model(self, model_type: str, ssl_model: str):
        if model_type == "wav2vec2_aasist":
            return Wav2Vec2AASIST(ssl_model_name=ssl_model, num_classes=2)
        elif model_type == "wav2vec2_linear":
            return Wav2Vec2Classifier(ssl_model_name=ssl_model, num_classes=2)
        elif model_type == "aasist":
            return AASIST(num_classes=2)
        else:
            return AASIST(num_classes=2)

    def analyze_audio(self, audio_input: Union[str, np.ndarray], sample_rate: int = 16000) -> Dict:
        """
        Analyzes audio file path or numpy waveform buffer and returns fraud risk assessment.
        """
        start_time = time.perf_counter()

        if isinstance(audio_input, str):
            waveform = load_and_standardize_audio(audio_input, target_sr=16000, is_train=False)
        elif isinstance(audio_input, np.ndarray):
            # Standardize buffer
            if audio_input.ndim > 1:
                audio_input = np.mean(audio_input, axis=1)
            if sample_rate != 16000:
                audio_input = librosa.resample(audio_input, orig_sr=sample_rate, target_sr=16000)
            
            # Pad or truncate to 64,600 samples
            target_len = 64600
            if len(audio_input) < target_len:
                repeat_factor = int(np.ceil(target_len / len(audio_input)))
                audio_input = np.tile(audio_input, repeat_factor)[:target_len]
            elif len(audio_input) > target_len:
                offset = (len(audio_input) - target_len) // 2
                audio_input = audio_input[offset:offset + target_len]
            waveform = audio_input.astype(np.float32)
        else:
            raise ValueError("audio_input must be a file path string or numpy array")

        tensor_wave = torch.from_numpy(waveform).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_wave)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        bonafide_prob = float(probabilities[0])
        spoof_prob = float(probabilities[1])
        risk_score_percent = spoof_prob * 100.0

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        label = "SPOOF / DEEPFAKE" if risk_score_percent >= 50.0 else "BONAFIDE / HUMAN"
        threat_level = "CRITICAL" if risk_score_percent > 85.0 else ("SUSPICIOUS" if risk_score_percent > 45.0 else "SAFE")

        return {
            "prediction": label,
            "threat_level": threat_level,
            "spoof_risk_percent": round(risk_score_percent, 2),
            "human_confidence_percent": round(bonafide_prob * 100.0, 2),
            "inference_latency_ms": round(latency_ms, 2),
            "device": str(self.device)
        }

def main():
    parser = argparse.ArgumentParser(description="ShieldVoice Voice Deepfake Detector CLI")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file (.wav, .flac, .mp3)")
    parser.add_argument("--checkpoint", type=str, default="saved_models/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    detector = DeepfakeVoiceDetector(checkpoint_path=args.checkpoint, device=args.device)
    result = detector.analyze_audio(args.audio)

    print("\n" + "="*45)
    print("      SHIELDVOICE FRAUD RISK ASSESSMENT      ")
    print("="*45)
    print(f" Input File:       {os.path.basename(args.audio)}")
    print(f" Result:           {result['prediction']}")
    print(f" Threat Level:     {result['threat_level']}")
    print(f" Deepfake Risk:    {result['spoof_risk_percent']}%")
    print(f" Human Confidence: {result['human_confidence_percent']}%")
    print(f" Inference Time:   {result['inference_latency_ms']} ms")
    print("="*45)

if __name__ == "__main__":
    main()
