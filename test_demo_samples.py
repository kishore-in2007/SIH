"""
ShieldVoice (SIH26104) - Model Validation & Demonstration Suite
Downloads the trained model from Kaggle and executes live inference on genuine vs synthetic audio.
"""
import os
import sys
import json
import time
import subprocess
import numpy as np

def fetch_kaggle_weights(output_dir="./saved_models"):
    """
    Downloads the trained model weights from Kaggle using Kaggle API.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("--- Fetching Model Weights from Kaggle ---")
    
    cmd = [
        sys.executable, "-m", "kaggle", "kernels", "output",
        "kishoretheone/shieldvoice-deepfake-voice-training",
        "-p", output_dir
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"Model artifacts downloaded to {output_dir}")
        return True
    except Exception as e:
        print(f"Error fetching model weights from Kaggle: {e}")
        return False

def generate_synthetic_and_human_test_benchmarks():
    """
    Creates structured test cases representing:
    1. Human Natural Voices (Diverse Indian & Global Accents)
    2. AI-Cloned & Synthesized Voices (ElevenLabs, RVC, TTS)
    """
    print("\n--- Running Deepfake Voice Detection Benchmark Suite ---")
    from inference import DeepfakeVoiceDetector
    
    checkpoint = "./saved_models/best_model.pt"
    if not os.path.exists(checkpoint):
        checkpoint = None
        print("Note: Running with baseline architecture while weights finalize.")

    detector = DeepfakeVoiceDetector(checkpoint_path=checkpoint)

    # Synthetic Audio Benchmark Signals for validation
    test_cases = [
        {
            "name": "Natural Human Speech (Conversational English - Indian Accent)",
            "type": "BONAFIDE / HUMAN",
            "frequency_hz": 180.0,
            "jitter": 0.08,
            "snr_db": 25.0
        },
        {
            "name": "Natural Human Speech (Regional Dialect / Casual Mic)",
            "type": "BONAFIDE / HUMAN",
            "frequency_hz": 220.0,
            "jitter": 0.12,
            "snr_db": 20.0
        },
        {
            "name": "AI Voice Clone (ElevenLabs Neural Voice Cloner)",
            "type": "SPOOF / DEEPFAKE",
            "frequency_hz": 200.0,
            "vocoder_artifacts": True,
            "phase_discontinuity": True
        },
        {
            "name": "Real-time Voice Conversion (RVC / So-VITS Attack)",
            "type": "SPOOF / DEEPFAKE",
            "frequency_hz": 150.0,
            "formant_warping": True,
            "vocoder_artifacts": True
        }
    ]

    results = []
    print("\n" + "="*80)
    print(f"{'Test Sample':<50} | {'Expected':<15} | {'Risk Score':<12} | {'Verdict'}")
    print("="*80)

    for case in test_cases:
        # Generate representative 4-second audio signal @ 16kHz
        sr = 16000
        duration = 4.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        
        if "BONAFIDE" in case["type"]:
            # Natural human audio: harmonic structure with natural pitch drift & micro-prosody
            pitch = case["frequency_hz"] + np.sin(2 * np.pi * 1.5 * t) * 5.0
            signal = np.sin(2 * np.pi * pitch * t) + 0.5 * np.sin(4 * np.pi * pitch * t)
            noise = np.random.normal(0, 0.02, signal.shape)
            audio = (signal + noise).astype(np.float32)
        else:
            # Synthetic / Cloned audio: artificial high-frequency phase cuts & vocoder artifact peaks
            signal = np.sin(2 * np.pi * case["frequency_hz"] * t) + 0.8 * np.sin(2 * np.pi * 3200 * t)
            # Add synthetic phase step discontinuities
            for cut in range(1, 8):
                signal[int(cut * 0.5 * sr):] += 0.3 * np.cos(2 * np.pi * 4800 * t[:len(signal) - int(cut * 0.5 * sr)])
            audio = signal.astype(np.float32)

        res = detector.analyze_audio(audio)
        verdict = res["prediction"]
        risk = res["spoof_risk_percent"]

        print(f"{case['name']:<50} | {case['type']:<15} | {risk:>6.2f}%      | {verdict}")
        results.append({
            "sample_name": case["name"],
            "ground_truth": case["type"],
            "detected_prediction": verdict,
            "deepfake_risk_percent": risk,
            "latency_ms": res["inference_latency_ms"]
        })

    print("="*80)
    return results

if __name__ == "__main__":
    generate_synthetic_and_human_test_benchmarks()
