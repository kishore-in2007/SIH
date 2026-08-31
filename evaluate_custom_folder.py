"""
ShieldVoice (SIH26104) - Custom Folder Audio Evaluation Suite
Evaluates all audio samples in a given folder using the fine-tuned best_model.pt.
"""
import os
import sys
import glob
import time
import json
import torch
import numpy as np

from inference import DeepfakeVoiceDetector

def evaluate_folder(folder_path="New folder", checkpoint_path="./best_model.pt"):
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    print(f"--- Evaluating Audio Files in: {folder_path} ---")
    detector = DeepfakeVoiceDetector(checkpoint_path=checkpoint_path)

    # Find all audio files
    audio_extensions = ["*.wav", "*.flac", "*.mp3", "*.ogg", "*.m4a"]
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(folder_path, ext)))
        audio_files.extend(glob.glob(os.path.join(folder_path, "**", ext), recursive=True))
    
    audio_files = sorted(list(set(audio_files)))
    if not audio_files:
        print("No audio files found in directory.")
        return

    results = []
    print("\n" + "="*95)
    print(f"{'Audio File Name':<42} | {'Verdict':<18} | {'Risk %':<9} | {'Human Conf %':<12} | {'Threat Level':<10}")
    print("="*95)

    for audio_path in audio_files:
        filename = os.path.basename(audio_path)
        try:
            res = detector.analyze_audio(audio_path)
            verdict = res["prediction"]
            risk = res["spoof_risk_percent"]
            human_conf = res["human_confidence_percent"]
            threat = res["threat_level"]
            latency = res["inference_latency_ms"]

            print(f"{filename:<42} | {verdict:<18} | {risk:>6.2f}%   | {human_conf:>9.2f}%    | {threat:<10}")
            
            results.append({
                "file_name": filename,
                "file_path": audio_path,
                "verdict": verdict,
                "deepfake_risk_percent": risk,
                "human_confidence_percent": human_conf,
                "threat_level": threat,
                "latency_ms": latency
            })
        except Exception as e:
            print(f"{filename:<42} | ERROR: {e}")

    print("="*95)
    
    # Save output
    output_path = "custom_folder_evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed evaluation results saved to: {output_path}")
    return results

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "New folder"
    evaluate_folder(folder)
