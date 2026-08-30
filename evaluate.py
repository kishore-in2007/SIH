"""
ShieldVoice (SIH26104) - Benchmark Evaluation & Testing Script
Computes Equal Error Rate (EER), Precision, Recall, F1, ROC Curve, and Confusion Matrix.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_curve,
    auc,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from tqdm import tqdm

from models import Wav2Vec2AASIST, Wav2Vec2Classifier, AASIST
from dataset import build_dataset_from_kagglehub, AudioSpoofDataset
from torch.utils.data import DataLoader

def compute_eer_and_roc(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    optimal_threshold = thresholds[idx]
    roc_auc = auc(fpr, tpr)
    return eer * 100.0, optimal_threshold, roc_auc, fpr, tpr

def main():
    parser = argparse.ArgumentParser(description="ShieldVoice Model Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved model checkpoint (.pt)")
    parser.add_argument("--dataset_path", type=str, default="", help="Path to test dataset directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_json", type=str, default="evaluation_results.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    cfg = checkpoint.get("config", {})
    model_type = cfg.get("model_type", "wav2vec2_aasist")
    ssl_model = cfg.get("ssl_model", "facebook/wav2vec2-xls-r-300m")

    # Load Model Architecture
    print(f"Instantiating model: {model_type}...")
    if model_type == "wav2vec2_aasist":
        model = Wav2Vec2AASIST(ssl_model_name=ssl_model, num_classes=2)
    elif model_type == "wav2vec2_linear":
        model = Wav2Vec2Classifier(ssl_model_name=ssl_model, num_classes=2)
    elif model_type == "aasist":
        model = AASIST(num_classes=2)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load Test Samples
    paths_dict = {"eval_dataset": args.dataset_path} if args.dataset_path else {"local_eval": "./data"}
    samples = build_dataset_from_kagglehub(paths_dict)
    if not samples:
        print("Error: No test samples found.")
        sys.exit(1)

    test_ds = AudioSpoofDataset(samples, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    all_targets = []
    all_scores = []
    all_preds = []

    print(f"Running inference on {len(samples)} evaluation audio files...")
    with torch.no_grad():
        for waveforms, targets in tqdm(test_loader):
            waveforms = waveforms.to(device)
            logits = model(waveforms)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_scores.extend(probs.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    acc = accuracy_score(all_targets, all_preds) * 100.0
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary', zero_division=0)
    eer, opt_thresh, roc_auc, _, _ = compute_eer_and_roc(all_targets, all_scores)
    cm = confusion_matrix(all_targets, all_preds).tolist()

    results = {
        "checkpoint": args.checkpoint,
        "model_type": model_type,
        "total_samples": len(samples),
        "accuracy_percent": acc,
        "precision_percent": precision * 100.0,
        "recall_percent": recall * 100.0,
        "f1_score_percent": f1 * 100.0,
        "equal_error_rate_eer": eer,
        "optimal_threshold": float(opt_thresh),
        "roc_auc": float(roc_auc),
        "confusion_matrix_TN_FP_FN_TP": cm
    }

    print("\n" + "="*50)
    print("           BENCHMARK EVALUATION RESULTS           ")
    print("="*50)
    print(f" Total Evaluation Samples: {len(samples)}")
    print(f" Accuracy:                 {acc:.2f}%")
    print(f" Precision:                {precision*100.0:.2f}%")
    print(f" Recall:                   {recall*100.0:.2f}%")
    print(f" F1-Score:                 {f1*100.0:.2f}%")
    print(f" Equal Error Rate (EER):   {eer:.2f}%")
    print(f" ROC AUC Score:            {roc_auc:.4f}")
    print("="*50)

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output_json}")

if __name__ == "__main__":
    main()
