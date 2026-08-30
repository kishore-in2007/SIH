"""
ShieldVoice (SIH26104) - Deepfake Voice Detection GPU Training Pipeline
Designed for Google Colab GPU (T4/V100/A100) and Kaggle environments with Google Drive persistence.
"""
import os
import sys
import time
import json
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_curve, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from config import AudioConfig, ModelConfig, TrainingConfig, DatasetConfig
from models import Wav2Vec2AASIST, Wav2Vec2Classifier, AASIST
from dataset import build_dataset_from_kagglehub, get_dataloaders

def compute_eer(y_true, y_scores):
    """
    Computes Equal Error Rate (EER) - the standard audio anti-spoofing evaluation metric.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    # Find the threshold where FPR and FNR are closest
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer * 100.0)

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp=True):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc="Training", leave=False)
    for waveforms, targets in pbar:
        waveforms = waveforms.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp and device.type == 'cuda':
            with autocast():
                logits = model(waveforms)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(waveforms)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = total_loss / len(loader)
    acc = accuracy_score(all_targets, all_preds) * 100.0
    return epoch_loss, acc

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_scores = []  # Probability of spoof (class 1)
    all_preds = []

    with torch.no_grad():
        for waveforms, targets in tqdm(loader, desc="Evaluating", leave=False):
            waveforms = waveforms.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(waveforms)
            loss = criterion(logits, targets)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)[:, 1]  # Spoof probability
            preds = torch.argmax(logits, dim=1)

            all_scores.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    val_loss = total_loss / len(loader)
    acc = accuracy_score(all_targets, all_preds) * 100.0
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary', zero_division=0)
    eer = compute_eer(all_targets, all_scores)

    metrics = {
        "val_loss": val_loss,
        "accuracy": acc,
        "precision": precision * 100.0,
        "recall": recall * 100.0,
        "f1": f1 * 100.0,
        "eer": eer
    }
    return metrics

def main():
    parser = argparse.ArgumentParser(description="ShieldVoice Colab GPU Deepfake Voice Training")
    parser.add_argument("--asvspoof19_path", type=str, default="", help="Path to ASVspoof 2019 dataset")
    parser.add_argument("--deepvoice_path", type=str, default="", help="Path to Deep Voice dataset")
    parser.add_argument("--asvspoof21_path", type=str, default="", help="Path to ASVspoof 2021 dataset")
    parser.add_argument("--mlaad_path", type=str, default="", help="Path to MLAAD dataset")
    parser.add_argument("--kagglehub_download", action="store_true", help="Automatically fetch datasets using kagglehub")
    
    parser.add_argument("--model_type", type=str, default="wav2vec2_aasist", choices=["wav2vec2_aasist", "wav2vec2_linear", "aasist"])
    parser.add_argument("--ssl_model", type=str, default="facebook/wav2vec2-xls-r-300m")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ssl_lr", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="./saved_models")
    parser.add_argument("--drive_save_dir", type=str, default="/content/drive/MyDrive/ShieldVoice_Models")
    
    args = parser.parse_args()

    # 1. Device Setup
    device = torch.device(args.device)
    print(f"=== ShieldVoice Training Pipeline ===")
    print(f"Hardware Device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # 2. Dataset Paths resolution
    dataset_paths = {}
    if args.kagglehub_download:
        print("\nDownloading datasets via kagglehub...")
        import kagglehub
        dataset_paths["asvspoof_2019"] = kagglehub.dataset_download('awsaf49/asvpoof-2019-dataset')
        dataset_paths["deep_voice"] = kagglehub.dataset_download('birdy654/deep-voice-deepfake-voice-recognition')
        dataset_paths["asvspoof_2021"] = kagglehub.dataset_download('mohammedabdeldayem/avsspoof-2021')
        dataset_paths["mlaad"] = kagglehub.dataset_download('trapka/mlaadthe-multi-languagaudioanti-spoofing-dataset')
    else:
        if args.asvspoof19_path: dataset_paths["asvspoof_2019"] = args.asvspoof19_path
        if args.deepvoice_path: dataset_paths["deep_voice"] = args.deepvoice_path
        if args.asvspoof21_path: dataset_paths["asvspoof_2021"] = args.asvspoof21_path
        if args.mlaad_path: dataset_paths["mlaad"] = args.mlaad_path

    # Fallback to local sample or synthetic data if no path passed
    if not dataset_paths:
        print("Warning: No dataset paths passed. Searching local directories...")
        dataset_paths["local_data"] = "./data"

    samples = build_dataset_from_kagglehub(dataset_paths)
    if len(samples) == 0:
        print("Error: No audio files indexed! Please check dataset paths.")
        sys.exit(1)

    train_loader, val_loader, test_loader = get_dataloaders(samples, batch_size=args.batch_size)

    # 3. Model Initialization
    print(f"\nBuilding Model Architecture: {args.model_type}...")
    if args.model_type == "wav2vec2_aasist":
        model = Wav2Vec2AASIST(ssl_model_name=args.ssl_model, num_classes=2)
    elif args.model_type == "wav2vec2_linear":
        model = Wav2Vec2Classifier(ssl_model_name=args.ssl_model, num_classes=2)
    elif args.model_type == "aasist":
        model = AASIST(num_classes=2)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    model = model.to(device)

    # 4. Optimizer & Loss Setup
    # Differentiate learning rates for pretrained SSL layers vs new head
    if hasattr(model, "ssl_model"):
        ssl_params = [p for p in model.ssl_model.parameters() if p.requires_grad]
        head_params = [p for n, p in model.named_parameters() if not n.startswith("ssl_model") and p.requires_grad]
        optimizer = optim.AdamW([
            {'params': ssl_params, 'lr': args.ssl_lr, 'weight_decay': 1e-4},
            {'params': head_params, 'lr': args.lr, 'weight_decay': 1e-4}
        ])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # 5. Directories Setup
    os.makedirs(args.save_dir, exist_ok=True)
    drive_available = os.path.exists(os.path.dirname(args.drive_save_dir))
    if drive_available:
        os.makedirs(args.drive_save_dir, exist_ok=True)
        print(f"Google Drive Checkpointing Active -> {args.drive_save_dir}")

    # 6. Training Loop
    best_eer = float('inf')
    best_acc = 0.0
    history = []

    print("\n" + "="*60)
    print(f"Starting Training for {args.epochs} Epochs")
    print("="*60)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start_time
        val_loss = val_metrics["val_loss"]
        val_acc = val_metrics["accuracy"]
        val_eer = val_metrics["eer"]

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, EER: {val_eer:.2f}%")

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            **val_metrics
        }
        history.append(epoch_record)

        # Checkpoint Saving
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "config": {
                "model_type": args.model_type,
                "ssl_model": args.ssl_model
            }
        }

        # Save latest
        latest_path = os.path.join(args.save_dir, "latest_checkpoint.pt")
        torch.save(checkpoint_data, latest_path)

        # Save best model (lowest EER)
        if val_eer < best_eer:
            best_eer = val_eer
            best_acc = val_acc
            best_path = os.path.join(args.save_dir, "best_model.pt")
            torch.save(checkpoint_data, best_path)
            print(f"  --> Saved new best model checkpoint! (EER: {best_eer:.2f}%, Acc: {best_acc:.2f}%)")

            # Persist to Google Drive
            if drive_available:
                try:
                    drive_best_path = os.path.join(args.drive_save_dir, "best_model.pt")
                    shutil.copyfile(best_path, drive_best_path)
                    print(f"  --> Synced best model to Google Drive: {drive_best_path}")
                except Exception as e:
                    print(f"  Warning: Drive sync failed: {e}")

    # 7. Final Test Evaluation
    print("\n" + "="*60)
    print("Running Final Evaluation on Test Set...")
    print("="*60)
    best_checkpoint = torch.load(os.path.join(args.save_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)

    print("\nFINAL TEST BENCHMARK RESULTS:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.2f}%")
    print(f"  Precision: {test_metrics['precision']:.2f}%")
    print(f"  Recall:    {test_metrics['recall']:.2f}%")
    print(f"  F1-Score:  {test_metrics['f1']:.2f}%")
    print(f"  EER:       {test_metrics['eer']:.2f}%")

    # Save training history JSON
    history_file = os.path.join(args.save_dir, "training_history.json")
    with open(history_file, "w") as f:
        json.dump({"history": history, "test_metrics": test_metrics}, f, indent=2)

    if drive_available:
        try:
            shutil.copyfile(history_file, os.path.join(args.drive_save_dir, "training_history.json"))
        except Exception:
            pass

    print(f"\nTraining Complete! All outputs and checkpoints saved in {args.save_dir}")

if __name__ == "__main__":
    main()
