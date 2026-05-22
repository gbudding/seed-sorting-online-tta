"""
train.py
--------
Trains a ResNet-50 (ImageNet pretrained) on the GrainSet binary
classification task: good (NOR) vs. bad (all DU categories + impurities).

Training protocol matches Fan et al. (2023):
  - ResNet-50 backbone, last FC replaced with 2-class head
  - SGD, lr=0.0012, weight_decay=1e-4, momentum=0.9
  - Step-LR schedule: decay by 0.1 at epochs 20 and 40
  - 50 epochs, batch size 128
  - Cross-entropy loss with inverse-frequency class weights
  - ImageNet mean/std normalisation

Checkpoints saved:
  <out>/best.pth       — weights with best validation F1
  <out>/last.pth       — weights after final epoch
  <out>/train_log.csv  — per-epoch metrics

Usage
-----
    python src/models/train.py \
        --datalist  runs/datalist/wheat \
        --out       runs/checkpoints/wheat_resnet50 \
        --epochs    50 \
        --batch-size 128 \
        --device    cuda
"""

import argparse
import csv
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data.dataset import (
    GrainSetDataset,
    train_transforms,
    eval_transforms,
    get_class_weights,
)


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    ResNet-50 with ImageNet weights; final FC replaced for binary task.

    We unfreeze all layers from the start — the dataset is large enough
    that full fine-tuning outperforms feature extraction.
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model   = models.resnet50(weights=weights)
    # Replace classifier head
    in_features     = model.fc.in_features
    model.fc        = nn.Linear(in_features, num_classes)
    return model


# ------------------------------------------------------------------
# Training helpers
# ------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    """Run one full epoch.  Returns (loss, accuracy, f1)."""
    model.train() if train else model.eval()

    total_loss = 0.0
    correct    = 0
    total      = 0
    tp = fp = fn = 0  # for F1 (positive = good, label 1)

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            preds = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            # F1 components (binary, positive = class 1)
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return avg_loss, accuracy, f1


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def train(args):
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == "cpu" else "cpu")
    print(f"Device: {device}")

    # ---- Datasets & loaders --------------------------------------
    train_ds = GrainSetDataset(
        os.path.join(args.datalist, "train.txt"),
        transform=train_transforms())
    val_ds   = GrainSetDataset(
        os.path.join(args.datalist, "val.txt"),
        transform=eval_transforms())

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    print(f"Train: {len(train_ds):,} samples | "
          f"Val: {len(val_ds):,} samples")

    # ---- Model ---------------------------------------------------
    model = build_model(num_classes=2, pretrained=True).to(device)
    print(f"Model: ResNet-50  "
          f"({sum(p.numel() for p in model.parameters()):,} params)")

    # ---- Loss with class weights ---------------------------------
    weights   = get_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    print(f"Loss weights: bad={weights[0]:.3f}  good={weights[1]:.3f}")

    # ---- Optimiser & scheduler -----------------------------------
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr, momentum=0.9,
        weight_decay=args.weight_decay)

    # Decay LR by 0.1 at epochs 20 and 40 (matches Fan et al.)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[20, 40], gamma=0.1)

    # ---- Training loop -------------------------------------------
    log_path  = os.path.join(args.out, "train_log.csv")
    best_f1   = -1.0
    best_epoch = 0

    with open(log_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "lr",
                         "train_loss", "train_acc", "train_f1",
                         "val_loss",   "val_acc",   "val_f1"])

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            tr_loss, tr_acc, tr_f1 = run_epoch(
                model, train_loader, criterion, optimizer, device, train=True)
            va_loss, va_acc, va_f1 = run_epoch(
                model, val_loader, criterion, optimizer, device, train=False)

            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0

            print(f"Epoch {epoch:3d}/{args.epochs}  "
                  f"lr={lr:.6f}  "
                  f"train_loss={tr_loss:.4f}  train_f1={tr_f1:.4f}  "
                  f"val_loss={va_loss:.4f}  val_f1={va_f1:.4f}  "
                  f"({elapsed:.0f}s)")

            writer.writerow([epoch, lr,
                             tr_loss, tr_acc, tr_f1,
                             va_loss, va_acc, va_f1])
            csvfile.flush()

            # Save best model (by val F1)
            if va_f1 > best_f1:
                best_f1    = va_f1
                best_epoch = epoch
                torch.save({
                    "epoch"     : epoch,
                    "state_dict": model.state_dict(),
                    "val_f1"    : va_f1,
                    "val_acc"   : va_acc,
                    "args"      : vars(args),
                }, os.path.join(args.out, "best.pth"))

    # Save final checkpoint
    torch.save({
        "epoch"     : args.epochs,
        "state_dict": model.state_dict(),
        "val_f1"    : va_f1,
        "args"      : vars(args),
    }, os.path.join(args.out, "last.pth"))

    print(f"\nTraining complete.")
    print(f"Best val F1 = {best_f1:.4f} at epoch {best_epoch}")
    print(f"Checkpoints saved to {args.out}/")


def load_model(checkpoint_path: str,
               device: torch.device = torch.device("cpu")) -> nn.Module:
    """
    Convenience function: load a saved checkpoint and return the model.
    Used by all experiment scripts.
    """
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model = build_model(num_classes=2, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ResNet-50 for binary seed classification.")
    parser.add_argument("--datalist",     required=True,
                        help="Dir containing train.txt / val.txt")
    parser.add_argument("--out",          required=True,
                        help="Dir to save checkpoints + log")
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch-size",   type=int,   default=128)
    parser.add_argument("--lr",           type=float, default=0.0012)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device",       default="cuda",
                        choices=["cuda", "cpu", "mps"])
    parser.add_argument("--num-workers",  type=int,   default=4)
    args = parser.parse_args()
    train(args)
