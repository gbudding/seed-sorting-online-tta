"""
baseline_curve.py
-----------------
Runs the trained model on the clean test set and computes:
  - Raw confidence scores (softmax probability for class 1 / "good")
  - Yield-threshold curve: yield(t) = fraction of seeds with score >= t
  - AUC-YT: area under the yield-threshold curve
  - Standard classification metrics (accuracy, F1, AUC-ROC)

Outputs (all saved to --out directory):
  baseline_scores.npz     — confidence scores + true labels
  baseline_metrics.json   — scalar metrics
  baseline_curve.pdf      — yield-threshold curve figure

The saved scores are reused by h1_deviations.py so the model
only needs to run once on the clean test set.

Usage
-----
    python src/experiments/baseline_curve.py \
        --datalist   runs/datalist/wheat \
        --checkpoint runs/checkpoints/wheat_resnet50/best.pth \
        --out        runs/results/wheat
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data.dataset       import GrainSetDataset, eval_transforms
from models.train       import load_model


# ------------------------------------------------------------------
# Core functions — reused by all experiment scripts
# ------------------------------------------------------------------

def get_confidence_scores(model, loader, device):
    """
    Run model on all batches and return:
      scores : np.ndarray shape [N]  — P(good | image)
      labels : np.ndarray shape [N]  — true binary labels
    """
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1)
            # Column 1 = probability of class 1 (good)
            scores = probs[:, 1].cpu().numpy()
            all_scores.append(scores)
            all_labels.append(labels.numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)


def yield_threshold_curve(scores, n_thresholds=1000):
    """
    Compute the yield-threshold curve.

    yield(t) = fraction of samples with score >= t

    Parameters
    ----------
    scores       : confidence scores, shape [N]
    n_thresholds : number of threshold values to evaluate

    Returns
    -------
    thresholds : np.ndarray shape [n_thresholds]  — evenly spaced [0, 1]
    yields     : np.ndarray shape [n_thresholds]  — yield at each threshold
    """
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    yields     = np.array([(scores >= t).mean() for t in thresholds])
    return thresholds, yields


def auc_yt(thresholds, yields):
    """Area under the yield-threshold curve (trapezoidal rule)."""
    return float(np.trapezoid(yields, thresholds))


def compute_metrics(scores, labels, threshold=0.5):
    """Standard binary classification metrics at a given threshold."""
    preds   = (scores >= threshold).astype(int)
    acc     = accuracy_score(labels, preds)
    f1      = f1_score(labels, preds, zero_division=0)
    auc_roc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else float("nan")
    return {"accuracy": acc, "f1": f1, "auc_roc": auc_roc}


def plot_curve(thresholds, yields_dict, title="Yield-Threshold Curve",
               save_path=None):
    """
    Plot one or more yield-threshold curves on the same axes.

    Parameters
    ----------
    yields_dict : dict  { label_str : yields_array }
                  The first key is treated as the baseline (drawn thicker).
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    cmap   = plt.cm.tab10
    colors = [cmap(i) for i in range(len(yields_dict))]

    for i, (name, yields) in enumerate(yields_dict.items()):
        lw = 2.5 if i == 0 else 1.5
        ls = "-"  if i == 0 else "--"
        ax.plot(thresholds, yields, label=name, color=colors[i],
                linewidth=lw, linestyle=ls)

    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Yield (fraction accepted)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved -> {save_path}")
    plt.close(fig)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(args):
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load model + data ---------------------------------------
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    test_path = os.path.join(args.datalist, "test.txt")
    test_ds   = GrainSetDataset(test_path, transform=eval_transforms())
    loader    = DataLoader(test_ds, batch_size=256, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    print(f"Test set: {len(test_ds):,} samples")

    # ---- Get scores ----------------------------------------------
    print("Running inference on clean test set ...")
    scores, labels = get_confidence_scores(model, loader, device)

    # ---- Yield-threshold curve -----------------------------------
    thresholds, yields = yield_threshold_curve(scores)
    auc = auc_yt(thresholds, yields)
    metrics = compute_metrics(scores, labels)

    print(f"\nBaseline results:")
    print(f"  AUC-YT   : {auc:.4f}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1       : {metrics['f1']:.4f}")
    print(f"  AUC-ROC  : {metrics['auc_roc']:.4f}")

    # ---- Save outputs --------------------------------------------
    # Raw scores (needed by experiment scripts)
    np.savez(os.path.join(args.out, "baseline_scores.npz"),
             scores=scores, labels=labels,
             thresholds=thresholds, yields=yields)

    # Scalar metrics as JSON
    metrics["auc_yt"] = auc
    with open(os.path.join(args.out, "baseline_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Yield-threshold curve figure
    plot_curve(
        thresholds,
        {"Clean baseline": yields},
        title=f"Baseline Yield-Threshold Curve  (AUC-YT = {auc:.3f})",
        save_path=os.path.join(args.out, "baseline_curve.pdf"))

    print(f"\nOutputs saved to {args.out}/")
    print("Next: run h1_deviations.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute baseline yield-threshold curve on clean test set.")
    parser.add_argument("--datalist",   required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out",        required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    main(args)
