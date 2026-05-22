"""
illustrate_yt_curve.py
----------------------
Produces an introductory figure explaining the yield-threshold (YT)
curve, for readers familiar with ROC / precision-recall but not the
YT curve.

Frouke's feedback (21 April 2026): readers will not know the
yield-threshold curve, so the report should show an example.  This
script generates a self-contained two-panel figure suitable for the
report's introduction.

Panel A: histograms of confidence scores for "good" and "bad" kernels,
         with a vertical line marking an example operator threshold
         t = 0.5.
Panel B: the corresponding yield-threshold curve, derived from the
         union of both histograms (yield(t) = fraction of *all* seeds
         scored ≥ t), with the same threshold marked and the AUC
         shaded.

The illustration uses synthetic confidence scores chosen to look like
typical ResNet-50 outputs on GrainSet (peak near 0.95 for good, broader
spread for bad).  This is purely illustrative — the report should
caption the figure as such.

Usage
-----
    python -m experiments.illustrate_yt_curve --out runs/figures/yt_curve_illustration.pdf
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def synthesise_scores(n_good: int = 1500, n_bad: int = 500,
                      seed: int = 42) -> tuple:
    """
    Sample synthetic confidence scores resembling a typical CNN classifier.

    Returns
    -------
    scores_good, scores_bad  — np.ndarray, each P(good | image) in [0, 1].
    """
    rng = np.random.default_rng(seed)

    # Good kernels: peaked near 0.95, with a small tail towards lower scores
    good = np.clip(rng.beta(8, 1.2, size=n_good), 0.0, 1.0)

    # Bad kernels: bimodal — most are clearly bad (low score), some are
    # near the decision boundary (mid scores).  Mixture.
    n_clear = int(0.7 * n_bad)
    n_amb   = n_bad - n_clear
    bad_clear = np.clip(rng.beta(1.2, 6, size=n_clear), 0.0, 1.0)
    bad_amb   = np.clip(rng.beta(3, 3, size=n_amb), 0.0, 1.0)
    bad = np.concatenate([bad_clear, bad_amb])
    rng.shuffle(bad)

    return good, bad


def yield_threshold_curve(scores: np.ndarray, n_thresholds: int = 1000):
    thresholds = np.linspace(0, 1, n_thresholds)
    yields     = np.array([(scores >= t).mean() for t in thresholds])
    return thresholds, yields


def make_figure(out_path: str, t_op: float = 0.5):
    good, bad = synthesise_scores()
    all_scores = np.concatenate([good, bad])

    thr, yi = yield_threshold_curve(all_scores)
    auc = float(np.trapezoid(yi, thr))

    # Yield at the operator threshold
    yield_at_op = float((all_scores >= t_op).mean())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # ---- Panel A: confidence-score histograms ---------------------
    ax = axes[0]
    bins = np.linspace(0, 1, 41)
    ax.hist(bad,  bins=bins, alpha=0.55, color="#d8443c",
            label=f"Bad kernels (n={len(bad)})")
    ax.hist(good, bins=bins, alpha=0.55, color="#3a8a52",
            label=f"Good kernels (n={len(good)})")
    ax.axvline(t_op, color="black", linestyle="--", linewidth=1.5,
               label=f"Operator threshold t = {t_op}")
    ax.set_xlabel("Confidence score $P(\\mathrm{good}\\mid x)$",
                  fontsize=11)
    ax.set_ylabel("Number of kernels", fontsize=11)
    ax.set_title("(a) Distribution of confidence scores",
                 fontsize=12)
    ax.legend(loc="upper center", fontsize=9)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    # ---- Panel B: yield-threshold curve ---------------------------
    ax = axes[1]
    ax.fill_between(thr, 0, yi, alpha=0.20, color="#1f77b4",
                    label=f"AUC-YT = {auc:.3f}")
    ax.plot(thr, yi, color="#1f77b4", linewidth=2.0,
            label="Yield-threshold curve")
    ax.axvline(t_op, color="black", linestyle="--", linewidth=1.5)
    ax.plot([t_op], [yield_at_op], marker="o", markersize=8,
            color="black",
            label=f"yield({t_op}) = {yield_at_op:.2f}")
    ax.set_xlabel("Threshold $t$", fontsize=11)
    ax.set_ylabel("Yield (fraction of kernels accepted)", fontsize=11)
    ax.set_title("(b) Yield-threshold curve", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "From confidence scores to the yield-threshold curve "
        "(synthetic illustration)",
        fontsize=13, y=1.02)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved -> {out_path}")
    print(f"  AUC-YT (synthetic) = {auc:.3f}")
    print(f"  yield(t={t_op})    = {yield_at_op:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/figures/yt_curve_illustration.pdf",
                   help="Output figure path (.pdf or .png)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Operator threshold to highlight")
    args = p.parse_args()
    make_figure(args.out, t_op=args.threshold)
