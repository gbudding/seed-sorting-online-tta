"""
preview_figures.py
------------------
Renders all four RQ figures from MOCK data, to validate the analysis
design before any real drift runs exist.

Outputs under runs/figures/preview/:
  rq1_trajectories_{brightness,blur,noise}.{pdf,png}
  rq2_auc_yt_heatmap.{pdf,png}
  rq3_recovery_{brightness,blur,noise}.{pdf,png}
  rq4_efficiency_scatter.{pdf,png}

Usage
-----
    python notebooks/preview_figures.py

To turn this into the real analysis:
  - Replace `mock_run(...)` with a loader that reads
    runs/results/drift/<scenario>/<corruption>/<method>__seed<N>.npz
    and returns the same dict shape (yields_t05, auc_yt_batch,
    wallclock_s, severities).
  - Everything else stays the same.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from data.drift_sequences import build_schedule


# ------------------------------------------------------------------
# Constants and method registry
# ------------------------------------------------------------------

CLEAN_BASELINE_YIELD  = 0.86
CLEAN_BASELINE_AUC_YT = 0.594

# Mock yield drops (relative to clean baseline) at each severity, by corruption.
# These approximate the prior study's findings: noise hardest, then blur, then
# brightness; severity 5 produces the biggest drop.  Indexed by severity 0..5.
NO_CORRECTION_DROPS = {
    "brightness": [0.00, 0.08, 0.18, 0.32, 0.48, 0.55],
    "blur":       [0.00, 0.05, 0.12, 0.24, 0.40, 0.50],
    "noise":      [0.00, 0.10, 0.25, 0.42, 0.58, 0.70],
}

# Multiplicative recovery factor: 1.0 = no recovery (= no_correction);
# 0.0 = perfect recovery to clean baseline.
METHOD_RECOVERY = {
    "no_correction":     {"brightness": 1.00, "blur": 1.00, "noise": 1.00},
    "ema_bn_m0.01":      {"brightness": 0.45, "blur": 0.42, "noise": 0.55},
    "ema_bn_m0.05":      {"brightness": 0.32, "blur": 0.30, "noise": 0.40},
    "ema_bn_m0.1":       {"brightness": 0.28, "blur": 0.27, "noise": 0.38},
    "eata":              {"brightness": 0.27, "blur": 0.28, "noise": 0.32},
    "sar":               {"brightness": 0.28, "blur": 0.26, "noise": 0.28},
    "cotta":             {"brightness": 0.30, "blur": 0.29, "noise": 0.36},
    "offline_bn_adapt":  {"brightness": 0.25, "blur": 0.22, "noise": 0.31},
    "offline_tent":      {"brightness": 0.30, "blur": 0.18, "noise": 0.40},
}

# Adaptation lag in batches: how slowly the method tracks a severity change.
# 0 = instant; offline methods don't adapt so they "track" the input instantly
# but at a frozen calibration point (modelled separately below).
METHOD_LAG = {
    "no_correction":     1,
    "ema_bn_m0.01":      30,
    "ema_bn_m0.05":      12,
    "ema_bn_m0.1":       6,
    "eata":              10,
    "sar":               8,
    "cotta":             5,
    "offline_bn_adapt":  1,
    "offline_tent":      1,
}

# Per-batch yield-noise scale (std dev of additive Gaussian).
METHOD_NOISE_SCALE = {
    "no_correction":     0.015,
    "ema_bn_m0.01":      0.015,
    "ema_bn_m0.05":      0.020,
    "ema_bn_m0.1":       0.025,
    "eata":              0.020,
    "sar":               0.018,
    "cotta":             0.025,
    "offline_bn_adapt":  0.012,
    "offline_tent":      0.014,
}

# Plausible per-batch wall-clock for ResNet-50 batch=100 on an RTX A4000.
METHOD_WALLCLOCK_MS = {
    "no_correction":     42,
    "ema_bn_m0.01":      95,
    "ema_bn_m0.05":      95,
    "ema_bn_m0.1":       95,
    "eata":              130,
    "sar":               220,
    "cotta":             300,
    "offline_bn_adapt":  42,
    "offline_tent":      42,
}

ONLINE_METHODS  = ["no_correction", "ema_bn_m0.01", "ema_bn_m0.05",
                   "ema_bn_m0.1", "eata", "sar", "cotta"]
OFFLINE_METHODS = ["offline_bn_adapt", "offline_tent"]
ALL_METHODS     = ONLINE_METHODS + OFFLINE_METHODS

CORRUPTIONS = ["brightness", "blur", "noise"]
SEEDS       = [42, 123, 7]

DISPLAY_NAMES = {
    "no_correction":     "no_correction",
    "ema_bn_m0.01":      "EMA-BN (m=0.01)",
    "ema_bn_m0.05":      "EMA-BN (m=0.05)",
    "ema_bn_m0.1":       "EMA-BN (m=0.1)",
    "eata":              "EATA",
    "sar":               "SAR",
    "cotta":             "CoTTA",
    "offline_bn_adapt":  "offline BN-adapt",
    "offline_tent":      "offline TENT",
}

# Paul Tol / ColorBrewer combination — distinguishable and colorblind-safe.
METHOD_COLORS = {
    "no_correction":     "#888888",
    "ema_bn_m0.01":      "#a6cee3",
    "ema_bn_m0.05":      "#1f78b4",
    "ema_bn_m0.1":       "#08306b",
    "eata":              "#33a02c",
    "sar":               "#e31a1c",
    "cotta":             "#ff7f00",
    "offline_bn_adapt":  "#6a3d9a",
    "offline_tent":      "#b15928",
}


# ------------------------------------------------------------------
# Mock data generator (the only thing that changes vs. real analysis)
# ------------------------------------------------------------------

def _seed_for(method: str, corruption: str, seed: int) -> int:
    """Stable per-condition seed so reruns are reproducible."""
    return (seed * 1_000_003 + abs(hash(method + corruption))) % (2**31 - 1)


def mock_yield_trajectory(method: str, corruption: str, scenario: str,
                          seed: int, n_batches: int = 300) -> tuple:
    """Return (yields_t05[n_batches], severities[n_batches])."""
    sched = build_schedule(scenario, n_batches=n_batches)
    severities = np.array([p.severity for p in sched.plans], dtype=np.int8)

    rng = np.random.default_rng(_seed_for(method, corruption, seed))

    drops_by_severity = NO_CORRECTION_DROPS[corruption]
    recovery_factor   = METHOD_RECOVERY[method][corruption]
    lag               = METHOD_LAG[method]
    noise_scale       = METHOD_NOISE_SCALE[method]

    # Per-batch "instantaneous" target yield (no lag yet).
    target_drops = np.array([drops_by_severity[s] for s in severities]) * recovery_factor
    target_yield = CLEAN_BASELINE_YIELD - target_drops

    # Apply adaptation lag for online methods via simple exponential smoothing.
    if lag <= 1:
        actual = target_yield.copy()
    else:
        actual = np.empty(n_batches)
        actual[0] = CLEAN_BASELINE_YIELD
        alpha = 1.0 / lag
        for i in range(1, n_batches):
            actual[i] = (1 - alpha) * actual[i-1] + alpha * target_yield[i]

    actual = actual + rng.normal(0, noise_scale, size=n_batches)
    actual = np.clip(actual, 0.0, 1.0).astype(np.float32)
    return actual, severities


def mock_auc_yt_from_yield(yields: np.ndarray) -> np.ndarray:
    """Map yield trajectory to a plausible AUC-YT trajectory.

    Pinned so that yield = CLEAN_BASELINE_YIELD → AUC-YT = CLEAN_BASELINE_AUC_YT.
    Real AUC-YT is computed from the full score distribution, not yield@0.5,
    so this is an approximation good only for preview purposes.
    """
    return np.clip(
        CLEAN_BASELINE_AUC_YT * (yields / CLEAN_BASELINE_YIELD) ** 1.2,
        0.0, 1.0
    ).astype(np.float32)


def mock_wallclock(method: str, n_batches: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(_seed_for(method, "wallclock", seed))
    mean_ms = METHOD_WALLCLOCK_MS[method]
    jitter  = mean_ms * 0.05
    return (rng.normal(mean_ms, jitter, n_batches) / 1000.0).astype(np.float32)


def mock_run(method: str, corruption: str, scenario: str, seed: int,
             n_batches: int = 300) -> dict:
    """Return the same dict shape as run_drift_experiment, sans scores/labels."""
    yields, severities = mock_yield_trajectory(method, corruption, scenario,
                                               seed, n_batches=n_batches)
    return {
        "yields_t05":   yields,
        "auc_yt_batch": mock_auc_yt_from_yield(yields),
        "wallclock_s":  mock_wallclock(method, n_batches, seed),
        "severities":   severities,
    }


def gather_runs(methods: list, corruption: str, scenario: str) -> dict:
    """Return {method: stacked_yields[n_seeds, n_batches]}."""
    out = {}
    for method in methods:
        seeds = [SEEDS[0]] if method in OFFLINE_METHODS else SEEDS
        traces = [mock_run(method, corruption, scenario, s)["yields_t05"]
                  for s in seeds]
        out[method] = np.stack(traces)
    return out


# ------------------------------------------------------------------
# Figure builders
# ------------------------------------------------------------------

def fig_rq1_trajectory(corruption: str, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    scenario = "gradual_degradation"
    traces = gather_runs(ONLINE_METHODS, corruption, scenario)

    for method in ONLINE_METHODS:
        arr  = traces[method]
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        x    = np.arange(len(mean))
        c    = METHOD_COLORS[method]
        ax.plot(x, mean, color=c, label=DISPLAY_NAMES[method], lw=1.5)
        ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.18)

    sched = build_schedule(scenario)
    for (start, _end, _sev) in sched.phase_table[1:]:
        ax.axvline(start, color="grey", lw=0.5, alpha=0.5)
    for (start, end, sev) in sched.phase_table:
        mid   = (start + end) / 2
        label = "clean" if sev == 0 else f"sev{sev}"
        ax.text(mid, 1.02, label, ha="center", va="bottom",
                fontsize=9, color="grey")

    ax.axhline(CLEAN_BASELINE_YIELD, color="black", ls="--", lw=0.8,
               alpha=0.5,
               label=f"clean baseline ({CLEAN_BASELINE_YIELD:.2f})")

    ax.set_xlabel("Batch")
    ax.set_ylabel("Yield at threshold t = 0.5")
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"RQ1: yield trajectories under gradual degradation — {corruption}\n"
                 f"(MOCK DATA — for design preview only)")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = out_dir / f"rq1_trajectories_{corruption}.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=120)
    plt.close(fig)
    print(f"  wrote {path.name}")
    return path


def fig_rq2_heatmap(out_dir: Path) -> Path:
    scenario = "gradual_degradation"
    auc_at_peak = np.zeros((len(ALL_METHODS), len(CORRUPTIONS)))
    for i, method in enumerate(ALL_METHODS):
        for j, corruption in enumerate(CORRUPTIONS):
            seeds = [SEEDS[0]] if method in OFFLINE_METHODS else SEEDS
            peaks = [mock_run(method, corruption, scenario, s)
                     ["auc_yt_batch"][-10:].mean() for s in seeds]
            auc_at_peak[i, j] = np.mean(peaks)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(auc_at_peak, cmap="RdYlGn", vmin=0.2, vmax=0.6, aspect="auto")

    ax.set_xticks(range(len(CORRUPTIONS)))
    ax.set_xticklabels(CORRUPTIONS)
    ax.set_yticks(range(len(ALL_METHODS)))
    ax.set_yticklabels([DISPLAY_NAMES[m] for m in ALL_METHODS])

    for i in range(len(ALL_METHODS)):
        for j in range(len(CORRUPTIONS)):
            v = auc_at_peak[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="black" if v > 0.35 else "white", fontsize=10)

    ax.axhline(len(ONLINE_METHODS) - 0.5, color="black", lw=1.5)
    ax.text(len(CORRUPTIONS) - 0.5, len(ONLINE_METHODS) - 0.5,
            "↑ online / ↓ offline", ha="right", va="center",
            fontsize=8, color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))

    cbar = fig.colorbar(im, ax=ax, label="AUC-YT (peak drift, batch 300)")
    cbar.ax.axhline(CLEAN_BASELINE_AUC_YT, color="black", lw=0.8)

    ax.set_xlabel("Corruption type")
    ax.set_title("RQ2: AUC-YT at peak drift (severity 5)\n"
                 f"clean baseline = {CLEAN_BASELINE_AUC_YT}  —  MOCK DATA")
    fig.tight_layout()

    path = out_dir / "rq2_auc_yt_heatmap.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=120)
    plt.close(fig)
    print(f"  wrote {path.name}")
    return path


def fig_rq3_recovery(corruption: str, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    scenario = "degradation_recovery"
    traces = gather_runs(ALL_METHODS, corruption, scenario)

    for method in ALL_METHODS:
        arr  = traces[method]
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        x    = np.arange(len(mean))
        c    = METHOD_COLORS[method]
        ls   = "--" if method in OFFLINE_METHODS else "-"
        ax.plot(x, mean, color=c, label=DISPLAY_NAMES[method], lw=1.5, ls=ls)
        if arr.shape[0] > 1:
            ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.12)

    # Severity reversal marker
    ax.axvline(150, color="black", lw=0.8, alpha=0.7)
    ax.text(150, 1.02, "severity reversal", ha="center", va="bottom",
            fontsize=9, color="black")

    sched = build_schedule(scenario)
    for (start, _end, _sev) in sched.phase_table:
        ax.axvline(start, color="grey", lw=0.3, alpha=0.35)

    # 5% recovery band around clean baseline
    ax.axhline(CLEAN_BASELINE_YIELD, color="black", ls=":", lw=0.8, alpha=0.5)
    ax.axhspan(CLEAN_BASELINE_YIELD * 0.95, CLEAN_BASELINE_YIELD * 1.05,
               color="grey", alpha=0.12)

    ax.set_xlabel("Batch")
    ax.set_ylabel("Yield at threshold t = 0.5")
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"RQ3: degradation-and-recovery — {corruption}\n"
                 "(MOCK DATA — shaded grey band = ±5% of clean baseline)")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = out_dir / f"rq3_recovery_{corruption}.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=120)
    plt.close(fig)
    print(f"  wrote {path.name}")
    return path


def fig_rq4_efficiency(out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))

    summary = []
    for method in ALL_METHODS:
        recoveries = []
        wallclocks = []
        for scenario in ("gradual_degradation", "degradation_recovery"):
            for corruption in CORRUPTIONS:
                seeds = [SEEDS[0]] if method in OFFLINE_METHODS else SEEDS
                for seed in seeds:
                    run = mock_run(method, corruption, scenario, seed)
                    nc  = mock_run("no_correction", corruption, scenario, seed)
                    method_auc = run["auc_yt_batch"].mean()
                    nc_auc     = nc["auc_yt_batch"].mean()
                    span       = CLEAN_BASELINE_AUC_YT - nc_auc
                    rec        = ((method_auc - nc_auc) / span * 100
                                  if span > 0 else 0.0)
                    recoveries.append(rec)
                    wallclocks.append(run["wallclock_s"].mean() * 1000)

        mean_rec = float(np.mean(recoveries))
        mean_wc  = float(np.mean(wallclocks))
        summary.append((method, mean_wc, mean_rec))

        c      = METHOD_COLORS[method]
        marker = "D" if method in OFFLINE_METHODS else "o"
        ax.scatter(mean_wc, mean_rec, c=c, marker=marker, s=140,
                   edgecolors="black", lw=0.8, zorder=3)
        ax.annotate(DISPLAY_NAMES[method], (mean_wc, mean_rec),
                    xytext=(8, 0), textcoords="offset points",
                    fontsize=9, va="center")

    # Upper-left = "best for deployment" callout
    xmax = max(wc for _, wc, _ in summary) * 1.15
    ymax = max(rec for _, _, rec in summary) * 1.15
    ax.set_xlim(0, xmax)
    ax.set_ylim(-5, max(100, ymax))
    ax.axvspan(0, xmax * 0.30, ymin=0.65, ymax=1.0, color="green", alpha=0.06)
    ax.text(xmax * 0.02, ymax * 0.95, "best for deployment\n(high recovery, low overhead)",
            fontsize=9, color="green", va="top")

    online_handle  = plt.Line2D([0], [0], marker="o", color="w",
                                markerfacecolor="grey", markersize=10,
                                label="online (per-batch update)")
    offline_handle = plt.Line2D([0], [0], marker="D", color="w",
                                markerfacecolor="grey", markersize=10,
                                label="offline (frozen post-warmup)")
    ax.legend(handles=[online_handle, offline_handle], loc="lower right")

    ax.set_xlabel("Mean per-batch wall-clock (ms)")
    ax.set_ylabel("Mean AUC-YT recovery vs no_correction (%)")
    ax.set_title("RQ4: efficiency vs recovery — averaged over all corruptions × scenarios\n"
                 "(MOCK DATA — A4000-plausible wall-clocks)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = out_dir / "rq4_efficiency_scatter.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=120)
    plt.close(fig)
    print(f"  wrote {path.name}")

    print()
    print("  RQ4 summary (mock):")
    print(f"  {'method':<22} {'ms/batch':>10} {'recovery %':>12}")
    for method, wc, rec in sorted(summary, key=lambda x: -x[2]):
        print(f"  {DISPLAY_NAMES[method]:<22} {wc:>10.0f} {rec:>12.1f}")
    return path


# ------------------------------------------------------------------
# Recovery-speed table (RQ3 supplementary numeric)
# ------------------------------------------------------------------

def print_recovery_speed_table() -> None:
    """Print recovery speed (batches to within 5% of clean baseline after peak)."""
    print()
    print("RQ3 recovery-speed table (mock; batches after batch 150 to return within 5% of clean):")
    header = f"  {'method':<22} " + " ".join(f"{c:>11}" for c in CORRUPTIONS)
    print(header)
    band_lo = CLEAN_BASELINE_YIELD * 0.95
    for method in ALL_METHODS:
        row = f"  {DISPLAY_NAMES[method]:<22} "
        for corruption in CORRUPTIONS:
            seeds = [SEEDS[0]] if method in OFFLINE_METHODS else SEEDS
            speeds = []
            for seed in seeds:
                run = mock_run(method, corruption, "degradation_recovery", seed)
                y   = run["yields_t05"]
                # batches after 150 until y[k] is back in the band and stays
                recovered_at = None
                for k in range(150, len(y)):
                    if y[k] >= band_lo:
                        recovered_at = k - 150
                        break
                speeds.append(recovered_at if recovered_at is not None else len(y) - 150)
            row += f"{np.mean(speeds):>11.0f} "
        print(row)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    out_dir = ROOT / "runs" / "figures" / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating preview figures (MOCK data) into {out_dir}")
    print()

    print("RQ1 — yield trajectories (gradual_degradation)")
    for corruption in CORRUPTIONS:
        fig_rq1_trajectory(corruption, out_dir)
    print()

    print("RQ2 — AUC-YT heatmap at peak drift")
    fig_rq2_heatmap(out_dir)
    print()

    print("RQ3 — recovery trajectories (degradation_recovery)")
    for corruption in CORRUPTIONS:
        fig_rq3_recovery(corruption, out_dir)
    print_recovery_speed_table()
    print()

    print("RQ4 — efficiency vs recovery")
    fig_rq4_efficiency(out_dir)
    print()

    print(f"Done. Open the PDFs/PNGs in {out_dir} to preview.")


if __name__ == "__main__":
    main()
