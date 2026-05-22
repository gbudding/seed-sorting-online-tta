"""
streaming_eval.py
-----------------
Drives a single (corruption × scenario × method × seed) drift run.

Per batch, this evaluator:
  1. Pulls (images, labels, batch_plan) from the streaming loader.
  2. Calls adapter.adapt_and_predict(images), recording wall-clock time.
  3. Computes:
       - yield_at_t(t = 0.5)               (primary metric, RQ1)
       - per-batch yield-threshold curve   (for AUC-YT at end-of-phase)
       - Wasserstein distance to clean baseline scores (secondary)
       - inference time (per batch, RQ4)
  4. Stores all per-batch metrics in arrays of shape [n_batches].

Outputs are saved as a single .npz file containing:
  scores         : array [n_batches, batch_size]  – raw P(good) scores
  labels         : array [n_batches, batch_size]
  yields_t05     : array [n_batches]              – yield at t=0.5
  auc_yt_batch   : array [n_batches]              – per-batch AUC-YT
  wasserstein    : array [n_batches]              – W-distance to baseline
  wallclock_s    : array [n_batches]              – per-batch seconds
  severities     : array [n_batches]              – per-batch severity
  meta           : dict                           – method/corruption/scenario/seed

Per-phase summaries (mean yield, AUC-YT at the last batch of each phase)
are derived afterwards in the analysis script.
"""

import json
import os
import sys
import time
from typing import Optional

import numpy as np
import torch
from scipy.stats import wasserstein_distance

# Local imports
from data.streaming        import StreamBatchLoader, StreamingPlan
from experiments.baseline_curve import yield_threshold_curve, auc_yt
from tta                   import Adapter


# ------------------------------------------------------------------
# Run a single drift experiment
# ------------------------------------------------------------------

def run_drift_experiment(
    adapter           : Adapter,
    plan              : StreamingPlan,
    baseline_scores   : np.ndarray,
    yield_threshold   : float = 0.5,
    progress_every    : int   = 25,
) -> dict:
    """
    Execute a streaming drift run.

    Parameters
    ----------
    adapter         : an Adapter instance.  For offline methods, warmup()
                      must have been called *before* this function.
    plan            : streaming plan (defines the batch order + corruptions).
    baseline_scores : 1-D array of clean-baseline scores, used as the
                      reference distribution for Wasserstein distance.
    yield_threshold : threshold t for the primary yield metric.  Default 0.5.

    Returns
    -------
    dict with all per-batch arrays and metadata.
    """
    n_batches = plan.schedule.n_batches
    B         = plan.batch_size

    yields_t05    = np.empty(n_batches, dtype=np.float32)
    auc_yt_batch  = np.empty(n_batches, dtype=np.float32)
    wasserstein   = np.empty(n_batches, dtype=np.float32)
    wallclock_s   = np.empty(n_batches, dtype=np.float32)
    severities    = np.empty(n_batches, dtype=np.int8)
    scores_all    = np.empty((n_batches, B), dtype=np.float32)
    labels_all    = np.empty((n_batches, B), dtype=np.int64)

    loader = StreamBatchLoader(plan)

    for batch_idx, (images, labels, bp) in enumerate(loader):
        t0 = time.perf_counter()
        scores = adapter.adapt_and_predict(images)
        t1 = time.perf_counter()

        # Per-batch metrics
        yields_t05[batch_idx]   = float((scores >= yield_threshold).mean())
        thr, yi                 = yield_threshold_curve(scores)
        auc_yt_batch[batch_idx] = float(auc_yt(thr, yi))
        wasserstein[batch_idx]  = float(wasserstein_distance(baseline_scores, scores))
        wallclock_s[batch_idx]  = t1 - t0
        severities[batch_idx]   = bp.severity
        scores_all[batch_idx]   = scores
        labels_all[batch_idx]   = labels.numpy()

        if progress_every and (batch_idx + 1) % progress_every == 0:
            mean_yield = yields_t05[: batch_idx + 1].mean()
            print(f"  [{batch_idx+1:>3d}/{n_batches}] sev={bp.severity}  "
                  f"yield(t=0.5)={yields_t05[batch_idx]:.3f}  "
                  f"AUC-YT={auc_yt_batch[batch_idx]:.3f}  "
                  f"W={wasserstein[batch_idx]:.3f}  "
                  f"running_mean_yield={mean_yield:.3f}  "
                  f"({wallclock_s[batch_idx]*1000:.0f} ms/batch)",
                  flush=True)

    meta = {
        "method"     : adapter.name,
        "corruption" : plan.corruption,
        "scenario"   : plan.schedule.name,
        "seed"       : plan.seed,
        "n_batches"  : n_batches,
        "batch_size" : B,
        "yield_threshold": yield_threshold,
        "adapter_state"  : adapter.state_summary(),
    }

    return {
        "scores"        : scores_all,
        "labels"        : labels_all,
        "yields_t05"    : yields_t05,
        "auc_yt_batch"  : auc_yt_batch,
        "wasserstein"   : wasserstein,
        "wallclock_s"   : wallclock_s,
        "severities"    : severities,
        "meta"          : meta,
    }


def save_run_results(results: dict, out_path: str) -> None:
    """
    Save a run's results to a single .npz + a .json sidecar.

    Parameters
    ----------
    results  : dict returned by run_drift_experiment.
    out_path : path *without* extension.  Two files are written:
                  out_path + ".npz"   (arrays)
                  out_path + ".json"  (meta)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path + ".npz",
        scores       = results["scores"],
        labels       = results["labels"],
        yields_t05   = results["yields_t05"],
        auc_yt_batch = results["auc_yt_batch"],
        wasserstein  = results["wasserstein"],
        wallclock_s  = results["wallclock_s"],
        severities   = results["severities"],
    )
    with open(out_path + ".json", "w") as f:
        json.dump(results["meta"], f, indent=2)
