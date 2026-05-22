"""
run_drift.py
------------
Top-level experiment driver for the streaming TTA study.

Runs a sweep over (corruption × scenario × method × seed), saving
per-condition outputs to <out_dir>/<scenario>/<corruption>/<method>__seed<N>.{npz,json}.

For offline methods (offline_bn_adapt, offline_tent), a warm-up loader
is constructed at peak severity (severity 5) using the same datalist;
this matches the protocol described in the proposal:

  "For offline methods, the warm-up pass uses Dstream at the peak
   severity level, consistent with the prior study protocol.  After
   warm-up, the adapted model is held fixed and applied to the entire
   300-batch trajectory."

Online methods are run with three seeds (42, 123, 7) per proposal;
offline methods are run with seed 42 only since their warm-up is
order-invariant.

Example invocation
------------------

    python -m experiments.run_drift \
        --datalist   runs/datalist/wheat \
        --checkpoint runs/checkpoints/wheat_resnet50/best.pth \
        --baseline   runs/results/wheat/baseline_full/baseline_scores.npz \
        --out        runs/results/wheat_drift \
        --scenarios  gradual_degradation degradation_recovery \
        --corruptions brightness blur noise \
        --methods    DEFAULT \
        --batch-size 100 \
        --n-batches  300

Smoke-test invocation (single condition, small):

    python -m experiments.run_drift \
        --datalist   runs/datalist/wheat_tiny \
        --checkpoint runs/checkpoints/wheat_resnet50/best.pth \
        --baseline   runs/results/wheat/baseline_scores.npz \
        --out        runs/results/smoke_test \
        --scenarios  gradual_degradation \
        --corruptions brightness \
        --methods    no_correction ema_bn_m0.1 \
        --batch-size 10 \
        --n-batches  30 \
        --seeds      42
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data.drift_sequences   import build_schedule
from data.streaming         import make_streaming_plan, StreamBatchLoader
from experiments.streaming_eval import run_drift_experiment, save_run_results
from models.train           import load_model
from tta                    import make_adapter, DEFAULT_METHODS


# ------------------------------------------------------------------
# Method classification
# ------------------------------------------------------------------

OFFLINE_METHODS = {"offline_bn_adapt", "offline_tent"}


def is_offline(method: str) -> bool:
    return method in OFFLINE_METHODS


# ------------------------------------------------------------------
# Warm-up loader: feeds the streaming-plan data at fixed peak severity
# ------------------------------------------------------------------

def make_warmup_loader(datalist_path, schedule, corruption, batch_size, peak_severity=5):
    """
    Build a 'warmup' iterable that yields (images, labels) at peak severity
    over the entire test set, using a deterministic order (seed=42).

    Implementation: build a one-phase schedule of the same length, all at
    peak severity, and reuse the streaming plan / loader machinery.
    """
    from data.drift_sequences import DriftSchedule, BatchPlan
    n_batches = schedule.n_batches
    plans = [BatchPlan(batch_index=b, severity=peak_severity)
             for b in range(n_batches)]
    flat_schedule = DriftSchedule(
        name=f"{schedule.name}__warmup_sev{peak_severity}",
        plans=plans,
        n_batches=n_batches,
        phase_table=[(0, n_batches - 1, peak_severity)],
    )
    plan = make_streaming_plan(
        datalist_path, flat_schedule, corruption,
        seed=42, batch_size=batch_size)
    loader = StreamBatchLoader(plan)
    # Yield (images, labels) tuples (the BatchPlan part is unused by warmup)
    def _iter():
        for images, labels, _bp in loader:
            yield images, labels
    return _iter()


# ------------------------------------------------------------------
# Single-condition driver
# ------------------------------------------------------------------

def run_single_condition(args, scenario, corruption, method, seed,
                         baseline_scores, device):
    """Build adapter + plan, optionally warmup, then run the streaming evaluator."""
    test_path = os.path.join(args.datalist, "test.txt")
    schedule  = build_schedule(scenario, n_batches=args.n_batches)

    # Re-load model fresh per condition: TTA mutates state
    print(f"\n--- {scenario} | {corruption} | {method} | seed={seed} ---")
    model = load_model(args.checkpoint, device)
    adapter = make_adapter(method, model, device)

    if is_offline(method):
        print(f"  Warmup pass at severity 5 ({args.n_batches} batches)...")
        warmup_iter = make_warmup_loader(
            test_path, schedule, corruption,
            batch_size=args.batch_size, peak_severity=5)
        t0 = time.perf_counter()
        adapter.warmup(warmup_iter)
        print(f"  Warmup complete in {time.perf_counter()-t0:.1f}s")

    plan = make_streaming_plan(
        test_path, schedule, corruption,
        seed=seed, batch_size=args.batch_size)

    print(f"  Running streaming evaluation ({args.n_batches} batches)...")
    t0 = time.perf_counter()
    results = run_drift_experiment(
        adapter, plan, baseline_scores,
        yield_threshold=args.yield_threshold,
        progress_every=args.progress_every)
    print(f"  Done in {time.perf_counter()-t0:.1f}s")

    out_dir  = os.path.join(args.out, scenario, corruption)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{method}__seed{seed}")
    save_run_results(results, out_path)
    print(f"  Saved -> {out_path}.{{npz,json}}")
    return results


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    print(f"Output dir: {args.out}")
    os.makedirs(args.out, exist_ok=True)

    # Save the run config for reproducibility
    with open(os.path.join(args.out, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Resolve method list
    methods = (DEFAULT_METHODS if args.methods == ["DEFAULT"] else args.methods)

    # Load clean baseline scores (used for Wasserstein reference)
    bl     = np.load(args.baseline)
    bl_scores = bl["scores"]
    print(f"Baseline scores: N={len(bl_scores)}  "
          f"mean={bl_scores.mean():.3f}  std={bl_scores.std():.3f}\n")

    n_total = (len(args.scenarios) * len(args.corruptions)
               * sum(1 + (len(args.seeds) - 1) * (1 - int(is_offline(m))) for m in methods))
    print(f"Total conditions to run: ~{n_total}\n")

    completed = 0
    for scenario in args.scenarios:
        for corruption in args.corruptions:
            for method in methods:
                # Seed sweep — offline methods only need seed 42
                seeds_to_run = [args.seeds[0]] if is_offline(method) else args.seeds
                for seed in seeds_to_run:
                    run_single_condition(
                        args, scenario, corruption, method, seed,
                        bl_scores, device)
                    completed += 1
                    print(f"  [progress] completed {completed} conditions")

    print(f"\nAll done.  Outputs saved to {args.out}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datalist",   required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--baseline",   required=True,
                   help="Path to baseline_scores.npz produced by baseline_curve.py")
    p.add_argument("--out",        required=True)
    p.add_argument("--scenarios",   nargs="+",
                   default=["gradual_degradation", "degradation_recovery"])
    p.add_argument("--corruptions", nargs="+",
                   default=["brightness", "blur", "noise"])
    p.add_argument("--methods", nargs="+", default=["DEFAULT"],
                   help="Method names, or 'DEFAULT' for the proposal sweep")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7])
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--n-batches",  type=int, default=300)
    p.add_argument("--yield-threshold", type=float, default=0.5)
    p.add_argument("--progress-every", type=int, default=25,
                   help="Print progress every N batches (0 = silent)")
    args = p.parse_args()
    main(args)
