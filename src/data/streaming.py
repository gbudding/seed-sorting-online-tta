"""
streaming.py
------------
Streaming dataset and per-batch corruption application for the
drift-sequence experiments.

Design
------
The proposal requires that:

  1. The test set is partitioned into a fixed number of equal-sized
     batches (default 300 batches × 100 images = 30,000 images).
  2. Phase *boundaries* (which batches are clean / sev1 / ... / sev5)
     are fixed across runs.
  3. Only the *order of images within a phase* varies between runs —
     governed by a per-run seed.

This module exposes:

  - `StreamingPlan`            : a list-of-lists of (path, label) per batch
                                 that has been phase-shuffled with a seed.
  - `make_streaming_plan(...)` : builds a StreamingPlan from a datalist
                                 file plus a DriftSchedule and a seed.
  - `StreamBatchLoader`        : produces tensor batches with the per-batch
                                 corruption applied, in the planned order.
"""

from dataclasses import dataclass
from pathlib import Path
import os
import random
import sys
from typing import Callable, List, Optional, Tuple

import torch
import torchvision.transforms as T
from PIL import Image

# Local imports (assume src/ on PYTHONPATH at runtime)
from .corruptions import get_corruption
from .dataset     import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE
from .drift_sequences import DriftSchedule


# ------------------------------------------------------------------
# Plan construction
# ------------------------------------------------------------------

@dataclass(frozen=True)
class StreamingPlan:
    """
    Per-batch list of samples scheduled for a single drift run.

    Attributes
    ----------
    batches      : list of length n_batches.  Each entry is a list of
                   (image_path, label) tuples of length batch_size.
    schedule     : the DriftSchedule used (provides per-batch severity).
    corruption   : 'brightness' | 'blur' | 'noise' (fixed for this run).
    seed         : random seed used for the within-phase shuffle.
    batch_size   : number of images per batch.
    """
    batches      : List[List[Tuple[str, int]]]
    schedule     : DriftSchedule
    corruption   : str
    seed         : int
    batch_size   : int


def _read_datalist(datalist_path: str) -> List[Tuple[str, int]]:
    """Read a (path, label) datalist from disk in deterministic order."""
    samples = []
    with open(datalist_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path, label = line.rsplit(" ", 1)
            samples.append((path, int(label)))
    # Sort so the baseline order is deterministic regardless of write order
    samples.sort(key=lambda x: x[0])
    return samples


def make_streaming_plan(
    datalist_path : str,
    schedule      : DriftSchedule,
    corruption    : str,
    seed          : int,
    batch_size    : int = 100,
) -> StreamingPlan:
    """
    Build a streaming plan: phase-shuffled, per-batch sample lists.

    Steps:
      1. Load all samples in deterministic baseline order.
      2. Truncate to schedule.n_batches × batch_size samples
         (drops at most batch_size-1 leftover images).
      3. Partition into n_batches contiguous chunks.
      4. Group chunks by phase (per schedule.phase_table).
      5. Within each phase, shuffle the union of its sample lists, using
         the given seed, and reassign chunks to batches in that phase.
      6. Return the per-batch sample lists.

    The number of *images per phase* is preserved by this reshuffle,
    so the empirical class distribution within each phase is identical
    across runs with different seeds.
    """
    if corruption not in ("brightness", "blur", "noise"):
        raise ValueError(
            f"corruption must be one of 'brightness', 'blur', 'noise'; got '{corruption}'")

    n_batches = schedule.n_batches
    samples   = _read_datalist(datalist_path)

    needed = n_batches * batch_size
    if len(samples) < needed:
        raise ValueError(
            f"Need {needed} samples (={n_batches}×{batch_size}) but only "
            f"{len(samples)} found in {datalist_path}")

    samples = samples[:needed]

    # Initial chunking: contiguous batches of size batch_size
    batches: List[List[Tuple[str, int]]] = [
        samples[i*batch_size : (i+1)*batch_size] for i in range(n_batches)
    ]

    # Per-phase reshuffle
    rng = random.Random(seed)
    for (start, end, _sev) in schedule.phase_table:
        phase_samples = []
        for b in range(start, end + 1):
            phase_samples.extend(batches[b])
        rng.shuffle(phase_samples)
        cursor = 0
        for b in range(start, end + 1):
            batches[b] = phase_samples[cursor : cursor + batch_size]
            cursor += batch_size

    return StreamingPlan(
        batches=batches,
        schedule=schedule,
        corruption=corruption,
        seed=seed,
        batch_size=batch_size,
    )


# ------------------------------------------------------------------
# Per-batch loader
# ------------------------------------------------------------------

def _build_transform(corruption: str, severity: int) -> T.Compose:
    """
    Build the per-batch transform: resize → (corruption if sev>0)
    → ToTensor → Normalize.

    This matches the prior study's H1/H3 transform pipeline (see
    src/experiments/h1_deviations.py) so corruption is applied in
    the PIL image domain before tensor conversion.
    """
    steps: List[Callable] = [T.Resize((INPUT_SIZE, INPUT_SIZE))]
    if severity > 0:
        steps.append(get_corruption(corruption, severity))
    steps.extend([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return T.Compose(steps)


class StreamBatchLoader:
    """
    Iterates over the batches of a StreamingPlan, yielding
    (images: Tensor[B, 3, H, W], labels: Tensor[B], plan: BatchPlan).

    The corruption transform is constructed *per batch* based on the
    severity in the schedule.  PIL→PIL corruption (brightness/blur/noise)
    is applied before normalisation, exactly as in the prior study.
    """

    def __init__(self, plan: StreamingPlan):
        self.plan = plan

    def __iter__(self):
        for batch_idx, batch_samples in enumerate(self.plan.batches):
            batch_plan = self.plan.schedule.plans[batch_idx]
            tfm = _build_transform(self.plan.corruption, batch_plan.severity)

            tensors = []
            labels  = []
            for path, label in batch_samples:
                img = Image.open(path).convert("RGB")
                tensors.append(tfm(img))
                labels.append(label)

            images_t = torch.stack(tensors)
            labels_t = torch.tensor(labels, dtype=torch.long)
            yield images_t, labels_t, batch_plan

    def __len__(self):
        return len(self.plan.batches)


# ------------------------------------------------------------------
# CLI smoke test (requires a datalist file)
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.drift_sequences import build_schedule

    p = argparse.ArgumentParser(description="Smoke-test streaming plan.")
    p.add_argument("--datalist",  required=True, help="Path to test.txt")
    p.add_argument("--scenario",  default="gradual_degradation",
                   choices=["gradual_degradation", "degradation_recovery"])
    p.add_argument("--corruption", default="brightness",
                   choices=["brightness", "blur", "noise"])
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--n-batches", type=int, default=300)
    p.add_argument("--n-show",    type=int, default=5,
                   help="Number of batches to actually load (for testing)")
    args = p.parse_args()

    sched = build_schedule(args.scenario, n_batches=args.n_batches)
    plan  = make_streaming_plan(args.datalist, sched, args.corruption,
                                seed=args.seed, batch_size=args.batch_size)

    print(f"Streaming plan built: {len(plan.batches)} batches × "
          f"{args.batch_size} samples = {len(plan.batches)*args.batch_size} images")
    print(f"Corruption type: {plan.corruption}   Seed: {plan.seed}\n")

    loader = StreamBatchLoader(plan)
    for i, (imgs, labels, bp) in enumerate(loader):
        print(f"  batch {bp.batch_index:3d}  sev={bp.severity}  "
              f"images={tuple(imgs.shape)}  labels={tuple(labels.shape)}  "
              f"good={(labels==1).sum().item():3d}/{len(labels)}")
        if i + 1 >= args.n_show:
            break
    print("\nSmoke-test passed ✓")
