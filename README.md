# Online Test-Time Adaptation for CNN-Based Seed Sorting

> **Follow-up study to the DNE assignment 2 paper**
>
> Gerben Budding

This repository extends the prior study
([Distribution Shift in CNN-Based Seed Sorting](https://github.com/gbudding/cnn-distribution-shift))
from offline correction methods to **streaming, online test-time adaptation
(TTA)** under continuous and recoverable imaging-deviation drift.

The research questions are:

1. **(RQ1)** How well does each online TTA method preserve the per-batch
   yield (at the operator threshold) under gradual and recoverable drift?
2. **(RQ2)** How quickly does each method recover when the drift returns
   to clean conditions?
3. **(RQ3)** How does each method compare to the offline references
   (BN-adapt and TENT, calibrated once at peak severity)?
4. **(RQ4)** What is the per-batch wall-clock cost of each method on a
   CPU-only deployment target?

## Repository structure

```
.
├── data/                      ← download instructions (data not committed)
├── runs/
│   ├── datalist/              ← train/val/test splits (generated)
│   ├── checkpoints/           ← model weights
│   └── results/               ← experiment outputs (.npz + .json + figures)
├── src/
│   ├── data/
│   │   ├── parse_annotations.py   ← XML parser  (from prior repo)
│   │   ├── split.py               ← stratified split  (from prior repo)
│   │   ├── dataset.py             ← PyTorch Dataset + transforms (from prior repo)
│   │   ├── corruptions.py         ← imaging-deviation functions (from prior repo)
│   │   ├── prepare_data.py        ← top-level data prep  (from prior repo)
│   │   ├── sanity_check.py        ← visual + statistical checks (from prior repo)
│   │   ├── drift_sequences.py     ← NEW: gradual + degradation-recovery schedules
│   │   └── streaming.py           ← NEW: phase-shuffled streaming dataloader
│   ├── models/
│   │   └── train.py               ← ResNet-50 trainer + load_model (from prior repo)
│   ├── tta/                       ← NEW: streaming TTA methods
│   │   ├── base.py                ← Adapter ABC + NoCorrection baseline
│   │   ├── ema_bn.py              ← Online EMA-BN
│   │   ├── eata.py                ← EATA (Niu et al. ICML 2022)
│   │   ├── sar.py                 ← SAR  (Niu et al. ICLR 2023)
│   │   ├── cotta.py               ← CoTTA (Wang et al. CVPR 2022)
│   │   └── offline.py             ← Offline BN-adapt + Offline TENT (reference)
│   └── experiments/
│       ├── baseline_curve.py      ← clean YT curve (from prior repo)
│       ├── streaming_eval.py      ← NEW: per-batch evaluator
│       ├── run_drift.py           ← NEW: top-level CLI sweep
│       └── illustrate_yt_curve.py ← NEW: introductory YT-curve figure for the report
├── tests/
│   ├── test_drift_sequences.py    ← 19 unit tests
│   ├── test_tta_methods.py        ← 13 unit tests on a tiny ConvNet
│   ├── test_streaming_eval.py     ← 8 integration tests with synthetic data
│   └── smoke_test_run_drift.py    ← end-to-end CLI smoke test
├── requirements.txt
└── README.md
```

## Methods evaluated

Online (per-batch state update):

- `no_correction`         — plain inference (lower bound)
- `ema_bn_m{0.01,0.05,0.1}` — Online EMA on BN running statistics
- `eata`                  — entropy-min on BN affines + reliability + diversity filter
- `sar`                   — sharpness-aware entropy-min + reset on instability
- `cotta`                 — augmentation-averaged pseudo-labels + stochastic restoration

Offline references (calibrated once at peak severity, then frozen):

- `offline_bn_adapt`      — BN re-estimation
- `offline_tent`          — TENT entropy-min on BN affines

## Drift scenarios

Both scenarios run for 300 batches × 100 images = 30 000 images.

- **gradual_degradation** — six 50-batch phases: clean, sev1, sev2, sev3, sev4, sev5.
- **degradation_recovery** — 30 clean / 5×24 up-ramp / 5×24 down-ramp / 30 clean.

Boundaries are fixed across runs; only image *order within* each phase
varies between seeds.

## Reproducing the experiments

### Step 1 — Data

Download the GrainSet wheat dataset; see `data/download.md` (inherited
from prior repo).  Run `prepare_data.py` to build the train/val/test
splits.

### Step 2 — Model

Either:
- Use the checkpoint from the prior study (preferred — same model
  weights guarantees comparability of online and offline methods); or
- Train fresh:
  ```bash
  python -m src.models.train \
      --datalist  runs/datalist/wheat \
      --out       runs/checkpoints/wheat_resnet50 \
      --device    cpu
  ```

### Step 3 — Clean baseline (yield-threshold curve)

```bash
python -m src.experiments.baseline_curve \
    --datalist   runs/datalist/wheat \
    --checkpoint runs/checkpoints/wheat_resnet50/best.pth \
    --out        runs/results/baseline
```

This produces `baseline_scores.npz`, used as the reference for
Wasserstein distance in the streaming evaluation.

### Step 4 — Pilot run (single condition, timing benchmark)

Run **one** condition on the full test set first to measure
per-batch wall-clock and decide whether to subsample:

```bash
python -m src.experiments.run_drift \
    --datalist    runs/datalist/wheat \
    --checkpoint  runs/checkpoints/wheat_resnet50/best.pth \
    --baseline    runs/results/baseline/baseline_scores.npz \
    --out         runs/results/pilot \
    --scenarios   gradual_degradation \
    --corruptions brightness \
    --methods     no_correction ema_bn_m0.1 \
    --seeds       42 \
    --batch-size  100 \
    --n-batches   300
```

### Step 5 — Full sweep

Once timing is acceptable, run the full sweep:

```bash
python -m src.experiments.run_drift \
    --datalist    runs/datalist/wheat \
    --checkpoint  runs/checkpoints/wheat_resnet50/best.pth \
    --baseline    runs/results/baseline/baseline_scores.npz \
    --out         runs/results/drift \
    --methods     DEFAULT
```

`--methods DEFAULT` runs all nine methods.  Online methods get three
seeds (42, 123, 7); offline methods get one seed (warm-up is
order-invariant).

## Tests

```bash
python -m unittest discover -s tests
```

40 tests, all components exercised through the streaming evaluator on
a tiny synthetic dataset.  Tests run in roughly 8 seconds on CPU.

## Development notes

The source code in this repository was developed with assistance from
Claude (Anthropic), an AI assistant, used primarily for code generation,
debugging, and architectural decisions. All experimental results,
analysis, and written conclusions are the author's own.
