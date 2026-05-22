"""
test_streaming_eval.py
----------------------
End-to-end integration test for the streaming evaluator.

Builds a tiny synthetic image dataset on disk, then runs every adapter
through the streaming evaluator on a short drift schedule, validating:

  - All output arrays have the expected shapes.
  - All metric values are finite and in plausible ranges.
  - Drift severity at each batch matches the schedule.
  - Output .npz / .json files are correctly written and re-readable.

This is the test that catches integration bugs (wrong tensor types,
plan/loader mismatches, etc.) that the per-component tests can miss.

Run: python tests/test_streaming_eval.py
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from data.drift_sequences  import build_schedule
from data.streaming        import make_streaming_plan
from experiments.streaming_eval import run_drift_experiment, save_run_results
from tta import (
    NoCorrection, OnlineEMABN, EATA, SAR, CoTTA, OfflineBNAdapt, OfflineTENT,
)


# ------------------------------------------------------------------
# Tiny model
# ------------------------------------------------------------------

class TinyConvNet(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def make_model(seed: int = 42) -> nn.Module:
    torch.manual_seed(seed)
    m = TinyConvNet(num_classes=2)
    m.train()
    with torch.no_grad():
        for _ in range(5):
            _ = m(torch.randn(8, 3, 32, 32))
    m.eval()
    return m


# ------------------------------------------------------------------
# Synthetic dataset
# ------------------------------------------------------------------

def make_synthetic_dataset(n_images: int, root: str) -> str:
    """
    Write n_images solid-colour PNGs to root/, return path to a datalist
    file with one '<path> <label>' per line.

    Labels alternate 0/1 with slight class imbalance to mimic GrainSet
    (60/40 ratio is approximately preserved).
    """
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(0)
    paths = []
    for i in range(n_images):
        label = 1 if rng.random() < 0.6 else 0
        # Class-conditional colour: good (1) brighter on average
        base  = 180 if label == 1 else 100
        arr   = np.clip(base + rng.normal(0, 20, (32, 32, 3)), 0, 255).astype(np.uint8)
        path  = os.path.join(root, f"img_{i:05d}.png")
        Image.fromarray(arr).save(path)
        paths.append((path, label))

    datalist = os.path.join(root, "test.txt")
    with open(datalist, "w") as f:
        for p, l in paths:
            f.write(f"{p} {l}\n")
    return datalist


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestStreamingEvalIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmpdir   = tempfile.mkdtemp(prefix="stream_eval_test_")
        # Schedule with 12 batches × 4 images = 48 images required
        cls.n_batches = 12
        cls.batch_sz  = 4
        cls.datalist = make_synthetic_dataset(
            n_images=cls.n_batches * cls.batch_sz,
            root=os.path.join(cls.tmpdir, "data"))
        cls.schedule = build_schedule("gradual_degradation",
                                      n_batches=cls.n_batches)

        # Pretend baseline scores (just random, plausibly distributed)
        cls.baseline_scores = np.random.default_rng(0).uniform(0.3, 0.9, 1000).astype(np.float32)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir)

    # ------ helper ------
    def _run(self, adapter):
        plan = make_streaming_plan(
            self.datalist, self.schedule,
            corruption="brightness", seed=42, batch_size=self.batch_sz)
        return run_drift_experiment(
            adapter, plan, self.baseline_scores,
            yield_threshold=0.5, progress_every=0)

    def _assert_output_shapes(self, results):
        n = self.n_batches
        b = self.batch_sz
        self.assertEqual(results["scores"].shape, (n, b))
        self.assertEqual(results["labels"].shape, (n, b))
        for key in ["yields_t05", "auc_yt_batch", "wasserstein",
                    "wallclock_s", "severities"]:
            self.assertEqual(results[key].shape, (n,),
                             f"{key} shape mismatch")
        # Scores in [0, 1]
        s = results["scores"]
        self.assertTrue(np.all(s >= 0.0) and np.all(s <= 1.0))
        # All metrics finite
        for key in ["yields_t05", "auc_yt_batch", "wasserstein", "wallclock_s"]:
            self.assertTrue(np.all(np.isfinite(results[key])),
                            f"{key} contains non-finite values")

    def _assert_severities_match_schedule(self, results):
        expected = np.array(self.schedule.severities(), dtype=np.int8)
        np.testing.assert_array_equal(results["severities"], expected)

    # ------ tests ------
    def test_no_correction(self):
        m = make_model()
        a = NoCorrection(m, torch.device("cpu"))
        r = self._run(a)
        self._assert_output_shapes(r)
        self._assert_severities_match_schedule(r)

    def test_ema_bn(self):
        m = make_model()
        a = OnlineEMABN(m, torch.device("cpu"), momentum=0.1)
        r = self._run(a)
        self._assert_output_shapes(r)

    def test_eata(self):
        m = make_model()
        a = EATA(m, torch.device("cpu"), e_margin_frac=2.0)  # accept all
        r = self._run(a)
        self._assert_output_shapes(r)

    def test_sar(self):
        m = make_model()
        a = SAR(m, torch.device("cpu"), e_margin_frac=2.0)
        r = self._run(a)
        self._assert_output_shapes(r)

    def test_cotta(self):
        m = make_model()
        a = CoTTA(m, torch.device("cpu"), num_augs=2)
        r = self._run(a)
        self._assert_output_shapes(r)

    def test_offline_bn_adapt(self):
        m = make_model()
        a = OfflineBNAdapt(m, torch.device("cpu"))
        # warmup with a small set of synthetic batches
        warmup = [(torch.randn(self.batch_sz, 3, 32, 32),
                   torch.randint(0, 2, (self.batch_sz,))) for _ in range(3)]
        a.warmup(warmup)
        r = self._run(a)
        self._assert_output_shapes(r)

    def test_offline_tent(self):
        m = make_model()
        a = OfflineTENT(m, torch.device("cpu"), lr=1e-3)
        warmup = [(torch.randn(self.batch_sz, 3, 32, 32),
                   torch.randint(0, 2, (self.batch_sz,))) for _ in range(3)]
        a.warmup(warmup)
        r = self._run(a)
        self._assert_output_shapes(r)

    def test_save_and_reload_results(self):
        m = make_model()
        a = NoCorrection(m, torch.device("cpu"))
        r = self._run(a)
        out_path = os.path.join(self.tmpdir, "results", "test_run")
        save_run_results(r, out_path)

        # Reload and verify
        npz  = np.load(out_path + ".npz")
        with open(out_path + ".json") as f:
            meta = json.load(f)

        np.testing.assert_array_equal(npz["scores"], r["scores"])
        np.testing.assert_array_equal(npz["yields_t05"], r["yields_t05"])
        self.assertEqual(meta["method"], "no_correction")
        self.assertEqual(meta["scenario"], "gradual_degradation")
        self.assertEqual(meta["seed"], 42)


if __name__ == "__main__":
    unittest.main(verbosity=2)
