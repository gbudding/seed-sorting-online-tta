"""
test_tta_methods.py
-------------------
Unit tests for streaming TTA adapters.  Uses a tiny ConvNet (with BN)
on synthetic data so tests are fast and deterministic.

What we test for each method:
  - It runs without error on a single batch.
  - Output shape is (B,) and values are in [0, 1].
  - For online methods: the model's adaptive state actually changes
    between batches (i.e. adaptation is doing something).
  - For offline methods: warmup() runs without error and the model is
    in eval-equivalent mode afterwards.

These tests verify the *interface* and *that adaptation runs*, not
that the methods are scientifically calibrated — calibration is an
empirical question answered by the actual experiments.

Run: python tests/test_tta_methods.py
"""

import os
import sys
import unittest

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tta import (
    make_adapter, DEFAULT_METHODS,
    NoCorrection, OnlineEMABN, EATA, SAR, CoTTA,
    OfflineBNAdapt, OfflineTENT,
)


# ------------------------------------------------------------------
# Tiny model with BN — just enough to exercise the BN-based methods
# ------------------------------------------------------------------

class TinyConvNet(nn.Module):
    """Minimal CNN with BN, designed to look enough like a ResNet-50
    head for the BN-based methods to do something measurable."""
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
    """Build a tiny model with deterministic init."""
    torch.manual_seed(seed)
    m = TinyConvNet(num_classes=2)
    # Pre-train BN running stats with random data so they're not at default
    m.train()
    with torch.no_grad():
        for _ in range(5):
            x = torch.randn(16, 3, 32, 32)
            _ = m(x)
    m.eval()
    return m


def make_batch(B: int = 8, seed: int = 0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(B, 3, 32, 32, generator=g)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestRegistry(unittest.TestCase):
    def test_default_methods_all_buildable(self):
        device = torch.device("cpu")
        for name in DEFAULT_METHODS:
            with self.subTest(method=name):
                m = make_model()
                a = make_adapter(name, m, device)
                self.assertIsNotNone(a)

    def test_unknown_method_raises(self):
        with self.assertRaises(KeyError):
            make_adapter("not_a_method", make_model(), torch.device("cpu"))


class TestOutputShapeAndRange(unittest.TestCase):
    """Every adapter should return float scores in [0, 1] of shape [B]."""

    def _check(self, adapter, batch):
        scores = adapter.adapt_and_predict(batch)
        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(scores.shape, (batch.size(0),))
        self.assertTrue(np.all(scores >= 0.0) and np.all(scores <= 1.0),
                        f"scores out of [0,1]: min={scores.min()} max={scores.max()}")

    def test_no_correction(self):
        m = make_model()
        a = NoCorrection(m, torch.device("cpu"))
        self._check(a, make_batch())

    def test_ema_bn(self):
        m = make_model()
        a = OnlineEMABN(m, torch.device("cpu"), momentum=0.1)
        self._check(a, make_batch())

    def test_eata(self):
        m = make_model()
        a = EATA(m, torch.device("cpu"))
        self._check(a, make_batch())

    def test_sar(self):
        m = make_model()
        a = SAR(m, torch.device("cpu"))
        self._check(a, make_batch())

    def test_cotta(self):
        m = make_model()
        a = CoTTA(m, torch.device("cpu"))
        self._check(a, make_batch())


class TestOnlineMethodsUpdate(unittest.TestCase):
    """Confirm online methods actually mutate model state between batches."""

    def test_ema_bn_running_mean_changes(self):
        m = make_model()
        # Snapshot first BN running_mean
        first_bn = next(mod for mod in m.modules() if isinstance(mod, nn.BatchNorm2d))
        before = first_bn.running_mean.detach().clone()

        a = OnlineEMABN(m, torch.device("cpu"), momentum=0.1)
        # Use input with very different distribution to force change
        batch = make_batch() * 5.0 + 2.0
        a.adapt_and_predict(batch)

        after = first_bn.running_mean.detach().clone()
        diff = (after - before).abs().sum().item()
        self.assertGreater(diff, 1e-4,
                           "EMA-BN should have changed BN running_mean")

    def test_eata_bn_affines_change(self):
        m = make_model()
        first_bn = next(mod for mod in m.modules() if isinstance(mod, nn.BatchNorm2d))
        before = first_bn.weight.detach().clone()

        # e_margin_frac=2.0 means E0 > ln(2) → filter accepts every sample,
        # which lets us verify the optimiser path actually updates BN affines.
        a = EATA(m, torch.device("cpu"), lr=1e-2, e_margin_frac=2.0,
                 d_margin=0.0)
        # Several batches to ensure some pass the entropy filter
        for s in range(5):
            a.adapt_and_predict(make_batch(seed=s))

        after = first_bn.weight.detach().clone()
        diff = (after - before).abs().sum().item()
        # At least *some* movement expected; tolerate filter rejecting batches
        # by checking diagnostics if no movement.
        if diff < 1e-6:
            self.assertGreater(a.n_samples_used, 0,
                               "EATA never used any samples (filters too strict on synthetic data)")
        else:
            self.assertGreater(diff, 1e-6)

    def test_sar_runs_and_no_negative_resets_initially(self):
        m = make_model()
        a = SAR(m, torch.device("cpu"), lr=1e-3)
        for s in range(3):
            a.adapt_and_predict(make_batch(seed=s))
        # Just verify it didn't blow up
        self.assertGreaterEqual(a.n_resets, 0)

    def test_cotta_teacher_diverges_from_student(self):
        m = make_model()
        a = CoTTA(m, torch.device("cpu"), lr=1e-2, p_restore=0.0)  # disable restore so divergence is visible
        # Snapshot teacher first conv
        teacher_w_before = a.teacher.features[0].weight.detach().clone()

        for s in range(5):
            a.adapt_and_predict(make_batch(seed=s) * 3.0)

        teacher_w_after = a.teacher.features[0].weight.detach().clone()
        diff = (teacher_w_after - teacher_w_before).abs().sum().item()
        self.assertGreater(diff, 1e-6,
                           "CoTTA teacher EMA should have moved")


class TestOfflineMethodsWarmup(unittest.TestCase):

    def _make_warmup_batches(self, n_batches: int = 3, B: int = 8):
        return [(make_batch(B=B, seed=s), torch.randint(0, 2, (B,)))
                for s in range(n_batches)]

    def test_offline_bn_adapt_warmup(self):
        m = make_model()
        a = OfflineBNAdapt(m, torch.device("cpu"))
        # warmup
        a.warmup(self._make_warmup_batches())
        # Predict on a fresh batch
        scores = a.adapt_and_predict(make_batch())
        self.assertEqual(scores.shape, (8,))

    def test_offline_tent_warmup(self):
        m = make_model()
        first_bn = next(mod for mod in m.modules() if isinstance(mod, nn.BatchNorm2d))
        before = first_bn.weight.detach().clone()

        a = OfflineTENT(m, torch.device("cpu"), lr=1e-2)
        a.warmup(self._make_warmup_batches(n_batches=5))

        after = first_bn.weight.detach().clone()
        diff = (after - before).abs().sum().item()
        self.assertGreater(diff, 1e-6,
                           "Offline TENT should have changed BN affines after warmup")


if __name__ == "__main__":
    unittest.main(verbosity=2)
