"""
ema_bn.py
---------
Online EMA-BN: BatchNorm statistics are updated batch-by-batch with
an exponential moving average (EMA), without modifying any model
parameters.

This is the natural online extension of the offline BN adaptation in
the prior study (Schneider et al., 2020; Li et al., 2017): rather than
a one-shot warm-up pass, every test-time batch contributes to the
running statistics.

Update rule, per BN layer, after the forward pass on each batch:

    μ_running ← (1 − m) · μ_running + m · μ_batch
    σ²_running ← (1 − m) · σ²_running + m · σ²_batch

PyTorch's BatchNorm2d already implements this update when in train()
mode and the `momentum` attribute is set to `m`.  The cleanest
implementation, therefore, is simply to:

    1. set BN layers to train mode (so running stats update),
    2. set BN momentum to m,
    3. forward the batch with no_grad (no parameter updates),
    4. set BN layers back to eval mode for the prediction pass.

Step 4 is necessary because if BN stays in train mode during prediction,
PyTorch's BN2d normalises the prediction with *batch* statistics, not
running statistics — but we explicitly want predictions normalised by
the EMA-tracked running statistics, since those reflect the
accumulated test-time distribution.
"""

import numpy as np
import torch
import torch.nn as nn

from .base import Adapter, collect_bn_layers


class OnlineEMABN(Adapter):
    """
    Online BatchNorm adaptation via EMA on running statistics.

    Hyperparameter
    --------------
    momentum : float in (0, 1].
        Weight on the new batch in the EMA update.  Smaller values =
        slower drift, more stable; larger = faster drift, more
        responsive.  Proposal sweeps m ∈ {0.01, 0.05, 0.1}.
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 momentum: float = 0.1):
        super().__init__(model, device)
        if not (0.0 < momentum <= 1.0):
            raise ValueError("momentum must be in (0, 1]")
        self.momentum = momentum
        self.name = f"ema_bn_m{momentum:g}"

        # Freeze every parameter — EMA-BN only updates running stats
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._bn_layers = collect_bn_layers(self.model)
        # Set the EMA momentum on every BN layer for the duration of
        # this adapter's lifetime.
        for bn in self._bn_layers:
            bn.momentum = self.momentum

    def adapt_and_predict(self, images: torch.Tensor) -> np.ndarray:
        images = images.to(self.device)

        # ---- Step 1: update running stats with this batch -------------
        # BN in train mode + no_grad: statistics update, params don't.
        for bn in self._bn_layers:
            bn.train()
        with torch.no_grad():
            _ = self.model(images)

        # ---- Step 2: predict using updated running stats --------------
        for bn in self._bn_layers:
            bn.eval()
        with torch.no_grad():
            logits = self.model(images)
            probs  = torch.softmax(logits, dim=1)
            return probs[:, 1].cpu().numpy()
