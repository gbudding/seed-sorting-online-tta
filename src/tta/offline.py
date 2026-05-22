"""
offline.py
----------
Offline reference baselines, ported to the Adapter interface so they
can be evaluated through the same streaming loop as the online methods.

These two methods are the offline H3 corrections from the prior study
(Budding, 2026; src/experiments/h3_correction.py).  In the streaming
context, they are calibrated *once* — via a warmup pass over Dstream
at peak severity — and then held fixed for the entire 300-batch
trajectory.  The trajectory metrics for these methods therefore vary
only with the input severity at each batch, not with model state.

This is the protocol described in the proposal:
  "Offline-method yield trajectories therefore vary only with the
   input distribution at each batch, not with model state, and serve
   as a fixed reference point for what a once-calibrated model achieves
   rather than as a competing streaming method."
"""

from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn

from .base import Adapter, collect_bn_affine_params, freeze_non_bn_affine


# ------------------------------------------------------------------
# BatchNorm adaptation: cumulative-mean over the warmup pass
# ------------------------------------------------------------------

class OfflineBNAdapt(Adapter):
    """
    BN running statistics estimated from a warm-up pass over Dstream
    at peak severity, then frozen.

    Mirrors `run_bn_adapt` in src/experiments/h3_correction.py.
    """
    name = "offline_bn_adapt"

    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__(model, device)

    def warmup(self, warmup_batches: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        # Reset BN running stats and accumulate a cumulative average
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
                m.momentum = None         # cumulative moving average
                m.reset_running_stats()
        with torch.no_grad():
            for images, _ in warmup_batches:
                _ = self.model(images.to(self.device))
        # Freeze and switch to eval mode for inference
        self.model.eval()

    def adapt_and_predict(self, images: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            logits = self.model(images.to(self.device))
            return logits.softmax(dim=1)[:, 1].cpu().numpy()


# ------------------------------------------------------------------
# TENT (offline variant): warm-up adapts BN affines once, then frozen
# ------------------------------------------------------------------

class OfflineTENT(Adapter):
    """
    Offline TENT: BN affine parameters are adapted by entropy
    minimisation across a warmup pass over peak-severity Dstream
    (one gradient step per warmup batch), then frozen.

    This is the offline interpretation of TENT used as a reference
    point in the prior study (Budding, 2026).  In the prior code,
    `run_tent` updated per batch through the test set; here we
    interpret that as the offline-once protocol described in the
    proposal so that the streaming trajectory uses a frozen model.
    """
    name = "offline_tent"

    def __init__(self, model: nn.Module, device: torch.device,
                 lr: float = 1e-3):
        super().__init__(model, device)
        self.lr = lr

    def warmup(self, warmup_batches: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        freeze_non_bn_affine(self.model)
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
                m.track_running_stats = False
                m.running_mean = None
                m.running_var  = None

        optim = torch.optim.Adam(collect_bn_affine_params(self.model), lr=self.lr)

        for images, _ in warmup_batches:
            images = images.to(self.device)
            logits = self.model(images)
            probs  = logits.softmax(dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
            optim.zero_grad(set_to_none=True)
            entropy.backward()
            optim.step()

        # Freeze for inference (still BN train mode w/ batch stats — matches prior study)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def adapt_and_predict(self, images: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            logits = self.model(images.to(self.device))
            return logits.softmax(dim=1)[:, 1].cpu().numpy()
