"""
base.py
-------
Common Adapter interface for all streaming TTA methods.

Every adapter must implement `adapt_and_predict(images)`, which:
  - Updates internal state (model parameters and/or BN statistics) using
    this batch, if the method does online adaptation;
  - Returns post-update softmax confidences as a numpy array of shape [B].

Returning P(class=1) (the "good" class) follows the convention from
src/experiments/baseline_curve.py and the prior study, so all
downstream metric functions (yield-threshold curve, AUC-YT, Wasserstein)
are reused unchanged.

Each adapter optionally implements `warmup(loader)` for offline methods
that need a pre-inference calibration pass over the peak-severity data.
For online methods this is a no-op.
"""

from abc import ABC, abstractmethod
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn


class Adapter(ABC):
    """Abstract base class for all streaming adapters."""

    name: str = "abstract"

    def __init__(self, model: nn.Module, device: torch.device):
        self.model  = model
        self.device = device

    # -----------------------------------------------------------
    # Optional warmup hook for offline methods
    # -----------------------------------------------------------
    def warmup(self, warmup_batches: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        """
        Run an offline calibration pass.  Default: no-op.

        Parameters
        ----------
        warmup_batches : iterable yielding (images, labels) tensor batches
                         all at peak severity.  The labels are not used
                         (test-time adaptation is unsupervised).
        """
        return None

    # -----------------------------------------------------------
    # Required: streaming step
    # -----------------------------------------------------------
    @abstractmethod
    def adapt_and_predict(self, images: torch.Tensor) -> np.ndarray:
        """
        Update internal state using `images` and return softmax confidences
        for class 1 ("good") of shape [B] as a numpy array.
        """
        ...

    # -----------------------------------------------------------
    # Optional inspection hook
    # -----------------------------------------------------------
    def state_summary(self) -> dict:
        """Return a small dict of state info for logging.  Default empty."""
        return {}


# ------------------------------------------------------------------
# No-correction baseline (just inference)
# ------------------------------------------------------------------

class NoCorrection(Adapter):
    """
    Plain inference, no adaptation.  Serves as the lower bound and the
    reference against which all methods are compared.
    """
    name = "no_correction"

    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__(model, device)
        self.model.eval()

    def adapt_and_predict(self, images: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            logits = self.model(images.to(self.device))
            probs  = torch.softmax(logits, dim=1)
            return probs[:, 1].cpu().numpy()


# ------------------------------------------------------------------
# Helpers shared across BN-based methods
# ------------------------------------------------------------------

def collect_bn_layers(model: nn.Module):
    """Return all BatchNorm2d (and 1d/3d) modules in a model."""
    return [m for m in model.modules()
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]


def collect_bn_affine_params(model: nn.Module):
    """Return list of BN affine parameters (weight + bias) only."""
    params = []
    for m in collect_bn_layers(model):
        if m.weight is not None:
            params.append(m.weight)
        if m.bias is not None:
            params.append(m.bias)
    return params


def freeze_non_bn_affine(model: nn.Module) -> None:
    """
    Set requires_grad=True for BN affine params, False for everything else.
    Used by TENT-family methods (TENT, EATA, SAR).
    """
    for p in model.parameters():
        p.requires_grad_(False)
    for p in collect_bn_affine_params(model):
        p.requires_grad_(True)
