"""
sar.py
------
SAR (Sharpness-Aware Reliable test-time adaptation), Niu et al. ICLR 2023.

Three additions over TENT:

1. Reliability filter (same idea as EATA): only samples whose entropy
   is below E0 = 0.4 · ln(C) are used.

2. Sharpness-aware update (SAM, Foret et al. 2021):
     a. Forward + backward to get gradient g₁ at current params θ.
     b. Compute perturbation ε = ρ · g₁ / ||g₁||₂  and apply: θ' = θ + ε.
     c. Forward + backward at θ' to get gradient g₂.
     d. Step optimiser using g₂ (and restore θ before stepping, since the
        optimiser will re-apply the update from θ).
   This finds parameter updates that are robust to small perturbations,
   which empirically improves stability under small batch sizes — the
   regime relevant for streaming TTA.

3. Reset on instability: a moving average of (post-update) loss values
   is tracked.  If it exceeds a threshold E1 (default 0.4) the model is
   reset to the source state and adaptation continues from scratch.

Reference
---------
Niu, S., Wu, J., Zhang, Y., Wen, Z., Chen, Y., Zhao, P., & Tan, M.
(2023). Towards stable test-time adaptation in dynamic wild world.
ICLR 2023.  https://arxiv.org/abs/2302.12400
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn

from .base import Adapter, collect_bn_affine_params, freeze_non_bn_affine


def _softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)


class SAR(Adapter):
    """
    Sharpness-aware reliable TTA.

    Hyperparameters
    ---------------
    lr             : base learning rate.  Default 1e-3.
    rho            : SAM perturbation radius.  Paper default 0.05.
    e_margin_frac  : E0 fraction of ln(C) for entropy filter.  Default 0.4.
    reset_thresh   : if EMA-loss exceeds this, reset to source.  Default 0.2.
    momentum_loss  : EMA momentum on the post-update loss.  Default 0.1.
    """
    name = "sar"

    def __init__(self, model: nn.Module, device: torch.device,
                 lr: float = 1e-3, rho: float = 0.05,
                 e_margin_frac: float = 0.4, reset_thresh: float = 0.2,
                 momentum_loss: float = 0.1, num_classes: int = 2):
        super().__init__(model, device)
        self.rho           = rho
        self.lr            = lr
        self.reset_thresh  = reset_thresh
        self.momentum_loss = momentum_loss

        freeze_non_bn_affine(self.model)
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.train()
                m.track_running_stats = False
                m.running_mean = None
                m.running_var  = None

        self.bn_params = collect_bn_affine_params(self.model)
        self.optim     = torch.optim.SGD(self.bn_params, lr=lr, momentum=0.9)

        self.e0        = e_margin_frac * math.log(num_classes)
        self._loss_ema = None

        # Snapshot of source-state BN affines for reset
        self._source_bn_state = [
            (p.detach().clone()) for p in self.bn_params
        ]

        self.n_resets = 0

    # -----------------------------------------------------------------
    def _restore_source(self) -> None:
        """Reset BN affine params to their original values."""
        with torch.no_grad():
            for p, src in zip(self.bn_params, self._source_bn_state):
                p.copy_(src)
        # Reset Adam-style optimiser state — for SGD with momentum we
        # also clear momentum buffers to avoid carrying instability.
        self.optim = torch.optim.SGD(self.bn_params, lr=self.lr, momentum=0.9)
        self._loss_ema = None

    # -----------------------------------------------------------------
    def adapt_and_predict(self, images: torch.Tensor) -> np.ndarray:
        images = images.to(self.device)

        # ---- First forward + filter ----------------------------------
        logits1  = self.model(images)
        ent1     = _softmax_entropy(logits1)
        reliable = ent1 < self.e0

        if not reliable.any():
            # No reliable samples — skip update.  Predict in eval-equivalent.
            with torch.no_grad():
                return logits1.softmax(dim=1)[:, 1].cpu().numpy()

        loss1 = ent1[reliable].mean()

        # ---- SAM perturbation step ----------------------------------
        self.optim.zero_grad(set_to_none=True)
        loss1.backward()

        # ε = ρ · g / ||g||
        with torch.no_grad():
            grad_norm = torch.sqrt(sum(
                p.grad.pow(2).sum() for p in self.bn_params
                if p.grad is not None) + 1e-12)
            scale = self.rho / grad_norm
            saved = []
            for p in self.bn_params:
                if p.grad is None:
                    saved.append(None)
                    continue
                e_w = p.grad * scale
                saved.append(e_w.clone())
                p.add_(e_w)

        # ---- Second forward at θ' ----------------------------------
        logits2  = self.model(images)
        ent2     = _softmax_entropy(logits2)
        reliable2 = ent2 < self.e0
        if reliable2.any():
            loss2 = ent2[reliable2].mean()
        else:
            # If all unreliable at θ', fall back to using all samples but
            # with a small weight to avoid stall.
            loss2 = ent2.mean()

        # Undo perturbation (restore θ) and zero grads.  IMPORTANT: in
        # SAM, backward on loss2 must happen *while parameters are still
        # perturbed* — the autograd graph for loss2 was built against
        # the perturbed weights, so reading their gradients requires
        # those weights to be unchanged at backward() time.  We then
        # restore θ before optim.step() so the step is applied from the
        # original point using the gradient computed at θ'.
        self.optim.zero_grad(set_to_none=True)
        loss2.backward()

        with torch.no_grad():
            for p, e_w in zip(self.bn_params, saved):
                if e_w is not None:
                    p.sub_(e_w)

        self.optim.step()

        # ---- Reset on instability ----------------------------------
        with torch.no_grad():
            current_loss = float(loss2.item())
        if self._loss_ema is None:
            self._loss_ema = current_loss
        else:
            self._loss_ema = ((1 - self.momentum_loss) * self._loss_ema
                              + self.momentum_loss * current_loss)

        if self._loss_ema > self.reset_thresh:
            self._restore_source()
            self.n_resets += 1

        # ---- Predict (post-update) ----------------------------------
        with torch.no_grad():
            logits_post = self.model(images)
            return logits_post.softmax(dim=1)[:, 1].cpu().numpy()

    def state_summary(self) -> dict:
        return {
            "loss_ema": self._loss_ema,
            "n_resets": self.n_resets,
        }
