"""
eata.py
-------
EATA (Efficient Anti-forgetting Test-time Adaptation), Niu et al. ICML 2022.

Two key additions over plain TENT:

1. Reliability filter: only samples whose prediction entropy falls
   below a threshold E0 are used to compute the loss.  High-entropy
   samples are predicted to be near the decision boundary and
   contribute noisy gradient signal.  Default E0 = 0.4 · ln(C) where
   C is the number of classes (so E0 = 0.4 · ln(2) ≈ 0.277 for the
   binary good/bad task).

2. Diversity filter: the prediction (softmax) of each candidate
   reliable sample is compared (cosine) to a moving average of past
   reliable predictions.  Samples too similar (cosine > 1 − D0) are
   skipped to avoid redundant updates.  Default D0 = 0.05.

Affine parameters of BatchNorm are the only updated weights, as in
TENT.  All other parameters are frozen.  The original paper also
includes a Fisher-information regulariser to prevent forgetting; this
is omitted here for simplicity (as is common in re-implementations:
the regulariser primarily helps in continual settings with multiple
domains, not a single drifting one).

Reference
---------
Niu, S., Wu, J., Zhang, Y., Chen, Y., Zheng, S., Zhao, P., & Tan, M.
(2022). Efficient test-time model adaptation without forgetting.
ICML 2022.  https://arxiv.org/abs/2204.02610
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Adapter, collect_bn_affine_params, freeze_non_bn_affine


def _softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax outputs.  Returns a vector of per-sample entropies."""
    return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)


class EATA(Adapter):
    """
    EATA online test-time adaptation.

    Hyperparameters
    ---------------
    lr            : learning rate for BN affine params (Adam).  Default 1e-3,
                    matching TENT in the prior study (h3_correction.py).
    e_margin_frac : E0 fraction of ln(num_classes); default 0.4 (paper default).
    d_margin      : cosine similarity above which a sample is considered
                    redundant and skipped.  Paper default 0.05 (i.e. cosine
                    similarity > 1 − 0.05 = 0.95 ⇒ skip).
    """
    name = "eata"

    def __init__(self, model: nn.Module, device: torch.device,
                 lr: float = 1e-3, e_margin_frac: float = 0.4,
                 d_margin: float = 0.05, num_classes: int = 2):
        super().__init__(model, device)

        freeze_non_bn_affine(self.model)
        # Put BN layers in train mode and disable running-stats tracking
        # (TENT-style: use batch statistics each step, do not pollute
        # running stats with corrupted batches).
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.train()
                m.track_running_stats = False
                m.running_mean = None
                m.running_var  = None

        self.optim = torch.optim.Adam(
            collect_bn_affine_params(self.model), lr=lr)

        self.e0       = e_margin_frac * math.log(num_classes)
        self.d_margin = d_margin
        self.num_classes = num_classes

        # Moving average of past reliable softmax outputs (initialised lazily)
        self._past_softmax_avg = None

        # Lightweight diagnostics
        self.n_samples_seen     = 0
        self.n_samples_used     = 0
        self.n_filtered_entropy = 0
        self.n_filtered_div     = 0

    def adapt_and_predict(self, images: torch.Tensor) -> np.ndarray:
        images = images.to(self.device)
        B = images.shape[0]
        self.n_samples_seen += B

        # ---- Forward + entropy ---------------------------------------
        logits  = self.model(images)
        entropy = _softmax_entropy(logits)
        probs   = logits.softmax(dim=1)

        # ---- Reliability filter --------------------------------------
        reliable = entropy < self.e0
        self.n_filtered_entropy += int((~reliable).sum())

        # ---- Diversity filter ----------------------------------------
        if self._past_softmax_avg is not None and reliable.any():
            past_mean = self._past_softmax_avg.unsqueeze(0)         # [1, C]
            cos_sim   = F.cosine_similarity(probs, past_mean, dim=1)  # [B]
            diverse   = cos_sim < (1.0 - self.d_margin)             # [B]
            sel_mask  = reliable & diverse
            self.n_filtered_div += int((reliable & ~diverse).sum())
        else:
            sel_mask = reliable

        # ---- Loss + step (only over selected samples) ----------------
        if sel_mask.any():
            loss = entropy[sel_mask].mean()
            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            self.optim.step()
            self.n_samples_used += int(sel_mask.sum())

            # Update past-softmax EMA with selected samples
            with torch.no_grad():
                batch_mean = probs[sel_mask].mean(dim=0)
                if self._past_softmax_avg is None:
                    self._past_softmax_avg = batch_mean.detach().clone()
                else:
                    self._past_softmax_avg = (
                        0.9 * self._past_softmax_avg + 0.1 * batch_mean)

        # ---- Predict (post-update) ----------------------------------
        with torch.no_grad():
            logits_post = self.model(images)
            return logits_post.softmax(dim=1)[:, 1].cpu().numpy()

    def state_summary(self) -> dict:
        return {
            "samples_seen"     : self.n_samples_seen,
            "samples_used"     : self.n_samples_used,
            "filtered_entropy" : self.n_filtered_entropy,
            "filtered_diversity": self.n_filtered_div,
        }
