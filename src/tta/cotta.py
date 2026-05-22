"""
cotta.py
--------
CoTTA (Continual Test-Time Adaptation), Wang et al. CVPR 2022.

Three components:

1. Teacher / student networks
   - Student: the model that is updated at every batch.
   - Teacher: an EMA of the student's weights.  It produces the
              pseudo-labels used as supervision.

2. Augmentation-averaged pseudo-labels
   - For each input, generate K augmented copies, run them through the
     teacher, and average the softmax outputs to produce a pseudo-label.
     Averaging across augmentations reduces label noise in domains where
     the teacher's prediction is sensitive to small input perturbations.
   - The student is then trained to match this pseudo-label
     (cross-entropy from softmax(student) towards averaged-teacher).

3. Stochastic restoration
   - After the gradient step, with probability p_restore (per-parameter),
     each parameter element is reset to its source (pre-adaptation) value.
   - This prevents catastrophic forgetting when adaptation drifts far
     from the source distribution and is important under prolonged drift.

Reference
---------
Wang, Q., Fink, O., Van Gool, L., & Dai, D. (2022). Continual test-time
domain adaptation.  CVPR 2022.  https://arxiv.org/abs/2203.13591
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

from .base import Adapter


# ---------------------------------------------------------------------
# Lightweight augmentations that don't change the spatial shape
# ---------------------------------------------------------------------
def _build_augmentations(num_augs: int = 4):
    """
    Returns a list of augmentation callables that can be applied to a
    *normalised* tensor batch (shape [B, 3, H, W]).
    """
    augs = []
    # Horizontal flip
    augs.append(lambda x: torch.flip(x, dims=[3]))
    # Vertical flip (kernels look fine flipped vertically too)
    augs.append(lambda x: torch.flip(x, dims=[2]))
    # 90-degree rotation
    augs.append(lambda x: torch.rot90(x, k=1, dims=[2, 3]))
    # Identity (so the original is included in the average too)
    augs.append(lambda x: x)
    return augs[:max(1, num_augs)]


class CoTTA(Adapter):
    """
    Continual TTA with EMA teacher and stochastic restoration.

    Hyperparameters
    ---------------
    lr            : student learning rate.  Default 1e-3.
    teacher_alpha : teacher EMA momentum (towards student).  Paper default 0.999.
    p_restore     : per-element probability of restoration to source.  Paper default 0.01.
    num_augs      : number of augmentations averaged for pseudo-labels.
                    Paper uses 32; we default to 4 to keep CPU runtime feasible.
    """
    name = "cotta"

    def __init__(self, model: nn.Module, device: torch.device,
                 lr: float = 1e-3, teacher_alpha: float = 0.999,
                 p_restore: float = 0.01, num_augs: int = 4):
        super().__init__(model, device)
        self.teacher_alpha = teacher_alpha
        self.p_restore     = p_restore
        self.augs          = _build_augmentations(num_augs)

        # Source state (frozen reference for stochastic restoration)
        self._source_state = copy.deepcopy(model.state_dict())

        # Teacher: deep copy of the model, no gradients
        self.teacher = copy.deepcopy(model).to(device)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

        # Student: optimise *all* parameters (CoTTA paper updates all weights)
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.model.train()

        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)

    # -----------------------------------------------------------------
    @torch.no_grad()
    def _update_teacher(self) -> None:
        """EMA update of teacher weights from student."""
        a = self.teacher_alpha
        for p_t, p_s in zip(self.teacher.parameters(), self.model.parameters()):
            p_t.data.mul_(a).add_(p_s.data, alpha=1.0 - a)
        # Also EMA the buffers (BN running stats etc.) — keeps teacher coherent.
        for b_t, b_s in zip(self.teacher.buffers(), self.model.buffers()):
            if b_t.dtype.is_floating_point:
                b_t.data.mul_(a).add_(b_s.data.float(), alpha=1.0 - a)
            else:
                b_t.data.copy_(b_s.data)

    # -----------------------------------------------------------------
    @torch.no_grad()
    def _stochastic_restore(self) -> None:
        """For each parameter element, with prob p_restore, reset to source."""
        for name, p in self.model.named_parameters():
            if name not in self._source_state:
                continue
            src   = self._source_state[name].to(p.device)
            mask  = (torch.rand_like(p) < self.p_restore)
            p.data = torch.where(mask, src, p.data)

    # -----------------------------------------------------------------
    @torch.no_grad()
    def _teacher_pseudo_label(self, images: torch.Tensor) -> torch.Tensor:
        """Average teacher softmax over augmentations.  Returns [B, C]."""
        accum = None
        for aug in self.augs:
            x = aug(images)
            logits = self.teacher(x)
            probs  = logits.softmax(dim=1)
            # If the augmentation is a flip in the spatial dimension,
            # the softmax is identical (we don't actually transform back),
            # so averaging is well-defined.
            accum = probs if accum is None else accum + probs
        return accum / len(self.augs)

    # -----------------------------------------------------------------
    def adapt_and_predict(self, images: torch.Tensor) -> np.ndarray:
        images = images.to(self.device)

        # 1) Pseudo-label from EMA teacher (no_grad)
        pseudo = self._teacher_pseudo_label(images)

        # 2) Student forward + cross-entropy to pseudo-label
        self.model.train()
        logits_s  = self.model(images)
        log_probs = logits_s.log_softmax(dim=1)
        loss      = -(pseudo * log_probs).sum(dim=1).mean()

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        self.optim.step()

        # 3) EMA teacher update + stochastic restoration
        self._update_teacher()
        self._stochastic_restore()

        # 4) Predict using the (just-updated) student in eval-mode forward
        self.model.eval()
        with torch.no_grad():
            logits_post = self.model(images)
            return logits_post.softmax(dim=1)[:, 1].cpu().numpy()
