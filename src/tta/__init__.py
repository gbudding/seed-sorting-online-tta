"""
tta/__init__.py
---------------
Adapter registry.  Use `make_adapter(name, model, device, **kwargs)` to
build any method by name.

Available names
---------------
- 'no_correction'        : plain inference (lower bound)
- 'offline_bn_adapt'     : offline BN re-estimation via warmup
- 'offline_tent'         : offline TENT (entropy-min on BN affines)
- 'ema_bn_m{value}'      : online EMA-BN with momentum value
                           e.g. 'ema_bn_m0.01', 'ema_bn_m0.05', 'ema_bn_m0.1'
- 'eata'                 : EATA (entropy-min + reliability+diversity filter)
- 'sar'                  : SAR (sharpness-aware entropy-min + reset)
- 'cotta'                : CoTTA (augmentation-averaged pseudo-labels)
"""

from typing import Callable, Dict

import torch
import torch.nn as nn

from .base     import Adapter, NoCorrection
from .ema_bn   import OnlineEMABN
from .eata     import EATA
from .sar      import SAR
from .cotta    import CoTTA
from .offline  import OfflineBNAdapt, OfflineTENT


def make_adapter(name: str, model: nn.Module, device: torch.device,
                 **kwargs) -> Adapter:
    """Build an Adapter by name.  Raises KeyError on unknown."""
    if name == "no_correction":
        return NoCorrection(model, device)
    if name == "offline_bn_adapt":
        return OfflineBNAdapt(model, device)
    if name == "offline_tent":
        return OfflineTENT(model, device, **kwargs)
    if name.startswith("ema_bn_m"):
        m_str = name[len("ema_bn_m"):]
        momentum = float(m_str)
        return OnlineEMABN(model, device, momentum=momentum)
    if name == "eata":
        return EATA(model, device, **kwargs)
    if name == "sar":
        return SAR(model, device, **kwargs)
    if name == "cotta":
        return CoTTA(model, device, **kwargs)

    raise KeyError(f"Unknown adapter '{name}'")


# All canonical method names that a default experiment sweep should run
DEFAULT_METHODS = [
    "no_correction",
    "offline_bn_adapt",
    "offline_tent",
    "ema_bn_m0.01",
    "ema_bn_m0.05",
    "ema_bn_m0.1",
    "eata",
    "sar",
    "cotta",
]


__all__ = [
    "Adapter",
    "NoCorrection", "OnlineEMABN", "EATA", "SAR", "CoTTA",
    "OfflineBNAdapt", "OfflineTENT",
    "make_adapter", "DEFAULT_METHODS",
]
