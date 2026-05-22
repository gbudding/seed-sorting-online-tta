"""
corruptions.py
--------------
Imaging deviation functions for H1/H2/H3 experiments.

Each corruption is implemented as a callable that accepts a PIL Image
and returns a PIL Image.  They can be composed with each other and
then followed by the standard eval_transforms() pipeline:

    from data.corruptions import BrightnessShift, GaussianBlur, GaussianNoise
    from data.dataset import eval_transforms
    import torchvision.transforms as T

    corruption = T.Compose([
        BrightnessShift(factor=1.3),
        GaussianBlur(sigma=2.0),
        eval_transforms(),
    ])
    deviated_ds = GrainSetDataset(datalist, transform=corruption)

Three deviation types (from the proposal, matching ImageNet-C conventions):
  1. BrightnessShift  — multiplies all pixel values by a scalar factor.
                        factor < 1 → darker;  factor > 1 → brighter.
  2. GaussianBlur     — applies a Gaussian low-pass filter (PIL filter).
  3. GaussianNoise    — adds zero-mean Gaussian noise with std σ.

Five severity levels are defined for each, chosen to span a realistic
range for industrial camera degradation while bracketing the clean
baseline at severity 0.

Reference severity scales are aligned with Hendrycks & Dietterich
(2019), ImageNet-C, for comparability.
"""

import numpy as np
from PIL import Image, ImageFilter


# -------------------------------------------------------------------
# Severity level tables
# Each list has 5 entries corresponding to severities 1–5.
# -------------------------------------------------------------------

BRIGHTNESS_FACTORS = [0.60, 0.75, 0.90, 1.25, 1.50]
# Severity 1 = strong darkening; severity 3 ≈ slight darkening;
# severity 4–5 = brightening.  We keep a dark-only and bright-only
# view in experiments by using subsets.

BLUR_SIGMAS = [0.5, 1.0, 2.0, 3.0, 4.5]
# Gaussian sigma in pixels (image already resized to 224×224).

NOISE_STDS = [0.02, 0.05, 0.10, 0.15, 0.20]
# σ in normalised [0, 1] pixel space.  Clipped to [0, 255] after scaling.


# -------------------------------------------------------------------
# Corruption classes
# -------------------------------------------------------------------

class BrightnessShift:
    """
    Multiply every pixel value by `factor` and clip to [0, 255].

    Simulates camera illumination drift or LED ageing.

    Parameters
    ----------
    factor : float
        Scaling factor.  <1 → darker image; >1 → brighter image.
    """

    def __init__(self, factor: float) -> None:
        self.factor = factor

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img, dtype=np.float32)
        arr = np.clip(arr * self.factor, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def __repr__(self) -> str:
        return f"BrightnessShift(factor={self.factor})"


class GaussianBlur:
    """
    Apply a Gaussian blur with the given sigma.

    Simulates sensor ageing, lens contamination, or focus drift.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian kernel in pixels.
    """

    def __init__(self, sigma: float) -> None:
        self.sigma = sigma

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.filter(ImageFilter.GaussianBlur(radius=self.sigma))

    def __repr__(self) -> str:
        return f"GaussianBlur(sigma={self.sigma})"


class GaussianNoise:
    """
    Add zero-mean Gaussian noise with the given standard deviation.

    Simulates sensor noise differences between cameras.

    Parameters
    ----------
    std : float
        Noise standard deviation in normalised [0, 1] pixel space.
        Internally scaled to [0, 255] range.
    """

    def __init__(self, std: float, seed: int = 0) -> None:
        self.std  = std
        self.rng  = np.random.default_rng(seed)

    def __call__(self, img: Image.Image) -> Image.Image:
        arr   = np.array(img, dtype=np.float32)
        noise = self.rng.normal(loc=0.0,
                                scale=self.std * 255.0,
                                size=arr.shape).astype(np.float32)
        arr   = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def __repr__(self) -> str:
        return f"GaussianNoise(std={self.std})"


# -------------------------------------------------------------------
# Convenience builders
# -------------------------------------------------------------------

def get_corruption(name: str, severity: int):
    """
    Return a corruption callable by name and severity level (1–5).

    Parameters
    ----------
    name     : one of 'brightness', 'blur', 'noise'
    severity : integer 1–5

    Returns
    -------
    A callable that maps PIL Image → PIL Image.
    """
    assert 1 <= severity <= 5, "severity must be between 1 and 5"
    idx = severity - 1  # 0-indexed

    if name == "brightness":
        return BrightnessShift(BRIGHTNESS_FACTORS[idx])
    elif name == "blur":
        return GaussianBlur(BLUR_SIGMAS[idx])
    elif name == "noise":
        return GaussianNoise(NOISE_STDS[idx])
    else:
        raise ValueError(f"Unknown corruption '{name}'. "
                         f"Choose from: brightness, blur, noise")


class CombinedCorruption:
    """
    Applies a sequence of corruptions in order.
    Defined at module level so it can be pickled by multiprocessing workers.
    """
    def __init__(self, fns):
        self.fns = fns
    def __call__(self, img):
        for fn in self.fns:
            img = fn(img)
        return img
    def __repr__(self):
        return f"CombinedCorruption({self.fns})"


def get_all_corruption_configs():
    """
    Return a list of (name, severity, corruption_callable) for all
    7 deviated test conditions used in H1 (single and combined):

      3 single corruptions × 5 severities  (only the 3 individual types)
      4 combination corruptions × 5 severities

    Combinations for severity s apply each component at severity s.

    Returns
    -------
    List of dicts with keys: 'tag', 'severity', 'corruption'
    """
    configs = []

    # Individual corruptions
    for name in ["brightness", "blur", "noise"]:
        for sev in range(1, 6):
            configs.append({
                "tag"       : name,
                "severity"  : sev,
                "corruption": get_corruption(name, sev),
            })

    # All 4 combinations at each severity level
    combo_defs = [
        ("brightness+blur",  ["brightness", "blur"]),
        ("brightness+noise", ["brightness", "noise"]),
        ("blur+noise",       ["blur",        "noise"]),
        ("all",              ["brightness", "blur",   "noise"]),
    ]
    for combo_name, components in combo_defs:
        for sev in range(1, 6):
            parts = [get_corruption(c, sev) for c in components]
            configs.append({
                "tag"       : combo_name,
                "severity"  : sev,
                "corruption": CombinedCorruption(parts),
            })

    return configs


if __name__ == "__main__":
    """Quick visual smoke test — saves example images to /tmp/corruption_test/"""
    import os
    import torchvision.transforms as T
    from PIL import Image
    import urllib.request

    # Download a test wheat kernel image from GrainSet paper figure (illustrative)
    # In practice point this to any PNG in your dataset.
    test_dir = "/tmp/corruption_test"
    os.makedirs(test_dir, exist_ok=True)

    # Create a synthetic 224×224 greyscale gradient image for testing
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(224):
        arr[i, :, :] = i  # gradient
    img = Image.fromarray(arr)
    img.save(os.path.join(test_dir, "original.png"))

    tests = [
        ("brightness_0.6", BrightnessShift(0.6)),
        ("brightness_1.5", BrightnessShift(1.5)),
        ("blur_sigma2",    GaussianBlur(2.0)),
        ("noise_std0.1",   GaussianNoise(0.10)),
    ]
    for tag, corruption in tests:
        out = corruption(img)
        out.save(os.path.join(test_dir, f"{tag}.png"))
        print(f"Saved {tag}.png")

    print(f"\nImages written to {test_dir}")
    print("Smoke-test passed ✓")
