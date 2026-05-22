"""
dataset.py
----------
PyTorch Dataset for GrainSet binary classification.

Reads from a plain-text datalist file (one "<path> <label>" per line)
and applies torchvision transforms.

Design notes
------------
- The Dataset is decoupled from the corruption logic: you pass a
  `transform` at construction time, which can be any callable that
  accepts a PIL Image and returns a tensor.  For training use
  `train_transforms()`; for clean evaluation use `eval_transforms()`;
  for H1/H2/H3 experiments wrap the eval transforms with a corruption
  function from corruptions.py.

- ImageNet normalisation statistics are used because the backbone
  (ResNet-50) is pre-trained on ImageNet.

Usage
-----
    from data.dataset import GrainSetDataset, train_transforms, eval_transforms

    train_ds = GrainSetDataset("runs/datalist/train.txt",
                               transform=train_transforms())
    test_ds  = GrainSetDataset("runs/datalist/test.txt",
                               transform=eval_transforms())

    loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                        num_workers=4, pin_memory=True)
"""

import os
from typing import Callable, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# -------------------------------------------------------------------
# ImageNet normalisation (used for all ResNet models)
# -------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

INPUT_SIZE = 224   # standard ResNet input


def train_transforms() -> transforms.Compose:
    """
    Augmentation pipeline for training.
    Matches the setup in Fan et al. (2023):
      random horizontal flip, random rotation, colour jitter, resize, normalise.
    """
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1,
                               saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def eval_transforms() -> transforms.Compose:
    """
    Deterministic pipeline for validation and test evaluation.
    No augmentation — only resize and normalise.
    """
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def eval_transforms_no_norm() -> transforms.Compose:
    """
    Resize only, without normalisation.
    Used when per-channel normalisation is applied at inference time
    (H3 experiment: per-channel input normalisation).
    """
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
    ])


class GrainSetDataset(Dataset):
    """
    PyTorch Dataset that reads a GrainSet datalist file.

    Parameters
    ----------
    datalist_path : str
        Path to a text file with lines of the form "<image_path> <label>".
    transform : callable, optional
        Transform applied to each PIL image before returning.
        If None, returns a plain PIL Image (useful for debugging).
    """

    def __init__(
        self,
        datalist_path: str,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.datalist_path = datalist_path
        self.transform     = transform
        self.samples       = self._load(datalist_path)

    @staticmethod
    def _load(path: str) -> List[Tuple[str, int]]:
        samples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_path, label = line.rsplit(" ", 1)
                samples.append((img_path, int(label)))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    @property
    def labels(self) -> List[int]:
        """All labels as a flat list — useful for class-weight computation."""
        return [lbl for _, lbl in self.samples]

    def class_counts(self) -> Tuple[int, int]:
        """Returns (n_good, n_bad)."""
        lbls   = self.labels
        n_good = sum(l == 1 for l in lbls)
        return n_good, len(lbls) - n_good


def get_class_weights(dataset: GrainSetDataset) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for use with
    torch.nn.CrossEntropyLoss(weight=...) to handle class imbalance.

    Returns a 1-D tensor of shape [2]: [weight_bad, weight_good].
    """
    n_good, n_bad = dataset.class_counts()
    n_total = n_good + n_bad
    # weight[c] = n_total / (n_classes * n_c)
    w_bad  = n_total / (2 * n_bad)  if n_bad  > 0 else 1.0
    w_good = n_total / (2 * n_good) if n_good > 0 else 1.0
    return torch.tensor([w_bad, w_good], dtype=torch.float32)


if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(
        description="Smoke-test the GrainSetDataset.")
    parser.add_argument("datalist", help="Path to a datalist .txt file")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    ds = GrainSetDataset(args.datalist, transform=eval_transforms())
    n_good, n_bad = ds.class_counts()
    print(f"Dataset: {len(ds):,} samples  "
          f"(good={n_good:,}  bad={n_bad:,})")

    weights = get_class_weights(ds)
    print(f"Class weights: bad={weights[0]:.3f}  good={weights[1]:.3f}")

    loader = DataLoader(ds, batch_size=args.batch_size,
                        num_workers=0, pin_memory=False)
    batch_imgs, batch_labels = next(iter(loader))
    print(f"\nFirst batch:")
    print(f"  images : {batch_imgs.shape}   dtype={batch_imgs.dtype}")
    print(f"  labels : {batch_labels.tolist()}")
    print(f"  pixel range after norm: "
          f"[{batch_imgs.min():.2f}, {batch_imgs.max():.2f}]")
    print("\nSmoke-test passed ✓")
