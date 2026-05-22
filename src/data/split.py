"""
split.py
--------
Creates a reproducible, stratified 70 / 15 / 15 train / val / test split
from a flat list of (image_path, label) pairs.

The split is written to three plain-text datalist files:
  <out_dir>/train.txt
  <out_dir>/val.txt
  <out_dir>/test.txt

Each line is:   <absolute_image_path> <label>

This format is compatible with both our custom Dataset class and the
official GrainSet datalists released by the authors.

Design decisions
----------------
- Stratification is performed independently per class so that the
  good/bad ratio is preserved in every split.
- A fixed random seed (42) guarantees reproducibility.
- Impurities (label 0, same as bad) are naturally included; no special
  handling needed since we collapse all non-NOR to 0.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple


SEED   = 42
RATIOS = (0.70, 0.15, 0.15)   # train, val, test


def split_samples(
    samples  : List[Tuple[str, int]],
    ratios   : Tuple[float, float, float] = RATIOS,
    seed     : int = SEED,
) -> Tuple[List, List, List]:
    """
    Stratified split of (path, label) pairs.

    Parameters
    ----------
    samples : list of (path, label)
    ratios  : (train_frac, val_frac, test_frac)  — must sum to 1.0
    seed    : random seed for reproducibility

    Returns
    -------
    train, val, test  — each a list of (path, label)
    """
    assert abs(sum(ratios) - 1.0) < 1e-9, "ratios must sum to 1"

    rng = random.Random(seed)

    # Separate by class
    by_class: dict = {}
    for path, label in samples:
        by_class.setdefault(label, []).append((path, label))

    train_all, val_all, test_all = [], [], []

    for label, class_samples in sorted(by_class.items()):
        shuffled = class_samples[:]
        rng.shuffle(shuffled)

        n      = len(shuffled)
        n_val  = max(1, round(n * ratios[1]))
        n_test = max(1, round(n * ratios[2]))
        n_train = n - n_val - n_test

        train_all.extend(shuffled[:n_train])
        val_all  .extend(shuffled[n_train : n_train + n_val])
        test_all .extend(shuffled[n_train + n_val :])

    # Shuffle each split so batches see mixed classes
    rng.shuffle(train_all)
    rng.shuffle(val_all)
    rng.shuffle(test_all)

    return train_all, val_all, test_all


def write_datalist(samples: List[Tuple[str, int]], path: str) -> None:
    """Write samples to a two-column text file: <path> <label>."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for img_path, label in samples:
            f.write(f"{img_path} {label}\n")
    print(f"Wrote {len(samples):,} samples → {path}")


def read_datalist(path: str) -> List[Tuple[str, int]]:
    """Read a datalist file back into a list of (path, label) tuples."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(" ", 1)
            samples.append((parts[0], int(parts[1])))
    return samples


def create_splits(
    samples  : List[Tuple[str, int]],
    out_dir  : str,
    ratios   : Tuple[float, float, float] = RATIOS,
    seed     : int = SEED,
) -> None:
    """
    Full pipeline: split samples and write the three datalist files.

    Parameters
    ----------
    samples : full list of (path, label) from parse_annotations
    out_dir : directory where train.txt / val.txt / test.txt are written
    """
    train, val, test = split_samples(samples, ratios=ratios, seed=seed)

    write_datalist(train, os.path.join(out_dir, "train.txt"))
    write_datalist(val,   os.path.join(out_dir, "val.txt"))
    write_datalist(test,  os.path.join(out_dir, "test.txt"))

    # Print a summary
    def dist(s):
        n_good = sum(1 for _, l in s if l == 1)
        return f"{len(s):>7,}  (good {n_good:,} / bad {len(s)-n_good:,})"

    print("\nSplit summary:")
    print(f"  Train : {dist(train)}")
    print(f"  Val   : {dist(val)}")
    print(f"  Test  : {dist(test)}")


if __name__ == "__main__":
    import argparse
    from parse_annotations import parse_xml

    parser = argparse.ArgumentParser(
        description="Build train/val/test splits for a GrainSet species.")
    parser.add_argument("xml",     help="Path to species XML")
    parser.add_argument("root",    help="Directory containing PNG images")
    parser.add_argument("out_dir", help="Directory to write datalist files")
    parser.add_argument("--view",  default="UP", choices=["UP", "DOWN"])
    parser.add_argument("--seed",  type=int, default=SEED)
    args = parser.parse_args()

    samples = parse_xml(args.xml, args.root, view=args.view)
    create_splits(samples, args.out_dir, seed=args.seed)
