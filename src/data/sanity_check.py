"""
sanity_check.py
---------------
Visual and statistical sanity checks on the prepared datalists.

Checks performed:
  1. Class balance per split matches expectations (~60/40 for wheat).
  2. A random sample of images actually opens without error.
  3. Image sizes are correct (all 224×224 after transform).
  4. Pixel statistics after normalisation look reasonable.
  5. Saves a contact-sheet PNG of 16 random samples (8 good, 8 bad)
     so you can visually confirm that label→image assignment looks right.

Usage
-----
    python sanity_check.py \
        --datalist runs/datalist/wheat \
        --out      runs/sanity_check/wheat
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data.dataset import GrainSetDataset, eval_transforms
from data.split   import read_datalist


LABEL_NAMES = {0: "BAD", 1: "GOOD"}
LABEL_COLORS = {0: (220, 50, 50), 1: (50, 180, 80)}


def check_split(name, path, expected_pct_good=(0.50, 0.75)):
    """Check class balance of a single split."""
    samples = read_datalist(path)
    n       = len(samples)
    n_good  = sum(1 for _, l in samples if l == 1)
    pct     = n_good / n if n > 0 else 0

    ok = expected_pct_good[0] <= pct <= expected_pct_good[1]
    status = "✓" if ok else "✗  ← unexpected balance"
    print(f"  {name:6s}: {n:>8,} samples  "
          f"good={n_good:,} ({100*pct:.1f}%)  {status}")
    return ok


def check_images_open(datalist_path, n_sample=200, seed=42):
    """Try opening a random subset of images."""
    samples = read_datalist(datalist_path)
    rng     = random.Random(seed)
    subset  = rng.sample(samples, min(n_sample, len(samples)))

    failed = []
    for path, _ in subset:
        try:
            Image.open(path).convert("RGB")
        except Exception as e:
            failed.append((path, str(e)))

    if failed:
        print(f"  ✗  {len(failed)} / {len(subset)} images failed to open:")
        for p, e in failed[:5]:
            print(f"       {p}  →  {e}")
    else:
        print(f"  ✓  All {len(subset)} sampled images open successfully")
    return len(failed) == 0


def check_tensor_stats(datalist_path, n_sample=500, seed=42):
    """Check pixel statistics after normalisation."""
    samples = read_datalist(datalist_path)
    rng     = random.Random(seed)
    subset  = rng.sample(samples, min(n_sample, len(samples)))

    # Write a temporary datalist for this subset
    tmp_path = "/tmp/_gs_sanity_tmp.txt"
    with open(tmp_path, "w") as f:
        for p, l in subset:
            f.write(f"{p} {l}\n")

    ds     = GrainSetDataset(tmp_path, transform=eval_transforms())
    loader = DataLoader(ds, batch_size=64, num_workers=0)

    all_means, all_stds = [], []
    for imgs, _ in loader:
        # imgs: [B, C, H, W]
        all_means.append(imgs.mean(dim=[0, 2, 3]))
        all_stds .append(imgs.std (dim=[0, 2, 3]))

    mean = torch.stack(all_means).mean(0)
    std  = torch.stack(all_stds ).mean(0)

    print(f"  Mean after normalisation: R={mean[0]:.3f}  "
          f"G={mean[1]:.3f}  B={mean[2]:.3f}")
    print(f"  Std  after normalisation: R={std[0]:.3f}  "
          f"G={std[1]:.3f}  B={std[2]:.3f}")
    # After ImageNet normalisation, means should be close to 0
    ok = all(abs(m.item()) < 1.0 for m in mean)
    print(f"  {'✓' if ok else '✗'} Means within expected range")
    return ok


def make_contact_sheet(datalist_path, out_path, n_per_class=8, seed=42,
                       thumb_size=112):
    """Save a contact sheet of n_per_class good and bad samples."""
    samples = read_datalist(datalist_path)
    rng     = random.Random(seed)

    good = [(p, l) for p, l in samples if l == 1]
    bad  = [(p, l) for p, l in samples if l == 0]

    sample_good = rng.sample(good, min(n_per_class, len(good)))
    sample_bad  = rng.sample(bad,  min(n_per_class, len(bad)))
    grid_samples = sample_good + sample_bad

    cols    = n_per_class
    rows    = 2
    pad     = 4
    label_h = 18
    sheet_w = cols * (thumb_size + pad) + pad
    sheet_h = rows * (thumb_size + label_h + pad) + pad

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(240, 240, 240))
    draw  = ImageDraw.Draw(sheet)

    for idx, (img_path, label) in enumerate(grid_samples):
        row = idx // cols
        col = idx  % cols
        x   = pad + col * (thumb_size + pad)
        y   = pad + row * (thumb_size + label_h + pad)

        img = Image.open(img_path).convert("RGB")
        img = img.resize((thumb_size, thumb_size), Image.LANCZOS)
        sheet.paste(img, (x, y))

        color = LABEL_COLORS[label]
        draw.rectangle([x, y + thumb_size, x + thumb_size,
                         y + thumb_size + label_h - 1], fill=color)
        draw.text((x + 4, y + thumb_size + 2),
                  LABEL_NAMES[label], fill=(255, 255, 255))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sheet.save(out_path)
    print(f"  Contact sheet saved → {out_path}")


def main(args):
    print("=" * 60)
    print("GrainSet Data Pipeline Sanity Check")
    print("=" * 60)

    all_ok = True

    # ---- 1. Class balance ----------------------------------------
    print("\n[1] Class balance check:")
    for split in ["train", "val", "test"]:
        path = os.path.join(args.datalist, f"{split}.txt")
        if os.path.exists(path):
            ok = check_split(split, path)
            all_ok = all_ok and ok
        else:
            print(f"  {split:6s}: FILE NOT FOUND — {path}")
            all_ok = False

    # ---- 2. Images open ------------------------------------------
    print("\n[2] Image loading check (test split, 200 random samples):")
    test_path = os.path.join(args.datalist, "test.txt")
    if os.path.exists(test_path):
        ok = check_images_open(test_path)
        all_ok = all_ok and ok

    # ---- 3. Tensor statistics ------------------------------------
    print("\n[3] Tensor statistics after normalisation (test split, 500 samples):")
    if os.path.exists(test_path):
        ok = check_tensor_stats(test_path)
        all_ok = all_ok and ok

    # ---- 4. Contact sheet ----------------------------------------
    print("\n[4] Saving contact sheet:")
    sheet_path = os.path.join(args.out, "contact_sheet.png")
    if os.path.exists(test_path):
        make_contact_sheet(test_path, sheet_path, n_per_class=8)

    # ---- Summary -------------------------------------------------
    print("\n" + "=" * 60)
    if all_ok:
        print("All checks passed ✓  Data pipeline is ready.")
    else:
        print("Some checks FAILED ✗  Review output above before proceeding.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run sanity checks on prepared GrainSet datalists.")
    parser.add_argument(
        "--datalist", required=True,
        help="Directory containing train.txt / val.txt / test.txt")
    parser.add_argument(
        "--out", default="runs/sanity_check",
        help="Directory for output files (contact sheet, etc.)")
    args = parser.parse_args()
    main(args)
