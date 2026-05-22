"""
prepare_data.py
---------------
Top-level data preparation script.

Works for any GrainSet species (full wheat, GrainSet-tiny, etc.) — the
logic only depends on the XML annotation format and the train/<category>/
folder layout, both of which are shared across the GrainSet datasets.

Run from the PROJECT ROOT.

Full GrainSet wheat (used by the streaming TTA experiments):
    python src/data/prepare_data.py \\
        --xml  data/wheat/wheat.xml \\
        --root data/wheat \\
        --out  runs/datalist/wheat

GrainSet-tiny (only needed if retraining the classifier):
    python src/data/prepare_data.py \\
        --xml  data/tiny_data/wheat_tiny.xml \\
        --root data/tiny_data/wheat \\
        --out  runs/datalist/wheat_tiny
"""

import argparse
import os
import sys

# --- Make all src.data.* imports work regardless of where script is run from ---
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # = src/
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
# -------------------------------------------------------------------------------

import torch
from torch.utils.data import DataLoader

from data.parse_annotations import (
    parse_xml, parse_from_folders, class_distribution
)
from data.split   import create_splits, read_datalist
from data.dataset import GrainSetDataset, eval_transforms, get_class_weights


def main(args):
    print("=" * 60)

    # ---- 1. Parse annotations ------------------------------------
    if args.use_folders:
        print(f"Parsing from folder structure: {args.root}")
        samples = parse_from_folders(args.root)
    else:
        if not args.xml:
            print("ERROR: provide --xml or use --use-folders")
            sys.exit(1)
        print(f"Parsing XML : {args.xml}")
        print(f"Image root  : {args.root}")
        samples = parse_xml(args.xml, args.root)

    if len(samples) == 0:
        print("ERROR: No samples found. Check your paths.")
        sys.exit(1)

    dist = class_distribution(samples)
    print(f"\nFound {dist['total']:,} samples")
    print(f"  Good (NOR) : {dist['good']:,}  ({dist['pct_good']:.1f}%)")
    print(f"  Bad (DU+IM): {dist['bad']:,}  ({dist['pct_bad']:.1f}%)")

    # ---- 2. Create splits ----------------------------------------
    print(f"\nCreating splits -> {args.out}")
    os.makedirs(args.out, exist_ok=True)
    create_splits(samples, out_dir=args.out, seed=args.seed)

    # ---- 3. DataLoader smoke-test --------------------------------
    print("\nRunning DataLoader smoke-test on test split ...")
    test_path = os.path.join(args.out, "test.txt")
    test_ds   = GrainSetDataset(test_path, transform=eval_transforms())

    n_good, n_bad = test_ds.class_counts()
    print(f"  Test: {len(test_ds):,} samples  (good={n_good}  bad={n_bad})")

    weights = get_class_weights(test_ds)
    print(f"  Class weights: bad={weights[0]:.3f}  good={weights[1]:.3f}")

    loader = DataLoader(test_ds, batch_size=min(8, len(test_ds)),
                        shuffle=False, num_workers=0)
    imgs, labels = next(iter(loader))
    print(f"  Batch shape : {imgs.shape}  dtype={imgs.dtype}")
    print(f"  Labels      : {labels.tolist()}")
    print(f"  Pixel range : [{imgs.min():.2f}, {imgs.max():.2f}]")

    print("\nAll checks passed ✓")
    print("=" * 60)
    print("Next step: train the model (only needed if not reusing the prior checkpoint)")
    # Derive a reasonable checkpoint dir name from the --out argument so
    # the suggestion adapts to whichever dataset was prepared.
    ckpt_name = os.path.basename(os.path.normpath(args.out))
    print(f"  python src/models/train.py --datalist {args.out} "
          f"--out runs/checkpoints/{ckpt_name}_resnet50 --device cpu")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare GrainSet data pipeline.")
    parser.add_argument("--xml",  default=None,
                        help="Path to species XML "
                             "(e.g. data/wheat/wheat.xml or "
                             "data/tiny_data/wheat_tiny.xml)")
    parser.add_argument("--root", required=True,
                        help="Directory containing train/ and test/ folders")
    parser.add_argument("--out",  required=True,
                        help="Output directory for datalist .txt files")
    parser.add_argument("--use-folders", action="store_true",
                        help="Derive labels from folder names, skip XML")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
