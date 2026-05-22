"""
smoke_test_run_drift.py
-----------------------
End-to-end smoke test: synthetic data + tiny model + run the full
experiment driver as it will run on real data.  Uses subprocess so it
exercises the CLI exactly as the user will invoke it.

Run: python tests/smoke_test_run_drift.py

This is a manual smoke test, not part of the unit-test suite, because
it writes a fake checkpoint and dataset to disk and uses subprocess.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------
# Tiny model SAVED as a checkpoint compatible with models/train.py:
# load_model expects a state_dict keyed against build_model() (ResNet-50).
# To avoid actually downloading ResNet-50 weights, we monkey-patch
# build_model to return a tiny CNN here.
# --------------------------------------------------------------------

def main():
    tmp = tempfile.mkdtemp(prefix="smoke_run_drift_")
    try:
        # ---- 1. Synthetic dataset -------------------------------------
        data_root = os.path.join(tmp, "data")
        os.makedirs(data_root, exist_ok=True)
        rng = np.random.default_rng(0)

        n_imgs = 30 * 5  # 30 batches × 5 imgs/batch = 150 images
        with open(os.path.join(data_root, "test.txt"), "w") as f:
            for i in range(n_imgs):
                label = 1 if rng.random() < 0.6 else 0
                base  = 180 if label == 1 else 100
                arr   = np.clip(base + rng.normal(0, 20, (32, 32, 3)),
                                0, 255).astype(np.uint8)
                path  = os.path.join(data_root, f"img_{i:04d}.png")
                Image.fromarray(arr).save(path)
                f.write(f"{path} {label}\n")

        # ---- 2. Tiny "ResNet-50" stand-in checkpoint ------------------
        # We ship a tiny CNN saved with the exact key structure
        # `load_model` expects.  load_model does:
        #     model = build_model(num_classes=2, pretrained=False)
        #     model.load_state_dict(ckpt["state_dict"])
        # We monkey-patch build_model in the run_drift environment to
        # return our tiny model, so we don't need to download ResNet-50.

        # The simplest path: write a small "build_model" override file
        # and inject it via PYTHONPATH so it shadows the real one.
        # Easier alternative: just use the actual ResNet-50, which torchvision
        # can build offline (it's in the torchvision install, no download
        # needed if pretrained=False).  We do that.

        # ---- 3. Build a real ResNet-50 randomly initialised checkpoint
        from torchvision import models
        m = models.resnet50(weights=None)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, 2)
        ckpt_path = os.path.join(tmp, "ckpt.pth")
        torch.save({
            "epoch":      0,
            "state_dict": m.state_dict(),
            "val_f1":     0.0,
            "args":       {},
        }, ckpt_path)

        # ---- 4. Baseline scores file --------------------------------
        bl_scores = rng.uniform(0.3, 0.9, 1000).astype(np.float32)
        bl_path   = os.path.join(tmp, "baseline_scores.npz")
        np.savez(bl_path, scores=bl_scores,
                 thresholds=np.linspace(0, 1, 100),
                 yields=np.linspace(1, 0, 100),
                 labels=np.zeros(1000, dtype=np.int64))

        # ---- 5. Invoke the runner via subprocess --------------------
        out_dir = os.path.join(tmp, "out")
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(ROOT, "src") + os.pathsep + env.get("PYTHONPATH", "")
        cmd = [
            sys.executable, "-m", "experiments.run_drift",
            "--datalist",   data_root,
            "--checkpoint", ckpt_path,
            "--baseline",   bl_path,
            "--out",        out_dir,
            "--scenarios",  "gradual_degradation",
            "--corruptions", "brightness",
            "--methods",    "no_correction", "ema_bn_m0.1",
            "--batch-size", "5",
            "--n-batches",  "30",
            "--seeds",      "42",
            "--progress-every", "10",
        ]
        # Make datalist look like a directory containing test.txt for
        # the runner; data_root contains test.txt already.
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                cwd=ROOT)
        print("--- stdout ---")
        print(result.stdout)
        print("--- stderr ---")
        print(result.stderr)
        if result.returncode != 0:
            print(f"FAILED with exit code {result.returncode}")
            sys.exit(1)

        # ---- 6. Verify outputs --------------------------------------
        for method in ["no_correction", "ema_bn_m0.1"]:
            run_path = os.path.join(out_dir, "gradual_degradation",
                                    "brightness", f"{method}__seed42")
            assert os.path.exists(run_path + ".npz"), f"missing: {run_path}.npz"
            assert os.path.exists(run_path + ".json"), f"missing: {run_path}.json"
            data = np.load(run_path + ".npz")
            assert data["yields_t05"].shape == (30,)
            assert data["scores"].shape == (30, 5)
            print(f"  {method}: yield trajectory min={data['yields_t05'].min():.3f}  "
                  f"max={data['yields_t05'].max():.3f}  "
                  f"mean wallclock={data['wallclock_s'].mean()*1000:.0f}ms/batch")

        print("\nSmoke test PASSED ✓")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
