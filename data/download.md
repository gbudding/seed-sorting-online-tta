# Data Downloads

## GrainSet Wheat (full dataset, used for streaming TTA evaluation)
- **URL:** https://doi.org/10.6084/m9.figshare.22992317.v2
- **Size:** ~20 GB
- **Contents:** 200,000 wheat kernel images + wheat.xml annotation
  file (download separately from same page, 53 MB)
- **Extract to:** `data/wheat/`
- **Citation:** Fan et al. (2023) https://doi.org/10.1038/s41597-023-02660-8

## GrainSet-tiny (used for the reused ResNet-50 checkpoint)
- **URL:** https://figshare.com/articles/figure/GrainSet-tiny_zip/22989029/1?file=40761737
- **Size:** ~200 MB
- **Contents:** 2,000 single-kernel wheat, maize, sorghum and rice
  images across eight damage categories, plus XML annotation files
- **Extract to:** `data/tiny_data/`
- **Citation:** Fan et al. (2023) https://doi.org/10.1038/s41597-023-02660-8
- **Note:** Only required if the ResNet-50 checkpoint is being
  retrained from scratch.  The streaming TTA experiments in this
  study reuse the checkpoint from the prior study (Budding, 2026),
  which was trained on GrainSet-tiny; if that checkpoint is reused
  unchanged, only the full wheat dataset is needed for these
  experiments.

## Notes
- All datasets are released under the CC BY 4.0 licence
- The `&` character in folder names (e.g. `1_F&S`) may cause issues
  with some upload tools (e.g. Kaggle). Rename affected folders if needed.
- After extracting, run the data pipeline from the project root:
  `python src/data/prepare_data.py --xml data/wheat/wheat.xml
  --root data/wheat --out runs/datalist/wheat`
