# Crack Direction Labels

In addition to the instance-segmentation ground truth (COCO polygons in
`train/valid/test/_annotations.coco.json`), every image in this dataset has
now been labeled with a **crack direction** class for use as a
classification task:

- `Horizontal`
- `Vertical`
- `Diagonal` (with a `diagonal_type` of `/` or `\`)
- `Mixed` (branching / map / alligator-style cracking with no single
  dominant direction)

## How the labels were produced

Rather than guessing from a visual pass (many of these crack images are
very low-contrast hairline cracks on noisy asphalt/concrete texture, which
makes manual labeling unreliable), each label is computed deterministically
from the existing ground-truth segmentation polygons:

1. For each image, every segmentation polygon vertex from every annotation
   belonging to it is pooled into one point cloud (a single crack is often
   split across several polygon fragments/annotations).
2. PCA (eigen-decomposition of the point covariance matrix) gives a
   principal-axis angle (0°/180° = horizontal, 90° = vertical) and an
   elongation ratio (`largest eigenvalue / smallest eigenvalue`) describing
   how line-like vs. blob/network-like the crack's spatial spread is.
3. Classification rule:
   - `elongation_ratio < 1.8` → `Mixed`
   - else, by principal angle: `[0°,22.5°) ∪ [157.5°,180°)` → `Horizontal`,
     `[67.5°,112.5°)` → `Vertical`, otherwise → `Diagonal`.

This rule was validated by overlaying the computed principal axis on top of
a random sample of images per class and confirming visually that the line
tracks the crack's actual orientation, and by checking that the
lowest-elongation-ratio images in the dataset are genuinely branching/blob
cracks rather than simple curved single-direction ones.

One image (`CRACK500_20160222_165218_641_721_jpg.rf.RAlbl8Jrf3mXEB8YGlQf.jpg`,
in `train/`) has no COCO annotation at all, so PCA could not be computed for
it; it was hand-labeled by direct visual inspection instead.

See `scripts/classify_crack_direction.py` for the full implementation — run
it from the repo root (`python3 scripts/classify_crack_direction.py`) to
regenerate the labels if the segmentation annotations ever change.

## Where the labels live

- `<split>/_annotations.coco.json`: each entry in `images[]` gained
  `direction`, `direction_diagonal_type`, `direction_angle_deg`,
  `direction_elongation_ratio`, and `direction_source` (`"pca"` or
  `"manual"`) fields. Nothing else in the COCO file was changed, so
  existing segmentation training pipelines keep working unmodified.
- `<split>/_direction_labels.csv`: a flat `file_name → direction` table
  (plus the angle/ratio/source metadata) for a classification dataloader
  that doesn't need to parse COCO.

## Class distribution

| Split | Horizontal | Vertical | Diagonal | Mixed | Total |
|---|---|---|---|---|---|
| train | 202 | 188 | 161 | 56 | 607 |
| valid | 54 | 60 | 49 | 11 | 174 |
| test | 34 | 24 | 20 | 11 | 89 |
| **all** | **290** | **272** | **230** | **78** | **870** |
