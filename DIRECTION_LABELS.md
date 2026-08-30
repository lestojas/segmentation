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
in `train/`) has no COCO annotation despite actually containing a crack, so
PCA could not be computed for it; it was hand-labeled by direct visual
inspection instead.

See `scripts/classify_crack_direction.py` for the full implementation — run
it from the repo root (`python3 scripts/classify_crack_direction.py`) to
regenerate the labels if the segmentation annotations ever change.

## Crack-free negative images

The dataset as originally exported had (with one exception) no genuine
crack-free images, which makes "crack vs. no-crack" detection accuracy
impossible to evaluate meaningfully (nothing to compute a true-negative
rate or specificity from). `scripts/generate_negative_images.py` fixes
this by cropping a real, in-domain negative from every image that has
enough crack-free space: it finds the largest rectangular strip of the
image lying entirely outside the union of that image's crack polygons,
and crops a random square sub-region from it (80–180px, clamped to what's
available). Because the crop comes from the same photo (same camera,
lighting, material) as a real annotated crack image, it's a genuine
negative example, not a synthetic or out-of-domain one.

~15% of each split's image count was added this way as zero-annotation
COCO image entries (`<original_stem>_negcrop.jpg`), tagged
`direction: "None"` / `direction_source: "negative"` in the COCO json,
and excluded from `_direction_labels.csv` (a crack-free image has no
direction to classify). Everything else about the existing images and
annotations is untouched.

| Split | Original images | Negative crops added | Total |
|---|---|---|---|
| train | 607 | 91 | 698 |
| valid | 174 | 26 | 200 |
| test | 89 | 13 | 102 |
| **all** | **870** | **130** | **1000** |

## Where the labels live

- `<split>/_annotations.coco.json`: each entry in `images[]` gained
  `direction`, `direction_diagonal_type`, `direction_angle_deg`,
  `direction_elongation_ratio`, and `direction_source` (`"pca"`,
  `"manual"`, or `"negative"`) fields. Nothing else in the COCO file was
  changed, so existing segmentation training pipelines keep working
  unmodified — negative images simply carry no annotations, which is
  COCO's native way of representing a background/negative example.
- `<split>/_direction_labels.csv`: a flat `file_name → direction` table
  (plus the angle/ratio/source metadata) for a classification dataloader
  that doesn't need to parse COCO. Only crack-containing images appear
  here.

## Class distribution (crack-containing images only)

| Split | Horizontal | Vertical | Diagonal | Mixed | Total |
|---|---|---|---|---|---|
| train | 202 | 188 | 161 | 56 | 607 |
| valid | 54 | 60 | 49 | 11 | 174 |
| test | 34 | 24 | 20 | 11 | 89 |
| **all** | **290** | **272** | **230** | **78** | **870** |
