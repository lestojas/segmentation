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
from the existing ground-truth segmentation polygons — **per fragment**,
not by pooling every vertex from every annotation into one point cloud.

1. For every annotation belonging to an image, PCA (eigen-decomposition of
   that annotation's own vertex covariance matrix) gives a principal-axis
   angle (0°/180° = horizontal, 90° = vertical) for that one crack
   fragment, weighted by its bounding-box diagonal (a proxy for its
   physical length).
2. Fragment angles are combined with a weighted **circular mean** (angles
   doubled to handle the 180° wraparound, averaged as unit vectors, then
   halved back — the standard technique for averaging orientations). The
   resultant vector's normalized length ("agreement", 0–1) measures how
   closely the fragments agree: 1.0 = perfectly aligned, and for two
   equal-weight fragments, 0.5 corresponds to a 60° split between them.
3. Classification rule:
   - `fragment_agreement < 0.5` → `Mixed` (the crack's fragments disagree
     by 60°+ — a bent or branching crack with no single dominant axis)
   - `elongation_ratio < 1.8` (pooled, whole-crack) → `Mixed` (blob/map/
     alligator cracking, no dominant axis regardless of fragment count)
   - else, by the combined principal angle: `[0°,22.5°) ∪ [157.5°,180°)` →
     `Horizontal`, `[67.5°,112.5°)` → `Vertical`, otherwise → `Diagonal`.

For a single-annotation image this is identical to running PCA once over
that annotation's own points.

### Why not just pool every vertex into one PCA (v1 bug)

The first version of this script pooled all vertices from all of an
image's annotations into a single point cloud before running PCA. That
conflates two different things: **the orientation of each crack
fragment** and **the spatial layout of the fragments relative to each
other in the frame**. Two concrete, confirmed failures this caused:

- Three separate hairline cracks in one train image, each with a bbox of
  roughly 12px wide × 110–220px tall (aspect ratio up to 1:18 —
  unambiguously vertical), sat at different x-offsets across the frame.
  The pooled cloud's centroid-to-centroid spread was wider than it was
  tall, so pooled PCA reported the image as **"Horizontal"** — the exact
  opposite of what every individual crack in it looks like.
- A bent, L-shaped crack (one long horizontal fragment, one long vertical
  fragment meeting at a corner) got blended by pooled PCA into a
  fictitious **"Diagonal"**, when a human — or a per-fragment agreement
  check — would call it what it is: two roughly perpendicular segments,
  i.e. `Mixed`.

The per-fragment + circular-mean approach fixes both: each fragment's own
shape determines its own angle (unaffected by where it sits in the
frame), and fragments are only combined into one direction when they
actually agree; when they don't, the label is `Mixed` rather than an
average that no fragment actually has.

This rule was validated by overlaying the computed principal axis on top of
a random sample of images per class and confirming visually that the line
tracks each crack's actual orientation, and by checking that the
lowest-agreement and lowest-elongation-ratio images in the dataset are
genuinely bent/branching/blob cracks rather than simple curved
single-direction ones.

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

~15% of the train/valid image count was added this way as zero-annotation
COCO image entries (`<original_stem>_negcrop.jpg`), tagged
`direction: "None"` / `direction_source: "negative"` in the COCO json,
and excluded from `_direction_labels.csv` (a crack-free image has no
direction to classify). Everything else about the existing images and
annotations is untouched.

The **test** split is treated differently: it is rebalanced to exact
**parity** between crack and crack-free images (89 vs. 89) so that
crack-vs-no-crack detection can be evaluated with a meaningful confusion
matrix (TP/FP/FN/TN), rather than the near-degenerate ~87/13 split
produced by the ~15% rule used for train/valid. Reaching parity required
some source images to contribute a second, non-overlapping crack-free
crop (`<original_stem>_negcrop2.jpg`) in addition to their first. The
test split is regenerated from a clean slate on every run of
`scripts/generate_negative_images.py` (so it's reproducible from one
deterministic pass, not accumulated across reruns); train/valid are left
untouched.

| Split | Original images | Negative crops added | Total |
|---|---|---|---|
| train | 607 | 91 | 698 |
| valid | 174 | 26 | 200 |
| test | 89 | 89 (parity) | 178 |
| **all** | **870** | **206** | **1076** |

## Where the labels live

- `<split>/_annotations.coco.json`: each entry in `images[]` gained
  `direction`, `direction_diagonal_type`, `direction_angle_deg`,
  `direction_elongation_ratio`, `direction_fragment_agreement`, and
  `direction_source` (`"pca"`, `"manual"`, or `"negative"`) fields.
  Nothing else in the COCO file was changed, so existing segmentation
  training pipelines keep working unmodified — negative images simply
  carry no annotations, which is COCO's native way of representing a
  background/negative example.
- `<split>/_direction_labels.csv`: a flat `file_name → direction` table
  (plus the angle/ratio/agreement/source metadata) for a classification
  dataloader that doesn't need to parse COCO. Only crack-containing images
  appear here.

## Class distribution (crack-containing images only)

Recomputed with the per-fragment + circular-mean method described above
(see "Why not just pool every vertex into one PCA (v1 bug)"). 99 of 870
crack images (~11%) changed label versus the original pooled-PCA version
— mostly out of a spurious `Diagonal` produced by blending fragments that
don't actually agree in orientation, redistributed into `Mixed`,
`Horizontal`, or `Vertical` once each fragment's own shape is measured on
its own terms.

| Split | Horizontal | Vertical | Diagonal | Mixed | Total |
|---|---|---|---|---|---|
| train | 191 | 200 | 129 | 87 | 607 |
| valid | 51 | 64 | 38 | 21 | 174 |
| test | 31 | 25 | 16 | 17 | 89 |
| **all** | **273** | **289** | **183** | **125** | **870** |
