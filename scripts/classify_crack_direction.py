#!/usr/bin/env python3
"""
Classify every image in this crack-segmentation dataset by crack direction
(Horizontal / Vertical / Diagonal / Mixed), using the existing ground-truth
COCO instance-segmentation polygons as the source of truth.

Why derive it from the segmentation masks
------------------------------------------
Many crack images in this dataset are very low-contrast (fine hairline
cracks on noisy asphalt/concrete texture), so a manual/visual pass is
unreliable and not reproducible. The COCO segmentation polygons already
mark exactly which pixels are crack, so we can compute each crack's
dominant orientation directly and deterministically from that ground
truth, then spot-check the result visually (see scripts/README below).

Method (per-fragment, not pooled)
----------------------------------
A single crack is often split across several polygon fragments/annotations
(e.g. the train split has 607 images but 1296 annotations) -- either
because the crack itself bends/branches, or simply because the annotator
broke a long or interrupted crack into several disconnected polygons that
individually still run in a consistent direction.

The first version of this script pooled every vertex from every
annotation belonging to an image into one point cloud and ran a single
PCA over it. That conflates two different things: the orientation of each
crack fragment, and the *spatial layout of the fragments relative to each
other*. Concretely, three separate near-vertical hairline cracks
positioned at different x-offsets across an image (bboxes ~12px wide by
110-220px tall each -- unambiguously vertical, aspect ratio up to 1:18)
got pooled into one cloud whose centroid-to-centroid spread was wider
than it was tall, and the pooled PCA reported that as "Horizontal" --
exactly backwards from what every individual fragment actually looks
like. The same pooling could also turn a bent/L-shaped crack (one long
horizontal fragment plus one long vertical fragment meeting at a corner)
into a fictitious "Diagonal" by blending the two.

This version fixes that by computing each annotation's own PCA
independently (its own principal angle and elongation ratio), then
combining fragments with circular statistics instead of pooling points:

1. For every annotation belonging to an image, run PCA on that
   annotation's own polygon vertices to get a principal angle theta_i in
   [0, 180) and an elongation ratio (how line-like that one fragment is).
2. Weight each fragment by its bounding-box diagonal, a proxy for its
   physical length -- a long fragment should influence the crack's
   overall orientation more than a short one.
3. Combine the fragment angles with a weighted circular mean in
   "doubled-angle" space (standard technique for averaging orientations
   that wrap at 180 degrees: map theta_i -> 2*theta_i, average as unit
   vectors, halve the result). The resultant vector's normalized length
   R in [0, 1] is the circular concentration: R close to 1 means the
   fragments agree closely on direction; R close to 0 means they point in
   substantially different directions.
4. If R < 0.5, the fragments disagree by 60 degrees or more (for two
   equal-weight fragments, R = cos(angle_between)/1, so R = 0.5 <=>
   a 60 degree split) -- the crack bends or branches enough that it has
   no single dominant direction, so it's classified "Mixed". This is the
   same threshold philosophy as circular-statistics convention (a mean
   resultant length below 0.5 is considered too dispersed to have a
   well-defined mean direction).
5. Otherwise the combined angle from the resultant vector is classified
   into Horizontal / Vertical / Diagonal using the same angle bins as
   before. The old "elongation ratio < 1.8" trigger for whole-crack blob
   / map / alligator cracking is still applied (computed over all pooled
   points, since a genuinely isotropic blob has no meaningful per-fragment
   layout to protect against) as a second, independent path to "Mixed".

For a single-annotation image this reduces to exactly the original
pooled-PCA computation (there's only one fragment, so there's nothing to
pool across).

Classification rule (unchanged angle bins)
-------------------------------------------
  [0, 22.5) or [157.5, 180)  -> "Horizontal"
  [67.5, 112.5)              -> "Vertical"
  otherwise                   -> "Diagonal"
      (tagged diagonal_type "\\" if the crack runs top-left to
       bottom-right in image pixel space, "/" if bottom-left to
       top-right)

One image in train/ (CRACK500_20160222_165218_641_721...) carries zero
annotations despite actually containing a crack (no polygon at all), so
PCA cannot be computed. It was inspected visually and hand-labeled (see
MANUAL_OVERRIDES below).

Every other zero-annotation image is a genuine crack-free negative (see
scripts/generate_negative_images.py) -- those get direction "None" and
are excluded from _direction_labels.csv (there's no direction to classify
when there's no crack), but are still tagged in the COCO json for
clarity. Every image with at least one annotation gets its label from the
deterministic rule above.

Outputs (written next to each split's _annotations.coco.json)
---------------------------------------------------------------
  <split>/_annotations.coco.json   - unchanged detection/segmentation
                                      content, plus a "direction",
                                      "direction_diagonal_type",
                                      "direction_angle_deg",
                                      "direction_elongation_ratio",
                                      "direction_fragment_agreement" and
                                      "direction_source" field added to
                                      each entry in "images".
  <split>/_direction_labels.csv    - flat file_name -> direction table,
                                      convenient for a classification
                                      dataloader that doesn't need COCO.
"""
import csv
import json
import math
import os

SPLITS = ["train", "valid", "test"]
MIXED_ELONGATION_THRESHOLD = 1.8  # pooled eigenvalue ratio below this => no dominant axis (blob/map/alligator cracking)
MIXED_AGREEMENT_THRESHOLD = 0.5   # circular concentration below this => fragments disagree by >=60 deg (bent/branching crack)

# file_name -> (direction, diagonal_type, note), for images with no
# ground-truth polygon to run PCA on. Determined by direct visual
# inspection of the image.
MANUAL_OVERRIDES = {
    "CRACK500_20160222_165218_641_721_jpg.rf.RAlbl8Jrf3mXEB8YGlQf.jpg": (
        "Diagonal", "/",
        "no COCO annotation present for this image; hand-labeled by "
        "visual inspection (thin crack running bottom-left to top-right)",
    ),
}


def annotation_points(ann):
    pts = []
    for seg in ann.get("segmentation", []):
        xs = seg[0::2]
        ys = seg[1::2]
        pts.extend(zip(xs, ys))
    return pts


def pca_direction(points):
    """Principal-axis angle (degrees, mod 180) and elongation ratio of a point cloud."""
    n = len(points)
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = syy = sxy = 0.0
    for x, y in points:
        dx, dy = x - mx, y - my
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    sxx /= n
    syy /= n
    sxy /= n

    tr = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = math.sqrt(max(tr * tr / 4 - det, 0.0))
    lam1 = tr / 2 + disc
    lam2 = tr / 2 - disc

    if abs(sxy) > 1e-9:
        vx, vy = lam1 - syy, sxy
    elif sxx >= syy:
        vx, vy = 1.0, 0.0
    else:
        vx, vy = 0.0, 1.0
    norm = math.hypot(vx, vy) or 1.0
    vx, vy = vx / norm, vy / norm

    angle = math.degrees(math.atan2(vy, vx)) % 180.0
    ratio = lam1 / max(lam2, 1e-9)
    return angle, ratio, vx, vy


def classify_angle(angle):
    if angle < 22.5 or angle >= 157.5:
        return "Horizontal", None
    if 67.5 <= angle < 112.5:
        return "Vertical", None
    vx, vy = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    sub = "\\" if vx * vy > 0 else "/"
    return "Diagonal", sub


def combine_fragments(anns):
    """
    Returns (angle_deg, elongation_ratio, agreement, label, sub) for an
    image with one or more crack annotations, using per-fragment PCA
    combined by weighted circular mean (see module docstring).
    """
    all_pts = []
    fragments = []  # (angle, weight)
    for ann in anns:
        pts = annotation_points(ann)
        if len(pts) < 2:
            continue
        all_pts.extend(pts)
        angle, _ratio, _vx, _vy = pca_direction(pts)
        bw, bh = ann["bbox"][2], ann["bbox"][3]
        weight = math.hypot(bw, bh) or 1.0
        fragments.append((angle, weight))

    pooled_angle, pooled_ratio, _vx, _vy = pca_direction(all_pts)

    if len(fragments) <= 1:
        # Nothing to combine across -- identical to the pooled computation.
        if pooled_ratio < MIXED_ELONGATION_THRESHOLD:
            return pooled_angle, pooled_ratio, 1.0, "Mixed", None
        label, sub = classify_angle(pooled_angle)
        return pooled_angle, pooled_ratio, 1.0, label, sub

    rx = sum(w * math.cos(math.radians(2 * a)) for a, w in fragments)
    ry = sum(w * math.sin(math.radians(2 * a)) for a, w in fragments)
    total_w = sum(w for _a, w in fragments)
    agreement = math.hypot(rx, ry) / total_w

    if agreement < MIXED_AGREEMENT_THRESHOLD or pooled_ratio < MIXED_ELONGATION_THRESHOLD:
        return pooled_angle, pooled_ratio, agreement, "Mixed", None

    combined_angle = math.degrees(math.atan2(ry, rx)) / 2.0 % 180.0
    label, sub = classify_angle(combined_angle)
    return combined_angle, pooled_ratio, agreement, label, sub


def process_split(split):
    path = os.path.join(split, "_annotations.coco.json")
    with open(path) as f:
        coco = json.load(f)

    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    rows = []
    for img in coco["images"]:
        override = MANUAL_OVERRIDES.get(img["file_name"])
        anns = anns_by_image.get(img["id"], [])
        n_ann = len(anns)
        if override is not None:
            label, sub, note = override
            angle = ratio = agreement = None
            source = "manual"
        elif n_ann == 0:
            # genuine crack-free negative image (see generate_negative_images.py) --
            # no crack, so no direction to compute or classify.
            label, sub, angle, ratio, agreement = "None", None, None, None, None
            note = "no crack present in this image (background/negative sample)"
            source = "negative"
        else:
            angle, ratio, agreement, label, sub = combine_fragments(anns)
            note = None
            source = "pca"

        img["direction"] = label
        img["direction_diagonal_type"] = sub
        img["direction_angle_deg"] = round(angle, 2) if angle is not None else None
        img["direction_elongation_ratio"] = round(ratio, 3) if ratio is not None else None
        img["direction_fragment_agreement"] = round(agreement, 3) if agreement is not None else None
        img["direction_source"] = source

        if source == "negative":
            continue  # excluded from _direction_labels.csv -- there's no direction to classify

        rows.append({
            "file_name": img["file_name"],
            "image_id": img["id"],
            "direction": label,
            "diagonal_type": sub or "",
            "angle_deg": img["direction_angle_deg"],
            "elongation_ratio": img["direction_elongation_ratio"],
            "fragment_agreement": img["direction_fragment_agreement"],
            "num_annotations": n_ann,
            "source": source,
            "note": note or "",
        })

    with open(path, "w") as f:
        json.dump(coco, f)

    csv_path = os.path.join(split, "_direction_labels.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return rows


def main():
    from collections import Counter
    for split in SPLITS:
        rows = process_split(split)
        c = Counter(r["direction"] for r in rows)
        print(split, dict(c), "total", len(rows))


if __name__ == "__main__":
    main()
