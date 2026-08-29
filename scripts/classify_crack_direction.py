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

Method
------
For every image, all segmentation polygon vertices from every annotation
belonging to it are pooled into one point cloud (a single crack is often
split across several polygon fragments/annotations - e.g. the train split
has 607 images but 1296 annotations).

PCA (eigen-decomposition of the point covariance matrix) gives:
  - a principal-axis angle theta in [0, 180) degrees (0/180 = horizontal,
    90 = vertical): the dominant direction the crack pixels are spread
    along.
  - an elongation ratio = lambda_max / lambda_min: how line-like
    (elongated, large ratio) vs. blob/network-like (isotropic, ratio
    close to 1) the crack's spatial spread is.

Classification rule
--------------------
  elongation_ratio < MIXED_THRESHOLD (1.8)
      -> "Mixed"        (branching / map / alligator-style cracking with
                          no single dominant direction; validated visually
                          against the lowest-ratio images in the dataset)
  else, by principal angle theta:
      [0, 22.5) or [157.5, 180)  -> "Horizontal"
      [67.5, 112.5)              -> "Vertical"
      otherwise                   -> "Diagonal"
          (tagged diagonal_type "\\" if the crack runs top-left to
           bottom-right in image pixel space, "/" if bottom-left to
           top-right)

One image in train/ (CRACK500_20160222_165218_641_721...) carries zero
annotations in the source dataset (no polygon at all), so PCA cannot be
computed. It was inspected visually and hand-labeled (see
MANUAL_OVERRIDES below); every other image's label comes from the
deterministic PCA rule above.

Outputs (written next to each split's _annotations.coco.json)
---------------------------------------------------------------
  <split>/_annotations.coco.json   - unchanged detection/segmentation
                                      content, plus a "direction",
                                      "direction_diagonal_type",
                                      "direction_angle_deg",
                                      "direction_elongation_ratio" and
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
MIXED_THRESHOLD = 1.8  # eigenvalue ratio below this => no dominant axis

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


def polygon_points(annotations_by_image, image_id):
    pts = []
    for ann in annotations_by_image.get(image_id, []):
        for seg in ann.get("segmentation", []):
            xs = seg[0::2]
            ys = seg[1::2]
            pts.extend(zip(xs, ys))
    return pts


def pca_direction(points):
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


def classify(angle, ratio, vx, vy):
    if ratio < MIXED_THRESHOLD:
        return "Mixed", None
    if angle < 22.5 or angle >= 157.5:
        return "Horizontal", None
    if 67.5 <= angle < 112.5:
        return "Vertical", None
    slope_sign = vx * vy
    sub = "\\" if slope_sign > 0 else "/"
    return "Diagonal", sub


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
        if override is not None:
            label, sub, note = override
            angle = ratio = None
            source = "manual"
        else:
            pts = polygon_points(anns_by_image, img["id"])
            angle, ratio, vx, vy = pca_direction(pts)
            label, sub = classify(angle, ratio, vx, vy)
            note = None
            source = "pca"

        img["direction"] = label
        img["direction_diagonal_type"] = sub
        img["direction_angle_deg"] = round(angle, 2) if angle is not None else None
        img["direction_elongation_ratio"] = round(ratio, 3) if ratio is not None else None
        img["direction_source"] = source

        rows.append({
            "file_name": img["file_name"],
            "image_id": img["id"],
            "direction": label,
            "diagonal_type": sub or "",
            "angle_deg": img["direction_angle_deg"],
            "elongation_ratio": img["direction_elongation_ratio"],
            "num_annotations": len(anns_by_image.get(img["id"], [])),
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
