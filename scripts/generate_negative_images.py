#!/usr/bin/env python3
"""
Generate genuine crack-free "negative" images from this dataset, so that
crack-vs-no-crack detection can be evaluated with a real confusion matrix
(TP/FP/FN/TN) instead of a near-degenerate one (this dataset currently has
essentially zero true negatives).

Method
------
For every image that already has crack annotations, compute the union
bounding box of all its crack polygons, then look at the four rectangular
strips of the image lying outside that bbox (left / right / top / bottom).
If the largest such strip is at least MIN_CROP x MIN_CROP, crop a random
square sub-region from it (size clamped to [MIN_CROP, MAX_CROP]) -- this
is a region of the *same* image (same camera, lighting, material texture)
that is guaranteed not to overlap any annotated crack pixel, so it's a
genuine, real, in-domain negative example, not a synthetic/external one.

A random subset of eligible candidates (~15% of each split's current
image count) is selected per split, cropped, and saved as new image files
alongside the originals, then added to that split's COCO json as
zero-annotation image entries (COCO's native way of representing a
background/negative image for detection training).
"""
import json
import os
import random

from PIL import Image

SPLITS = ["train", "valid", "test"]
MIN_CROP = 80
MAX_CROP = 180
NEGATIVE_FRACTION = 0.15  # ~15% of each split's existing image count
SEED = 0


def union_bbox(anns):
    x0 = min(a["bbox"][0] for a in anns)
    y0 = min(a["bbox"][1] for a in anns)
    x1 = max(a["bbox"][0] + a["bbox"][2] for a in anns)
    y1 = max(a["bbox"][1] + a["bbox"][3] for a in anns)
    return x0, y0, x1, y1


def best_free_strip(w, h, bbox):
    x0, y0, x1, y1 = bbox
    candidates = [
        ("left", 0, 0, x0, h),
        ("right", x1, 0, w - x1, h),
        ("top", 0, 0, w, y0),
        ("bottom", 0, y1, w, h - y1),
    ]
    feasible = [c for c in candidates if c[3] >= MIN_CROP and c[4] >= MIN_CROP]
    if not feasible:
        return None
    return max(feasible, key=lambda c: c[3] * c[4])


def process_split(split, rng):
    coco_path = os.path.join(split, "_annotations.coco.json")
    with open(coco_path) as f:
        coco = json.load(f)

    anns_by_image = {}
    for a in coco["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)

    candidates = []
    for im in coco["images"]:
        anns = anns_by_image.get(im["id"], [])
        if not anns:
            continue
        strip = best_free_strip(im["width"], im["height"], union_bbox(anns))
        if strip is not None:
            candidates.append((im, strip))

    target = max(1, round(NEGATIVE_FRACTION * len(coco["images"])))
    target = min(target, len(candidates))
    chosen = rng.sample(candidates, target)

    next_id = max(im["id"] for im in coco["images"]) + 1
    new_images = []
    for im, (_, sx, sy, sw, sh) in chosen:
        sx, sy, sw, sh = int(sx), int(sy), int(sw), int(sh)
        crop_size = max(MIN_CROP, min(MAX_CROP, sw, sh))
        max_x = sx + sw - crop_size
        max_y = sy + sh - crop_size
        cx = rng.randint(sx, max_x) if max_x > sx else sx
        cy = rng.randint(sy, max_y) if max_y > sy else sy

        src_path = os.path.join(split, im["file_name"])
        stem, ext = os.path.splitext(im["file_name"])
        neg_name = f"{stem}_negcrop{ext}"
        dst_path = os.path.join(split, neg_name)

        with Image.open(src_path) as img:
            crop = img.crop((cx, cy, cx + crop_size, cy + crop_size))
            crop.save(dst_path)
            actual_w, actual_h = crop.size  # ground truth for the saved file, not the requested size

        new_images.append({
            "id": next_id,
            "license": im.get("license", 1),
            "file_name": neg_name,
            "height": actual_h,
            "width": actual_w,
            "date_captured": im.get("date_captured", ""),
            "extra": {"name": neg_name, "source_image": im["file_name"],
                      "note": "cropped from a crack-free region of the source image; no crack present"},
        })
        next_id += 1

    coco["images"].extend(new_images)
    with open(coco_path, "w") as f:
        json.dump(coco, f)

    return len(coco["images"]) - len(new_images), len(new_images)


def main():
    rng = random.Random(SEED)
    for split in SPLITS:
        n_before, n_added = process_split(split, rng)
        print(f"{split}: {n_before} original images -> +{n_added} negative crops "
              f"-> {n_before + n_added} total")


if __name__ == "__main__":
    main()
