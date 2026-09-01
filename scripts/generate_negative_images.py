#!/usr/bin/env python3
"""
Generate genuine crack-free "negative" images from this dataset, so that
crack-vs-no-crack detection can be evaluated with a real confusion matrix
(TP/FP/FN/TN) instead of a near-degenerate one.

Method
------
For every image that already has crack annotations, compute the union
bounding box of all its crack polygons, then look at the four rectangular
strips of the image lying outside that bbox (left / right / top / bottom).
For each strip that is at least MIN_CROP x MIN_CROP, a random square
sub-region can be cropped from it (size clamped to [MIN_CROP, MAX_CROP]) --
this is a region of the *same* image (same camera, lighting, material
texture) guaranteed not to overlap any annotated crack pixel, so it's a
genuine, real, in-domain negative example, not a synthetic/external one.
Up to `max_crops_per_image` non-overlapping crops can be taken from a
single source image when it has room for more than one.

Per-split behavior (see SPLIT_CONFIG)
--------------------------------------
train/valid: left completely untouched -- these already have their
negatives from an earlier run (~15% of split size, one crop per image)
and this script does not regenerate or add to them.

test: regenerated from a clean slate every run (any previously-generated
negatives are removed first, so the result is reproducible from one
deterministic pass rather than accumulating across reruns), targeting
*parity* with the split's cracked-image count -- i.e. as close to a
50/50 crack vs. crack-free test set as real, non-overlapping crops allow.
Single-crop candidates (one per source image) are exhausted before any
image contributes a second crop, to keep the negatives as diverse
(different source photos) as possible. The achieved count is printed --
exact parity is not always reachable from real crops without relaxing
crop-size/quality constraints, and this script does not do that silently.
"""
import json
import os
import random

from PIL import Image

MIN_CROP = 80
MAX_CROP = 180
SEED = 0

SPLIT_CONFIG = {
    "test": {"target_mode": "parity", "max_crops_per_image": 2, "regenerate": True},
}


def union_bbox(anns):
    x0 = min(a["bbox"][0] for a in anns)
    y0 = min(a["bbox"][1] for a in anns)
    x1 = max(a["bbox"][0] + a["bbox"][2] for a in anns)
    y1 = max(a["bbox"][1] + a["bbox"][3] for a in anns)
    return x0, y0, x1, y1


def free_strips(w, h, bbox):
    """All rectangular strips outside the crack bbox that are >= MIN_CROP in
    both dimensions, largest-area first."""
    x0, y0, x1, y1 = bbox
    candidates = [
        (0, 0, x0, h),
        (x1, 0, w - x1, h),
        (0, 0, w, y0),
        (0, y1, w, h - y1),
    ]
    feasible = [c for c in candidates if c[2] >= MIN_CROP and c[3] >= MIN_CROP]
    feasible.sort(key=lambda c: c[2] * c[3], reverse=True)
    return feasible


def boxes_overlap(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def crop_box_from_strip(sx, sy, sw, sh, rng):
    sx, sy, sw, sh = int(sx), int(sy), int(sw), int(sh)
    crop_size = max(MIN_CROP, min(MAX_CROP, sw, sh))
    max_x = sx + sw - crop_size
    max_y = sy + sh - crop_size
    cx = rng.randint(sx, max_x) if max_x > sx else sx
    cy = rng.randint(sy, max_y) if max_y > sy else sy
    return (cx, cy, cx + crop_size, cy + crop_size)


def clear_existing_negatives(split, coco):
    """Remove any previously-generated negative-image entries (and their files)
    for this split, so regeneration starts from a clean, reproducible slate."""
    kept, removed = [], 0
    for im in coco["images"]:
        if im.get("direction") == "None":
            path = os.path.join(split, im["file_name"])
            if os.path.exists(path):
                os.remove(path)
            removed += 1
            continue
        kept.append(im)
    coco["images"] = kept
    return removed


def build_candidate_crops(cracked_images, anns_by_image, max_crops_per_image, rng):
    """One list of (image, box) per crop 'slot' -- all first-crops (one per
    eligible image) before any second-crops, so single-image diversity is
    exhausted before any image contributes twice."""
    slots = [[] for _ in range(max_crops_per_image)]
    for im in cracked_images:
        w, h = im["width"], im["height"]
        strips = free_strips(w, h, union_bbox(anns_by_image[im["id"]]))
        taken = []
        for strip in strips:
            if len(taken) >= max_crops_per_image:
                break
            box = crop_box_from_strip(*strip, rng)
            if any(boxes_overlap(box, t) for t in taken):
                continue
            taken.append(box)
            slots[len(taken) - 1].append((im, box))

    ordered = []
    for slot in slots:
        rng.shuffle(slot)
        ordered.extend(slot)
    return ordered


def save_negative(split, im, box, next_id, rng):
    cx0, cy0, cx1, cy1 = box
    src_path = os.path.join(split, im["file_name"])
    stem, ext = os.path.splitext(im["file_name"])
    neg_name = f"{stem}_negcrop{ext}"
    dst_path = os.path.join(split, neg_name)
    n = 2
    while os.path.exists(dst_path):
        neg_name = f"{stem}_negcrop{n}{ext}"
        dst_path = os.path.join(split, neg_name)
        n += 1

    with Image.open(src_path) as img:
        crop = img.crop(box)
        crop.save(dst_path)
        actual_w, actual_h = crop.size

    return {
        "id": next_id,
        "license": im.get("license", 1),
        "file_name": neg_name,
        "height": actual_h,
        "width": actual_w,
        "date_captured": im.get("date_captured", ""),
        "extra": {"name": neg_name, "source_image": im["file_name"],
                  "note": "cropped from a crack-free region of the source image; no crack present"},
    }


def process_split(split, rng):
    coco_path = os.path.join(split, "_annotations.coco.json")
    with open(coco_path) as f:
        coco = json.load(f)

    cfg = SPLIT_CONFIG.get(split)
    if cfg is None:
        n_negs = sum(1 for im in coco["images"] if im.get("direction") == "None")
        print(f"{split}: left untouched ({len(coco['images'])} images, {n_negs} existing negatives)")
        return

    if cfg.get("regenerate"):
        removed = clear_existing_negatives(split, coco)
        print(f"{split}: cleared {removed} previously-generated negative(s) before regenerating")

    anns_by_image = {}
    for a in coco["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)
    cracked_images = [im for im in coco["images"] if anns_by_image.get(im["id"])]

    candidates = build_candidate_crops(
        cracked_images, anns_by_image, cfg.get("max_crops_per_image", 1), rng)

    if cfg["target_mode"] == "parity":
        target = len(cracked_images)
    else:
        target = max(1, round(cfg.get("fraction", 0.15) * len(coco["images"])))
    chosen = candidates[:target]

    next_id = (max((im["id"] for im in coco["images"]), default=-1)) + 1
    new_images = []
    for im, box in chosen:
        new_images.append(save_negative(split, im, box, next_id, rng))
        next_id += 1

    coco["images"].extend(new_images)
    with open(coco_path, "w") as f:
        json.dump(coco, f)

    print(f"{split}: {len(cracked_images)} cracked images, target {target} negatives "
          f"(parity) -> achieved {len(new_images)} "
          f"({'exact parity' if len(new_images) == len(cracked_images) else 'short of exact parity -- see docstring'})")


def main():
    rng = random.Random(SEED)
    for split in ["train", "valid", "test"]:
        process_split(split, rng)


if __name__ == "__main__":
    main()
