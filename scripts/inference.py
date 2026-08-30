#!/usr/bin/env python3
"""
Field-test inference: run a trained YOLO11-seg crack model on new, unlabeled image(s)
to answer "is there a crack, and what does it look like?" -- crack vs. no-crack
detection plus per-instance segmentation masks. This mirrors the confusion-matrix /
mask-IoU evaluation already done in colab/train_crack_direction_model.ipynb (Sections
5-6), but for real field images instead of the held-out test split.

A single YOLO11-seg checkpoint (any of runs/crack_seg*/weights/best.pt from the
training notebook) does BOTH detection and segmentation jointly -- that's what a
"-seg" model is. Detection is derived the same way the notebook's Section 5 does:
an image counts as "crack detected" if the model fires at least one box above
--conf, and each fired box's paired mask is the segmentation result for that crack.

Usage
-----
    python3 scripts/inference.py \
        --weights runs/crack_seg_third/weights/best.pt \
        --source path/to/image.jpg path/to/a_folder/ \
        --output field_results/

Optional: also classify each crack-containing image's direction (Horizontal /
Vertical / Diagonal / Mixed) with a trained YOLO11-cls checkpoint:

    python3 scripts/inference.py \
        --weights runs/crack_seg_third/weights/best.pt \
        --cls-weights runs/crack_direction_cls_third/weights/best.pt \
        --source path/to/image.jpg --output field_results/

`--source` accepts individual image paths and/or directories (every .jpg/.jpeg/.png/
.bmp/.tif/.tiff directly inside a directory is processed, non-recursively). For each
image this prints a one-line verdict (CRACK DETECTED / NO CRACK DETECTED) with
per-instance confidence/bbox/mask area, saves an annotated visualization into
--output, and writes a combined detections.json summary covering every image.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ultralytics import YOLO

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def collect_images(source):
    paths = []
    for s in source:
        p = Path(s)
        if p.is_dir():
            paths.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXTS))
        elif p.is_file():
            paths.append(p)
        else:
            raise FileNotFoundError(f'--source path not found: {p}')
    if not paths:
        raise ValueError('No images found for the given --source.')
    return paths


def polygon_area(xy):
    """Shoelace formula for an (N,2) array of pixel coordinates."""
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def annotate(image_path, result, out_path):
    """Overlay each detected crack's mask + bbox + confidence on the source image."""
    img = Image.open(image_path).convert('RGB')
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if result.masks is not None:
        for poly in result.masks.xy:
            pts = [tuple(p) for p in poly]
            if len(pts) >= 3:
                draw.polygon(pts, fill=(255, 0, 0, 90), outline=(255, 0, 0, 255))

    if result.boxes is not None:
        for box, conf in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
            x0, y0, x1, y1 = box
            draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 0, 255), width=2)
            draw.text((x0 + 2, max(0, y0 - 12)), f'crack {conf:.2f}', fill=(255, 255, 0, 255))

    composed = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    composed.save(out_path)


def classify_direction(cls_model, image_path, imgsz):
    preds = cls_model.predict(source=str(image_path), imgsz=imgsz, verbose=False)
    p = preds[0]
    class_names = cls_model.names
    direction = class_names[int(p.probs.top1)]
    confidence = float(p.probs.top1conf)
    return direction, round(confidence, 4)


def run_inference(weights, cls_weights, source_paths, conf, imgsz, output_dir):
    model = YOLO(weights)
    cls_model = YOLO(cls_weights) if cls_weights else None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_summary = []
    for img_path in source_paths:
        result = model.predict(source=str(img_path), conf=conf, imgsz=imgsz, verbose=False)[0]

        n_instances = len(result.boxes) if result.boxes is not None else 0
        crack_detected = n_instances > 0

        instances = []
        if crack_detected:
            confs = result.boxes.conf.tolist()
            boxes_xyxy = result.boxes.xyxy.tolist()
            masks_xy = result.masks.xy if result.masks is not None else [None] * n_instances
            for i, (conf_i, box_i, poly_i) in enumerate(zip(confs, boxes_xyxy, masks_xy)):
                area_px = (float(polygon_area(np.asarray(poly_i)))
                           if poly_i is not None and len(poly_i) >= 3 else None)
                instances.append({
                    'instance_id': i,
                    'confidence': round(float(conf_i), 4),
                    'bbox_xyxy': [round(v, 1) for v in box_i],
                    'mask_area_px': round(area_px, 1) if area_px is not None else None,
                })

        out_img_path = output_dir / f'{img_path.stem}_annotated{img_path.suffix}'
        if crack_detected:
            annotate(img_path, result, out_img_path)
        else:
            Image.open(img_path).convert('RGB').save(out_img_path)

        direction, direction_confidence = (None, None)
        if crack_detected and cls_model is not None:
            direction, direction_confidence = classify_direction(cls_model, img_path, imgsz=224)

        verdict = 'CRACK DETECTED' if crack_detected else 'NO CRACK DETECTED'
        line = f'{img_path.name}: {verdict}'
        if crack_detected:
            line += f' -- {n_instances} instance(s)'
            if direction is not None:
                line += f' -- direction: {direction} ({direction_confidence:.2f})'
        print(line)
        for inst in instances:
            print(f"    instance {inst['instance_id']}: confidence={inst['confidence']:.2f}  "
                  f"bbox={inst['bbox_xyxy']}  mask_area_px={inst['mask_area_px']}")

        results_summary.append({
            'file_name': img_path.name,
            'crack_detected': crack_detected,
            'n_instances': n_instances,
            'instances': instances,
            'direction': direction,
            'direction_confidence': direction_confidence,
            'annotated_image': str(out_img_path),
        })

    report_path = output_dir / 'detections.json'
    with open(report_path, 'w') as f:
        json.dump(results_summary, f, indent=2)

    n_crack = sum(r['crack_detected'] for r in results_summary)
    print(f'\n{len(results_summary)} image(s) processed -- {n_crack} with crack(s) detected, '
          f'{len(results_summary) - n_crack} crack-free.')
    print(f'Annotated images + detections.json written to {output_dir}')
    return results_summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--weights', required=True,
                    help='Path to a trained YOLO11-seg best.pt checkpoint (detection + segmentation).')
    p.add_argument('--cls-weights', default=None,
                    help='Optional path to a trained YOLO11-cls best.pt checkpoint, to also report '
                         'crack direction (Horizontal/Vertical/Diagonal/Mixed) for images with a '
                         'detected crack.')
    p.add_argument('--source', required=True, nargs='+',
                    help='Image file(s) and/or directories to run inference on.')
    p.add_argument('--output', default='field_results',
                    help='Directory to write annotated images + detections.json (default: field_results).')
    p.add_argument('--conf', type=float, default=0.25,
                    help='Confidence threshold for a detection to count (default 0.25, matches the '
                         'evaluation notebook).')
    p.add_argument('--imgsz', type=int, default=640,
                    help='Inference image size for the seg model (default 640, matches training).')
    return p.parse_args()


def main():
    args = parse_args()
    source_paths = collect_images(args.source)
    run_inference(args.weights, args.cls_weights, source_paths, args.conf, args.imgsz, args.output)


if __name__ == '__main__':
    main()
