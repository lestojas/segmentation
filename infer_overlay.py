#!/usr/bin/env python3
"""
infer_overlay.py — Overlay predicted crack masks onto an input image.

This script is fully INDEPENDENT of training: it only needs a trained
YOLO-seg weights file (e.g. best.pt) and an input image. Rerun it as many
times as you like without retraining.

It runs YOLO11-seg prediction, then paints each predicted crack mask onto
the original image as a semi-transparent fill that follows the *exact
crack contour* (the pixel mask outline), plus a crisp solid outline drawn
along that same contour. No bounding boxes are drawn.

Usage
-----
    python infer_overlay.py --weights best.pt --source crack.jpg
    python infer_overlay.py --weights runs/best.pt --source img.jpg \
        --out overlay.jpg --conf 0.25 --alpha 0.5 --color 0,0,255

Install (if needed)
-------------------
    pip install ultralytics opencv-python

Notes
-----
- `--color` is given as B,G,R (OpenCV order). Default is red (0,0,255).
- `retina_masks=True` makes the predicted masks match the original image
  resolution exactly, so the overlay hugs the true crack shape instead of
  a downscaled approximation.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_color(value: str) -> tuple:
    """Parse a 'B,G,R' string into an (int, int, int) tuple."""
    parts = [int(c.strip()) for c in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("color must be 'B,G,R', e.g. 0,0,255")
    return tuple(max(0, min(255, c)) for c in parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Overlay predicted crack masks on an image.")
    p.add_argument("--weights", required=True, help="Path to trained YOLO-seg weights (e.g. best.pt).")
    p.add_argument("--source", required=True, help="Path to the input image.")
    p.add_argument("--out", default=None, help="Output path. Default: <source>_overlay.jpg next to the input.")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default 0.25).")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size (default 640).")
    p.add_argument("--alpha", type=float, default=0.5, help="Mask fill opacity 0..1 (default 0.5).")
    p.add_argument("--color", type=parse_color, default=(0, 0, 255), help="Fill/outline color as B,G,R (default red).")
    p.add_argument("--outline", type=int, default=2, help="Contour outline thickness in px; 0 disables (default 2).")
    p.add_argument("--device", default=None, help="Device, e.g. '0', 'cpu'. Default: auto.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    src = Path(args.source)
    if not src.exists():
        raise FileNotFoundError(f"Input image not found: {src}")

    out_path = Path(args.out) if args.out else src.with_name(f"{src.stem}_overlay.jpg")

    # Load the trained model (weights only — no training state needed).
    model = YOLO(args.weights)

    # retina_masks=True -> masks returned at ORIGINAL image resolution.
    results = model.predict(
        source=str(src),
        conf=args.conf,
        imgsz=args.imgsz,
        retina_masks=True,
        device=args.device,
        verbose=False,
    )
    result = results[0]

    # Original image (BGR) as loaded by the model.
    image = result.orig_img.copy()
    h, w = image.shape[:2]

    if result.masks is None or len(result.masks) == 0:
        print("No cracks detected — saving a copy of the original image.")
        cv2.imwrite(str(out_path), image)
        print(f"Saved: {out_path.resolve()}")
        return

    # Combine all instance masks into one binary union mask (single class).
    # masks.data: (N, H, W) float in [0,1]; threshold at 0.5 for a hard mask.
    masks = result.masks.data.cpu().numpy()
    union = np.any(masks > 0.5, axis=0).astype(np.uint8)

    # Safety: ensure the mask matches the image size (it should with retina_masks).
    if union.shape != (h, w):
        union = cv2.resize(union, (w, h), interpolation=cv2.INTER_NEAREST)

    # 1) Semi-transparent color fill following the exact mask shape.
    color_layer = np.zeros_like(image)
    color_layer[:] = args.color
    mask_bool = union.astype(bool)
    blended = image.copy()
    blended[mask_bool] = cv2.addWeighted(
        image, 1 - args.alpha, color_layer, args.alpha, 0
    )[mask_bool]

    # 2) Crisp solid outline along the true crack contour (not a bbox).
    if args.outline > 0:
        contours, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, args.color, args.outline)

    n = len(result.masks)
    cv2.imwrite(str(out_path), blended)
    print(f"Detected {n} crack instance(s). Saved overlay: {out_path.resolve()}")


if __name__ == "__main__":
    main()
