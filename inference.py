"""
Crack detection, segmentation, and direction classification -- inference module.

Wraps two trained YOLO models behind one call:
  - yolo11n-seg  -> is there a crack? where? what shape (mask)?
  - yolo11m-cls  -> which direction does the crack run? (only run when a crack is found)

Designed to be imported by a web backend and load both models ONCE at process
start (not per-request) -- see the FastAPI example at the bottom of this file.

Expected weight locations (matching the training notebook's GitHub-backup layout):
  trained_weights/segmentation/yolo11n_best.pt
  trained_weights/direction_classification/yolo11m_best.pt
Override with SEG_MODEL_PATH / CLS_MODEL_PATH env vars or constructor args if yours differ.

Install: pip install ultralytics pillow numpy
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from ultralytics import YOLO

DEFAULT_SEG_MODEL_PATH = os.environ.get(
    "SEG_MODEL_PATH", "trained_weights/segmentation/yolo11n_best.pt"
)
DEFAULT_CLS_MODEL_PATH = os.environ.get(
    "CLS_MODEL_PATH", "trained_weights/direction_classification/yolo11m_best.pt"
)

DEFAULT_CONF_THRES = 0.25
DEFAULT_IMGSZ = 640

ImageInput = Union[str, Path, bytes, Image.Image, np.ndarray]


def _load_image(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image)).convert("RGB")
    return Image.open(image).convert("RGB")


def _pil_to_base64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class CrackAnalyzer:
    """Loads both models once; call .analyze() per image as many times as you like."""

    def __init__(
        self,
        seg_model_path: str = DEFAULT_SEG_MODEL_PATH,
        cls_model_path: str = DEFAULT_CLS_MODEL_PATH,
        conf_thres: float = DEFAULT_CONF_THRES,
        imgsz: int = DEFAULT_IMGSZ,
    ):
        self.seg_model = YOLO(seg_model_path)
        self.cls_model = YOLO(cls_model_path)
        self.conf_thres = conf_thres
        self.imgsz = imgsz

    def analyze(
        self,
        image: ImageInput,
        include_annotated_image: bool = True,
    ) -> dict:
        """Run crack detection+segmentation, then (only if a crack is found) direction
        classification. Returns a JSON-serializable dict -- safe to return directly
        from a web endpoint."""
        pil_image = _load_image(image)

        seg_pred = self.seg_model.predict(
            source=np.array(pil_image),
            conf=self.conf_thres,
            imgsz=self.imgsz,
            verbose=False,
        )[0]

        cracks = []
        if seg_pred.boxes is not None:
            boxes_xyxy = seg_pred.boxes.xyxy.cpu().numpy()
            confidences = seg_pred.boxes.conf.cpu().numpy()
            mask_polygons = seg_pred.masks.xy if seg_pred.masks is not None else [None] * len(boxes_xyxy)
            for box, conf, poly in zip(boxes_xyxy, confidences, mask_polygons):
                cracks.append({
                    "confidence": float(conf),
                    "bbox": [float(v) for v in box],  # [x1, y1, x2, y2] in original image pixels
                    "mask_polygon": poly.tolist() if poly is not None else None,
                })

        has_crack = len(cracks) > 0
        result = {
            "has_crack": has_crack,
            "num_cracks": len(cracks),
            "detection_confidence": max((c["confidence"] for c in cracks), default=None),
            "cracks": cracks,
            "direction": None,
            "direction_confidence": None,
        }

        if has_crack:
            cls_pred = self.cls_model.predict(
                source=np.array(pil_image), imgsz=224, verbose=False
            )[0]
            top1 = int(cls_pred.probs.top1)
            result["direction"] = cls_pred.names[top1]
            result["direction_confidence"] = float(cls_pred.probs.top1conf)

        if include_annotated_image:
            annotated = Image.fromarray(seg_pred.plot()[:, :, ::-1])  # BGR -> RGB
            result["annotated_image_base64"] = _pil_to_base64_png(annotated)

        return result


# Lazily-created module-level singleton so a web app can just do:
#   from inference import get_analyzer
#   result = get_analyzer().analyze(image_bytes)
# without worrying about re-loading the models on every request.
_analyzer: CrackAnalyzer | None = None


def get_analyzer() -> CrackAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = CrackAnalyzer()
    return _analyzer


# --- FastAPI example -------------------------------------------------------
# from fastapi import FastAPI, File, UploadFile
# from inference import get_analyzer
#
# app = FastAPI()
#
# @app.on_event("startup")
# def load_models():
#     get_analyzer()  # load both models once, at server startup, not per-request
#
# @app.post("/analyze")
# async def analyze(file: UploadFile = File(...)):
#     image_bytes = await file.read()
#     return get_analyzer().analyze(image_bytes)
# ----------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run crack analysis on a single image.")
    parser.add_argument("image_path", help="Path to an image file")
    parser.add_argument("--seg-model", default=DEFAULT_SEG_MODEL_PATH)
    parser.add_argument("--cls-model", default=DEFAULT_CLS_MODEL_PATH)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRES)
    parser.add_argument("--save-annotated", help="Path to save the annotated image (PNG)")
    args = parser.parse_args()

    analyzer = CrackAnalyzer(seg_model_path=args.seg_model, cls_model_path=args.cls_model,
                              conf_thres=args.conf)
    output = analyzer.analyze(args.image_path, include_annotated_image=bool(args.save_annotated))

    if args.save_annotated and "annotated_image_base64" in output:
        img_bytes = base64.b64decode(output.pop("annotated_image_base64"))
        Path(args.save_annotated).write_bytes(img_bytes)
        print(f"Annotated image saved to {args.save_annotated}")

    print(json.dumps(output, indent=2))
