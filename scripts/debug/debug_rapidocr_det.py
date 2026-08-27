from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR


def main():
    parser = argparse.ArgumentParser(description="Inspect raw RapidOCR text_det boxes")
    parser.add_argument("--image", required=True, help="Path to image")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"IMAGE_NOT_FOUND: {image_path}")

    img = np.array(Image.open(image_path).convert("RGB"))
    rapid = RapidOCR()
    dt_boxes, _ = rapid.text_detector(img)
    count = 0 if dt_boxes is None else len(dt_boxes)
    print(f"[DBG] image={image_path}", flush=True)
    print(f"[DBG] det_boxes={count}", flush=True)
    if dt_boxes is None:
        return
    for i, box in enumerate(dt_boxes[:30]):
        arr = np.array(box, dtype=float)
        x1 = float(arr[:, 0].min())
        y1 = float(arr[:, 1].min())
        x2 = float(arr[:, 0].max())
        y2 = float(arr[:, 1].max())
        print(
            f"[DBG] box[{i}] rect=({x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f}) pts={arr.tolist()}",
            flush=True,
        )


if __name__ == "__main__":
    main()
