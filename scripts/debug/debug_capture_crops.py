from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import onnxruntime as ort
from rapidocr_onnxruntime import RapidOCR

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ocr_engine import OCREngine


def _safe(text: str, limit: int = 120) -> str:
    snippet = str(text or "").replace("\n", " ").strip()[:limit]
    try:
        return snippet.encode("unicode_escape").decode("ascii", errors="ignore")
    except Exception:
        return repr(snippet)


def _crop_with_padding(image: np.ndarray, x: int, y: int, w: int, h: int, pad_x: int, pad_y: int) -> np.ndarray:
    h_img, w_img = image.shape[:2]
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w_img, x + w + pad_x)
    y2 = min(h_img, y + h + pad_y)
    return image[y1:y2, x1:x2, :]


def _upscale(image: np.ndarray, scale: float) -> np.ndarray:
    pil = Image.fromarray(image)
    out = pil.resize((int(pil.width * scale), int(pil.height * scale)), Image.Resampling.LANCZOS)
    return np.array(out)


def main():
    parser = argparse.ArgumentParser(description="Debug OCR on exported block crops")
    parser.add_argument("--image", required=True, help="Path to capture image")
    parser.add_argument("--source-language", default="ja", help="Source language")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"IMAGE_NOT_FOUND: {image_path}")

    img = np.array(Image.open(image_path).convert("RGB"))
    out_dir = REPO_ROOT / ".logs" / "rapidocr-debug" / "capture-crops"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DBG] image={image_path}", flush=True)
    print(f"[DBG] preload onnxruntime={ort.__version__} rapidocr={RapidOCR.__name__}", flush=True)
    print(f"[DBG] crop_dir={out_dir}", flush=True)

    cfg = {
        "primary_backend": "rapidocr",
        "fallback_backend": "none",
        "winrt_enabled": False,
        "easyocr_enabled": False,
        "safe_boot_native": False,
        "allow_winrt_mismatch_fallback": False,
        "confidence_thresh": 0.6,
    }
    ocr = OCREngine(source_language=args.source_language, config=cfg)
    blocks = ocr.detect(img)
    print(f"[DBG] initial_blocks={len(blocks)}", flush=True)

    for i, block in enumerate(blocks):
        print(
            f"[DBG] block[{i}] text={_safe(block.text)} conf={block.confidence:.3f} bbox=({block.x},{block.y},{block.width},{block.height})",
            flush=True,
        )
        base_crop = _crop_with_padding(img, block.x, block.y, block.width, block.height, 12, 12)
        variants = [
            ("base", base_crop),
            ("pad24", _crop_with_padding(img, block.x, block.y, block.width, block.height, 24, 18)),
            ("pad36", _crop_with_padding(img, block.x, block.y, block.width, block.height, 36, 24)),
            ("base_2x", _upscale(base_crop, 2.0)),
            ("pad24_2x", _upscale(_crop_with_padding(img, block.x, block.y, block.width, block.height, 24, 18), 2.0)),
            ("pad36_2x", _upscale(_crop_with_padding(img, block.x, block.y, block.width, block.height, 36, 24), 2.0)),
        ]
        for name, crop in variants:
            crop_path = out_dir / f"block{i}_{name}.png"
            Image.fromarray(crop).save(crop_path)
            crop_blocks = ocr.detect(crop)
            texts = " | ".join(_safe(b.text) for b in crop_blocks[:6]) if crop_blocks else "<none>"
            print(
                f"[DBG]   crop={name} size={crop.shape[1]}x{crop.shape[0]} blocks={len(crop_blocks)} texts={texts}",
                flush=True,
            )


if __name__ == "__main__":
    main()
