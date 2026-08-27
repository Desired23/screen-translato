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

from ocr_engine import OCREngine, merge_paragraph_blocks, merge_same_line_blocks
from translator import TranslationEngine


def _safe(text: str, limit: int = 120) -> str:
    snippet = str(text or "").replace("\n", " ").strip()[:limit]
    try:
        return snippet.encode("unicode_escape").decode("ascii", errors="ignore")
    except Exception:
        return repr(snippet)


def _upscale(image: np.ndarray, scale: float) -> np.ndarray:
    pil = Image.fromarray(image)
    out = pil.resize((int(pil.width * scale), int(pil.height * scale)), Image.Resampling.LANCZOS)
    return np.array(out)


def _grayscale_contrast(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32)
    if arr.ndim == 3:
        gray = arr[..., :3].mean(axis=2)
    else:
        gray = arr
    p5 = np.percentile(gray, 5)
    p95 = np.percentile(gray, 95)
    if p95 - p5 > 1:
        gray = (gray - p5) * (255.0 / (p95 - p5))
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=2)
    return rgb


def _threshold(image: np.ndarray) -> np.ndarray:
    gray = _grayscale_contrast(image)[..., 0]
    thr = int(np.clip(np.percentile(gray, 72), 80, 220))
    binary = np.where(gray < thr, 0, 255).astype(np.uint8)
    return np.stack([binary, binary, binary], axis=2)


def _sharpen(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.int16)
    center = arr * 5
    up = np.roll(arr, -1, axis=0)
    down = np.roll(arr, 1, axis=0)
    left = np.roll(arr, -1, axis=1)
    right = np.roll(arr, 1, axis=1)
    sharp = center - up - down - left - right
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    return sharp


def _variants(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    gray = _grayscale_contrast(image)
    return [
        ("raw", image),
        ("upscale_1_5x", _upscale(image, 1.5)),
        ("upscale_2x", _upscale(image, 2.0)),
        ("gray_contrast", gray),
        ("gray_upscale_2x", _upscale(gray, 2.0)),
        ("threshold", _threshold(image)),
        ("threshold_upscale_2x", _upscale(_threshold(image), 2.0)),
        ("sharpen", _sharpen(image)),
        ("sharpen_upscale_2x", _upscale(_sharpen(image), 2.0)),
    ]


def _is_joined_token(text: str) -> bool:
    token = str(text or "").strip()
    if " " in token:
        return False
    letters = [c for c in token if c.isalpha()]
    if len(letters) < 6:
        return False
    upper_ratio = sum(1 for c in letters if c.isupper()) / max(len(letters), 1)
    return upper_ratio >= 0.8


def _quality_score(blocks: list) -> float:
    if not blocks:
        return -1.0
    avg_conf = sum(getattr(b, "confidence", 0.0) for b in blocks) / len(blocks)
    joined = sum(1 for b in blocks if _is_joined_token(getattr(b, "text", "")))
    spaced = sum(1 for b in blocks if " " in getattr(b, "text", "").strip())
    return (avg_conf * 1.2) + (len(blocks) * 0.015) + (spaced * 0.02) - (joined * 0.05)


def _print_blocks(title: str, blocks: list) -> None:
    print(f"[DBG] {title} count={len(blocks)}", flush=True)
    for i, block in enumerate(blocks[:10]):
        print(
            f"[DBG]   [{i}] text={_safe(block.text)} conf={getattr(block, 'confidence', 0.0):.3f} "
            f"bbox=({block.x},{block.y},{block.width},{block.height})",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description="Debug OCR/translation pipeline on a capture image")
    parser.add_argument("--image", required=True, help="Path to capture image")
    parser.add_argument("--source-language", default="ja", help="Source language")
    parser.add_argument("--target-language", default="vi", help="Target language")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"IMAGE_NOT_FOUND: {image_path}")

    img = np.array(Image.open(image_path).convert("RGB"))
    print(f"[DBG] image={image_path}", flush=True)
    print(f"[DBG] size={img.shape[1]}x{img.shape[0]}", flush=True)
    print(f"[DBG] preload onnxruntime={ort.__version__} rapidocr={RapidOCR.__name__}", flush=True)

    cfg = {
        "primary_backend": "rapidocr",
        "fallback_backend": "winrt",
        "winrt_enabled": True,
        "easyocr_enabled": False,
        "safe_boot_native": False,
        "allow_winrt_mismatch_fallback": True,
        "confidence_thresh": 0.6,
        "paragraph_merge_enabled": True,
        "auto_document_line_mode_enabled": True,
        "auto_ui_line_mode_enabled": True,
        "ocr_dark_ui_retry_enabled": True,
    }

    translator = TranslationEngine(
        target_language=args.target_language,
        source_language=args.source_language,
        translation_backend="google",
        fallback_to_google=True,
        auto_source_routing_enabled=True,
        language_detector_backend="langid",
        context_refine_enabled=False,
        manga_mode=False,
        game_term_guard_enabled=False,
        game_post_edit_enabled=False,
        auto_document_guard=False,
    )

    best_variant = None
    best_score = -1.0

    for name, variant in _variants(img):
        print(f"\n[DBG] ===== variant={name} =====", flush=True)
        ocr = OCREngine(source_language=args.source_language, config=cfg)
        raw_blocks = ocr.detect(variant)
        _print_blocks("raw_blocks", raw_blocks)

        line_blocks = merge_same_line_blocks(raw_blocks)
        _print_blocks("line_blocks", line_blocks)

        para_blocks = merge_paragraph_blocks(line_blocks)
        _print_blocks("paragraph_blocks", para_blocks)

        score = _quality_score(raw_blocks)
        print(f"[DBG] quality_score={score:.3f}", flush=True)

        chosen_blocks = para_blocks or line_blocks or raw_blocks
        if chosen_blocks:
            translations = translator.translate_batch([b.text for b in chosen_blocks])
        else:
            translations = []
        for i, out in enumerate(translations[:10]):
            print(f"[DBG]   translated[{i}]={_safe(out)}", flush=True)

        if score > best_score:
            best_score = score
            best_variant = (name, raw_blocks, line_blocks, para_blocks, translations)

    if best_variant is not None:
        name, raw_blocks, line_blocks, para_blocks, translations = best_variant
        print(f"\n[DBG] ===== best_variant={name} score={best_score:.3f} =====", flush=True)
        _print_blocks("best_raw_blocks", raw_blocks)
        _print_blocks("best_line_blocks", line_blocks)
        _print_blocks("best_paragraph_blocks", para_blocks)
        for i, out in enumerate(translations[:10]):
            print(f"[DBG]   best_translated[{i}]={_safe(out)}", flush=True)


if __name__ == "__main__":
    main()
