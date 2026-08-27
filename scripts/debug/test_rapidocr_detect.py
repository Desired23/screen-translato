from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ocr_engine import OCREngine
from translator import TranslationEngine


def _safe(text: str) -> str:
    try:
        return str(text).encode("unicode_escape").decode("ascii", errors="ignore")
    except Exception:
        return repr(text)


def main():
    parser = argparse.ArgumentParser(description="Debug RapidOCR detect on a sample image")
    parser.add_argument("--image", default=".tmp/ja_sample.png", help="Path to sample image")
    parser.add_argument("--source-language", default="ja", help="OCR source language")
    parser.add_argument("--translate", action="store_true", help="Also run translation to Vietnamese")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"IMAGE_NOT_FOUND: {image_path}")

    print(f"[DBG] test_rapidocr_detect image={image_path}", flush=True)
    img = np.array(Image.open(image_path).convert("RGB"))
    config = {
        "primary_backend": "rapidocr",
        "fallback_backend": "winrt",
        "winrt_enabled": True,
        "easyocr_enabled": False,
        "safe_boot_native": False,
        "allow_winrt_mismatch_fallback": True,
        "confidence_thresh": 0.6,
    }
    try:
        ocr = OCREngine(source_language=args.source_language, config=config)
        blocks = ocr.detect(img)
        print(
            f"[DBG] ocr backend={ocr.backend_name} fallback={ocr.fallback_name} blocks={len(blocks)}",
            flush=True,
        )
        for i, block in enumerate(blocks[:10]):
            print(
                f"[DBG] block[{i}] text={_safe(block.text)} conf={block.confidence:.3f}",
                flush=True,
            )
        if args.translate and blocks:
            tr = TranslationEngine(
                target_language="vi",
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
            out = tr.translate_batch([b.text for b in blocks])
            for i, text in enumerate(out[:10]):
                print(f"[DBG] translated[{i}]={_safe(text)}", flush=True)
    except Exception as exc:
        print(f"[DBG] detect=fail error={exc!r}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
