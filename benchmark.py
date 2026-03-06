"""
benchmark.py - Measure translation pipeline speed with a synthetic image.
Simulates Korean text on screen, runs OCR + translation, reports timings.
"""
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Create synthetic Korean text image ──────────────────────────────────────
def make_test_image(width=600, height=400):
    """Create a white image with Korean+English text lines."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    lines = [
        "안녕하세요. 오늘 날씨가 매우 좋습니다.",
        "이 프로그램은 화면을 실시간으로 번역합니다.",
        "Chapter 15: Advanced Settings",
        "사용자 인터페이스를 조정하세요.",
        "Please enable the translation feature.",
        "번역 속도를 최적화하는 중입니다.",
    ]

    y = 30
    for line in lines:
        try:
            # Try to use a Korean-capable font
            font = ImageFont.truetype("malgun.ttf", 24)
        except Exception:
            font = ImageFont.load_default()
        draw.text((20, y), line, fill=(0, 0, 0), font=font)
        y += 50

    return np.array(img)

# ── Benchmark ────────────────────────────────────────────────────────────────
def bench(label, fn):
    t0 = time.perf_counter()
    result = fn()
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  [{label}] {elapsed:.0f} ms")
    return result, elapsed

def main():
    print("=" * 60)
    print("BENCHMARK: Screen Translator Pipeline")
    print("=" * 60)

    # Create test image
    image, _ = bench("Create test image", make_test_image)
    print()

    # ── Stage 1: OCR ────────────────────────────────────────────────────────
    print("[Stage 1] EasyOCR (Korean+English)")
    from ocr_engine import OCREngine
    ocr = OCREngine(source_language="ko")
    ocr._ensure_engine()  # pre-warm
    print("  EasyOCR model loaded.")

    blocks, ocr_ms = bench("OCR readtext", lambda: ocr.detect(image))
    print(f"  → {len(blocks)} blocks detected")
    for b in blocks:
        print(f"     '{b.text[:40]}' conf={b.confidence:.2f}")
    print()

    # ── Stage 2: Translation ─────────────────────────────────────────────────
    print("[Stage 2] Translation (Google)")
    from translator import TranslationEngine
    tr = TranslationEngine(target_language="vi", source_language="ko")

    texts = [b.text for b in blocks] if blocks else [
        "안녕하세요",
        "오늘 날씨가 매우 좋습니다",
        "이 프로그램은 화면을 실시간으로 번역합니다",
        "Chapter 15",
        "사용자 인터페이스를 조정하세요",
    ]

    translated, tr_ms = bench("Batch translate", lambda: tr.translate_batch(texts))
    print(f"  → {len(translated)} translations")
    for orig, trans in zip(texts[:3], translated[:3]):
        print(f"     '{orig[:30]}' → '{str(trans)[:40]}'")
    print()

    # ── Summary ──────────────────────────────────────────────────────────────
    total = ocr_ms + tr_ms
    print("=" * 60)
    print(f"  OCR:         {ocr_ms:>6.0f} ms")
    print(f"  Translation: {tr_ms:>6.0f} ms")
    print(f"  TOTAL:       {total:>6.0f} ms  (target: <1000ms)")
    print("=" * 60)

    if total < 1000:
        print("✅ Target met!")
    else:
        print(f"❌ {total - 1000:.0f} ms over target. Needs optimization.")

    print()
    print("[Stage 3] Second run (cache warm)")
    _, ocr2 = bench("OCR (2nd run)", lambda: ocr.detect(image))
    _, tr2 = bench("Translate (2nd run, cached)", lambda: tr.translate_batch(texts))
    print(f"  TOTAL (warm): {ocr2 + tr2:.0f} ms")

if __name__ == "__main__":
    main()
