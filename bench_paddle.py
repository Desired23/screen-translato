"""
bench_paddle.py - Compare PaddleOCR vs EasyOCR speed on Korean text.
"""
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def make_test_image(width=600, height=400):
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    lines = [
        "안녕하세요. 오늘 날씨가 매우 좋습니다.",
        "이 프로그램은 화면을 실시간으로 번역합니다.",
        "Chapter 15: Advanced Settings",
        "사용자 인터페이스를 조정하세요.",
    ]
    y = 30
    for line in lines:
        try:
            font = ImageFont.truetype("malgun.ttf", 24)
        except Exception:
            font = ImageFont.load_default()
        draw.text((20, y), line, fill=(0, 0, 0), font=font)
        y += 70
    return np.array(img)


def bench(label, fn):
    t0 = time.perf_counter()
    result = fn()
    ms = (time.perf_counter() - t0) * 1000
    print(f"  {label}: {ms:.0f} ms")
    return result, ms


def main():
    print("=" * 50)
    print("PaddleOCR vs EasyOCR Speed Comparison")
    print("=" * 50)

    image = make_test_image()
    img_pil = Image.fromarray(image)

    # ── PaddleOCR ────────────────────────────────────────
    print("\n[PaddleOCR]")
    try:
        from paddleocr import PaddleOCR
        t0 = time.perf_counter()
        ocr_p = PaddleOCR(lang="korean", use_angle_cls=False, show_log=False)
        load_ms = (time.perf_counter() - t0) * 1000
        print(f"  Load model: {load_ms:.0f} ms")

        results, run_ms = bench("Inference (1st)", lambda: ocr_p.ocr(image, cls=False))
        blocks_p = []
        if results and results[0]:
            for line in results[0]:
                bbox, (text, conf) = line
                if conf > 0.3 and text.strip():
                    blocks_p.append(text)
        print(f"  → {len(blocks_p)} blocks: {blocks_p}")

        _, run2_ms = bench("Inference (2nd)", lambda: ocr_p.ocr(image, cls=False))
        paddle_total = run_ms
    except Exception as e:
        print(f"  PaddleOCR error: {e}")
        paddle_total = 99999
        run2_ms = 99999

    # ── EasyOCR ──────────────────────────────────────────
    print("\n[EasyOCR]")
    try:
        import easyocr
        t0 = time.perf_counter()
        reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
        load_ms2 = (time.perf_counter() - t0) * 1000
        print(f"  Load model: {load_ms2:.0f} ms")

        results_e, easy_ms = bench("Inference (1st)", lambda: reader.readtext(image))
        blocks_e = [t for _, t, c in results_e if c > 0.3]
        print(f"  → {len(blocks_e)} blocks: {blocks_e[:3]}")

        _, easy2_ms = bench("Inference (2nd)", lambda: reader.readtext(image))
    except Exception as e:
        print(f"  EasyOCR error: {e}")
        easy_ms = 99999
        easy2_ms = 99999

    # ── Summary ──────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"  PaddleOCR 1st: {paddle_total:.0f} ms   2nd: {run2_ms:.0f} ms")
    print(f"  EasyOCR   1st: {easy_ms:.0f} ms   2nd: {easy2_ms:.0f} ms")
    print(f"  Speedup:  {easy_ms/max(paddle_total,1):.1f}x faster (1st run)")
    print("=" * 50)


if __name__ == "__main__":
    main()
