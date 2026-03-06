"""
bench_rapid.py - Benchmark RapidOCR vs EasyOCR.
"""
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def make_test_image():
    img = Image.new("RGB", (600, 300), color=(255, 255, 255))
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
            font = ImageFont.truetype("malgun.ttf", 22)
        except Exception:
            font = ImageFont.load_default()
        draw.text((20, y), line, fill=(0, 0, 0), font=font)
        y += 60
    return np.array(img)


def bench(label, fn, n=3):
    times = []
    result = None
    for i in range(n):
        t0 = time.perf_counter()
        result = fn()
        ms = (time.perf_counter() - t0) * 1000
        times.append(ms)
    avg = sum(times) / len(times)
    mn = min(times)
    print(f"  {label}: avg={avg:.0f}ms  min={mn:.0f}ms")
    return result, avg


def main():
    print("=" * 50)
    print("RapidOCR Benchmark")
    print("=" * 50)

    image = make_test_image()

    # ── RapidOCR ──────────────────────────────────────────
    print("\n[RapidOCR]")
    try:
        from rapidocr_onnxruntime import RapidOCR
        t0 = time.perf_counter()
        rapid = RapidOCR()
        print(f"  Load: {(time.perf_counter()-t0)*1000:.0f} ms")

        # Warm up
        rapid(image)
        print("  Warm up done.")

        result, avg = bench("inference (3 runs)", lambda: rapid(image))
        texts = []
        if result[0]:
            for box, text, conf in result[0]:
                try:
                    cval = float(conf)
                except Exception:
                    cval = conf
                texts.append(f"'{text}' ({cval})")
        print(f"  Detected: {texts[:3]}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  RapidOCR error: {e}")

    # ── EasyOCR (reference) ───────────────────────────────
    print("\n[EasyOCR reference (1 run)]")
    try:
        import easyocr
        reader = easyocr.Reader(["ko"], gpu=False, verbose=False)
        t0 = time.perf_counter()
        r = reader.readtext(image)
        print(f"  1 run: {(time.perf_counter()-t0)*1000:.0f} ms  → {len(r)} blocks")
    except Exception as e:
        print(f"  EasyOCR error: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
