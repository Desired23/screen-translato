"""
bench_fast_ocr.py - Test EasyOCR optimization options.
"""
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def make_test_image(width=600, height=300):
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    lines = [
        "안녕하세요. 오늘 날씨가 매우 좋습니다.",
        "이 프로그램은 화면을 실시간으로 번역합니다.",
        "Chapter 15: Advanced Settings",
    ]
    y = 30
    for line in lines:
        try:
            font = ImageFont.truetype("malgun.ttf", 22)
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


def resize_image(img, max_dim):
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    import cv2
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def main():
    from ocr_engine import OCREngine

    print("Creating unified OCREngine (lang=ko)...")
    t0 = time.perf_counter()
    engine = OCREngine("ko")
    load_ms = (time.perf_counter() - t0) * 1000
    backend = engine._backend.__class__.__name__ if engine._backend else "<none>"
    print(f"  Loaded in {load_ms:.0f} ms   backend={backend}")
    if backend == "<none>":
        print("\n[WARNING] No OCR backend successfully initialized.\n" \
              "Check previous error messages and install/repair dependencies.")

    image = make_test_image()
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    print()

    # simple helpers for demonstration
    def run(label, fn):
        t0 = time.perf_counter()
        r = fn()
        ms = (time.perf_counter() - t0) * 1000
        print(f"  {label}: {ms:.0f} ms")
        return r

    print("[Test 1] full read")
    run("detect", lambda: engine.detect(image))

    print("[Test 2] half-size")
    img_small = image[::2, ::2, :]
    run("detect", lambda: engine.detect(img_small))

    print("[Test 3] non-block output (only text)")
    # backend may or may not support detail arg; just call detect and ignore bboxes
    run("detect", lambda: [b.text for b in engine.detect(image)])

    print("\nDone.")


if __name__ == "__main__":
    main()
