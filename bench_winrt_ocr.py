"""
bench_winrt_ocr.py - Benchmark Windows built-in OCR API speed.
"""
import asyncio
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
    return img


async def run_windows_ocr(pil_image, lang_tag="ko"):
    import winrt.windows.media.ocr as winrt_ocr
    import winrt.windows.graphics.imaging as imaging
    import winrt.windows.storage.streams as streams

    # Find the Korean language
    lang = None
    for l in winrt_ocr.OcrEngine.get_available_recognizer_languages():
        if l.language_tag.startswith(lang_tag):
            lang = l
            break

    if lang is None:
        # fallback: list available
        available = [l.language_tag for l in winrt_ocr.OcrEngine.get_available_recognizer_languages()]
        print(f"  Korean not found. Available: {available}")
        return []

    engine = winrt_ocr.OcrEngine.try_create_from_language(lang)
    if engine is None:
        print("  Failed to create OCR engine")
        return []

    # Convert PIL image to Windows SoftwareBitmap
    img_bytes = pil_image.convert("RGBA").tobytes()
    width, height = pil_image.size

    # Create SoftwareBitmap from raw bytes
    writer = streams.DataWriter()
    writer.write_bytes(list(img_bytes))
    buf = writer.detach_buffer()

    bitmap = imaging.SoftwareBitmap.create_copy_from_buffer(
        buf,
        imaging.BitmapPixelFormat.RGBA8,
        width,
        height,
        imaging.BitmapAlphaMode.PREMULTIPLIED,
    )

    result = await engine.recognize_async(bitmap)
    texts = []
    for line in result.lines:
        texts.append(line.text)
    return texts


async def main():
    print("=" * 50)
    print("Windows OCR API Benchmark (Korean)")
    print("=" * 50)

    img = make_test_image()

    # Warm up
    print("Warming up...")
    await run_windows_ocr(img)
    print("Done\n")

    # Benchmark
    N = 3
    times = []
    for i in range(N):
        t0 = time.perf_counter()
        texts = await run_windows_ocr(img)
        ms = (time.perf_counter() - t0) * 1000
        times.append(ms)
        print(f"  Run {i+1}: {ms:.0f} ms  → {texts[:2]}")

    avg = sum(times) / len(times)
    print(f"\n  Average: {avg:.0f} ms")
    print(f"  EasyOCR was: ~1400 ms")
    print(f"  Speedup: {1400/max(avg,1):.1f}x")


if __name__ == "__main__":
    asyncio.run(main())
