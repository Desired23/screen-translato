import sys

print('start test_order')
# mimic WinRT check
try:
    import winrt.windows.media.ocr as winrt_ocr
    print('winrt module loaded')
    # try to find languages (simulate)
    langs = list(winrt_ocr.OcrEngine.get_available_recognizer_languages())
    print('winrt langs', langs[:5])
except Exception as e:
    print('winrt fail', e)

# now try RapidOCR
import traceback
try:
    from rapidocr_onnxruntime import RapidOCR
    print('rapid module imported')
    r = RapidOCR()
    print('rapid instance ok')
except Exception as e:
    print('rapid failed')
    traceback.print_exc()
