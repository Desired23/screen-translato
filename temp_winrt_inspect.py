import winrt.windows.media.ocr as ocr
import inspect
print(inspect.getsource(ocr.OcrEngine))
print('dir', dir(ocr.OcrEngine))
print('doc', ocr.OcrEngine.__doc__)
for name in dir(ocr.OcrEngine):
    if 'recognizer' in name.lower() or 'language' in name.lower():
        print('member:', name, getattr(ocr.OcrEngine, name))