import winrt.windows.media.ocr as ocr
print('members=', dir(ocr.OcrEngine))
print(ocr.OcrEngine.__doc__)
# try calling language listing if exists
if hasattr(ocr.OcrEngine, 'get_available_recognizer_languages'):
    print('langs', ocr.OcrEngine.get_available_recognizer_languages())
else:
    print('no get_available_recognizer_languages, available attributes:')
    for attr in dir(ocr.OcrEngine):
        if 'language' in attr.lower():
            print(' ', attr)