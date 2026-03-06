from ocr_engine import OCREngine
try:
    eng = OCREngine('ko')
    print('engine backend', eng.backend_name)
except Exception as e:
    import traceback; traceback.print_exc()
