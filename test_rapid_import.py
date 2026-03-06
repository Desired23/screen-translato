import traceback
try:
    from rapidocr_onnxruntime import RapidOCR
    print('rapid import ok')
except Exception as e:
    traceback.print_exc()
