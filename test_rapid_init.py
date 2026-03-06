import traceback
try:
    from rapidocr_onnxruntime import RapidOCR
    print('module imported')
    r = RapidOCR()
    print('instance created', r)
except Exception as e:
    traceback.print_exc()
