# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs


hiddenimports = [
    "pynput.keyboard._win32",
    "pynput.mouse._win32",
    "rapidocr_onnxruntime",
    "paddle",
    "paddlex",
    "paddleocr",
    "ctranslate2",
    "transformers",
    "sentencepiece",
]

excludes = [
    "torch",
    "torchvision",
    "torchaudio",
    "easyocr",
    "argostranslate",
    "winrt",
    "spacy",
    "tensorflow",
    "matplotlib",
    "pandas",
    "scipy",
    "sklearn",
    "tensorboard",
    "pytest",
]

datas = []
for pkg in ("rapidocr_onnxruntime", "paddleocr", "paddlex", "paddle"):
    datas += collect_data_files(pkg, include_py_files=False)
binaries = []
for pkg in ("onnxruntime", "ctranslate2", "paddle"):
    try:
        binaries += collect_dynamic_libs(pkg)
    except Exception:
        pass


a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ScreenTranslator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ScreenTranslator",
)
