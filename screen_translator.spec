# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files


hiddenimports = [
    "pynput.keyboard._win32",
    "pynput.mouse._win32",
    "rapidocr_onnxruntime",
]

# Optional backends are intentionally excluded to keep release size manageable.
# Lite build stays online-first with bundled RapidOCR only.
excludes = [
    "torch",
    "easyocr",
    "paddleocr",
    "argostranslate",
    "ctranslate2",
    "transformers",
    "winrt",
]

datas = collect_data_files("rapidocr_onnxruntime", include_py_files=False)


a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=[],
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
