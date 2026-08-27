from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path


def _print_header():
    print("[DBG] test_onnxruntime_import", flush=True)
    print(f"[DBG] exe={sys.executable}", flush=True)
    print(f"[DBG] version={sys.version}", flush=True)
    print(f"[DBG] cwd={Path.cwd()}", flush=True)


def _print_path_summary():
    print("[DBG] PATH summary:", flush=True)
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        lowered = entry.lower()
        if any(key in lowered for key in ("python", "onnxruntime", "torch", "ctranslate2")):
            print(f"[DBG]   {entry}", flush=True)


def _print_ort_files(ort_module):
    ort_root = Path(ort_module.__file__).resolve().parent
    capi_dir = ort_root / "capi"
    print(f"[DBG] onnxruntime.__file__={ort_module.__file__}", flush=True)
    print(f"[DBG] capi_dir={capi_dir}", flush=True)
    if capi_dir.is_dir():
        for item in sorted(capi_dir.iterdir()):
            if item.is_file():
                print(f"[DBG]   capi:{item.name}", flush=True)


def main():
    _print_header()
    _print_path_summary()
    try:
        import onnxruntime as ort

        print(f"[DBG] import=ok version={ort.__version__}", flush=True)
        _print_ort_files(ort)
    except Exception as exc:
        print(f"[DBG] import=fail error={exc!r}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
