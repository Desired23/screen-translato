from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path


def _print_header():
    print("[DBG] test_rapidocr_init", flush=True)
    print(f"[DBG] exe={sys.executable}", flush=True)
    print(f"[DBG] version={sys.version}", flush=True)
    print(f"[DBG] cwd={Path.cwd()}", flush=True)
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        lowered = entry.lower()
        if any(key in lowered for key in ("python", "onnxruntime", "torch", "ctranslate2")):
            print(f"[DBG] PATH {entry}", flush=True)


def main():
    _print_header()
    try:
        from rapidocr_onnxruntime import RapidOCR

        print("[DBG] import=ok", flush=True)
        engine = RapidOCR()
        print(f"[DBG] init=ok type={type(engine).__name__}", flush=True)
    except Exception as exc:
        print(f"[DBG] init=fail error={exc!r}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
