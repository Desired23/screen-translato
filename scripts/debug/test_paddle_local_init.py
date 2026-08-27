from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config
from ocr_engine import _discover_paddle_model_dirs, _PaddleEngine


def main() -> int:
    cfg = load_config()
    lang = cfg.get("source_language", "ja")
    paths = _discover_paddle_model_dirs(lang, cfg)
    print(json.dumps({"lang": lang, "paths": paths}, ensure_ascii=False, indent=2))
    engine = _PaddleEngine(lang, config=cfg)
    print("model_ready=", engine._model is not None)
    return 0 if engine._model is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
