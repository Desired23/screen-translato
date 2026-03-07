"""
install_nllb_ct2.py - Download and convert NLLB model for local CTranslate2 inference.

Usage:
    python install_nllb_ct2.py
    python install_nllb_ct2.py --model-id facebook/nllb-200-distilled-600M --quantization int8
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from ctranslate2.converters import TransformersConverter
except Exception as exc:
    print("[NLLB] ctranslate2 converter is unavailable.")
    print("Install first: pip install ctranslate2")
    raise SystemExit(1) from exc

try:
    from transformers import AutoTokenizer
except Exception as exc:
    print("[NLLB] transformers is unavailable.")
    print("Install first: pip install transformers sentencepiece")
    raise SystemExit(1) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install local NLLB CTranslate2 model")
    parser.add_argument(
        "--model-id",
        default="facebook/nllb-200-distilled-600M",
        help="Hugging Face source model id",
    )
    parser.add_argument(
        "--tokenizer-id",
        default=None,
        help="Tokenizer source id (default: same as --model-id)",
    )
    parser.add_argument(
        "--output-dir",
        default=".models/nllb-ct2-int8",
        help="Output directory for converted CT2 model + tokenizer files",
    )
    parser.add_argument(
        "--quantization",
        default="int8",
        help="CTranslate2 quantization mode (int8, int8_float16, float32, ...)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[NLLB] Converting model: {args.model_id}")
    print(f"[NLLB] Output directory: {output_dir}")
    converter = TransformersConverter(args.model_id)
    converter.convert(
        str(output_dir),
        quantization=args.quantization,
        force=args.force,
    )

    tokenizer_id = args.tokenizer_id or args.model_id
    print(f"[NLLB] Downloading tokenizer: {tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.save_pretrained(str(output_dir))

    print("[NLLB] Installation completed.")
    print(f"[NLLB] Set nllb_model_dir to: {output_dir}")
    print(f"[NLLB] Set nllb_tokenizer_path to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
