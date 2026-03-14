"""Safe probes for native backends that may crash the main process on import/init."""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path


def _run_probe(target, args: tuple, timeout_sec: float) -> tuple[bool, str]:
    ctx = mp.get_context("spawn")
    with tempfile.NamedTemporaryFile(prefix="st_probe_", suffix=".txt", delete=False) as tf:
        result_path = tf.name
    p = ctx.Process(target=target, args=(result_path, *args), daemon=True)
    p.start()
    p.join(timeout_sec)

    if p.is_alive():
        p.terminate()
        p.join(2.0)
        return False, "probe timeout"

    if p.exitcode not in (0, None):
        return False, f"probe crashed with exitcode={p.exitcode}"

    try:
        raw = Path(result_path).read_text(encoding="utf-8", errors="replace").strip()
        if raw.startswith("1|"):
            return True, raw[2:]
        if raw.startswith("0|"):
            return False, raw[2:]
        return False, f"invalid probe output: {raw[:120]}"
    except Exception:
        return False, "probe produced no result"
    finally:
        try:
            Path(result_path).unlink(missing_ok=True)
        except Exception:
            pass


def _write_probe_result(path: str, ok: bool, msg: str):
    prefix = "1|" if ok else "0|"
    Path(path).write_text(prefix + str(msg or ""), encoding="utf-8", errors="replace")


def _probe_rapidocr_child(result_path: str):
    try:
        from rapidocr_onnxruntime import RapidOCR

        _ = RapidOCR()
        _write_probe_result(result_path, True, "ok")
    except Exception as exc:
        _write_probe_result(result_path, False, str(exc))


def _probe_argos_child(result_path: str):
    try:
        import argostranslate.translate as _argos_translate  # noqa: F401

        _write_probe_result(result_path, True, "ok")
    except Exception as exc:
        _write_probe_result(result_path, False, str(exc))


def _probe_nllb_child(result_path: str, model_dir: str, tokenizer_path: str, device: str, compute_type: str):
    try:
        import ctranslate2
        from transformers import AutoTokenizer

        if not os.path.isdir(model_dir):
            _write_probe_result(result_path, False, f"model_dir_not_found:{model_dir}")
            return

        tok_src = tokenizer_path if os.path.isdir(tokenizer_path) else model_dir
        _ = ctranslate2.Translator(model_dir, device=device, compute_type=compute_type)
        _ = AutoTokenizer.from_pretrained(tok_src, use_fast=False, local_files_only=True)
        _write_probe_result(result_path, True, "ok")
    except Exception as exc:
        _write_probe_result(result_path, False, str(exc))


@lru_cache(maxsize=8)
def probe_rapidocr(timeout_sec: float = 8.0) -> tuple[bool, str]:
    return _run_probe(_probe_rapidocr_child, tuple(), timeout_sec)


@lru_cache(maxsize=8)
def probe_argos(timeout_sec: float = 8.0) -> tuple[bool, str]:
    return _run_probe(_probe_argos_child, tuple(), timeout_sec)


def _resolve_local_dir(path_value: str) -> str:
    raw = Path(str(path_value or "").strip()).expanduser()
    if raw.is_absolute():
        return str(raw.resolve())
    base_candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        base_candidates.append(Path(sys.executable).resolve().parent)
    base_candidates.append(Path(__file__).resolve().parent)
    base_candidates.append(Path.cwd())
    for base in base_candidates:
        candidate = (base / raw).resolve()
        if candidate.exists():
            return str(candidate)
    return str((base_candidates[0] / raw).resolve())


@lru_cache(maxsize=8)
def probe_nllb(
    model_dir: str,
    tokenizer_path: str | None,
    device: str = "cpu",
    compute_type: str = "int8",
    timeout_sec: float = 20.0,
) -> tuple[bool, str]:
    resolved_model = _resolve_local_dir(model_dir)
    resolved_tok = _resolve_local_dir(tokenizer_path or model_dir)
    return _run_probe(
        _probe_nllb_child,
        (resolved_model, resolved_tok, str(device or "cpu"), str(compute_type or "int8")),
        timeout_sec,
    )
