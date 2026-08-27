"""Safe probes for native backends that may crash the main process on import/init."""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import tempfile
import importlib.util
from functools import lru_cache
from pathlib import Path


def _sanitize_frozen_dll_path():
    if not getattr(sys, "frozen", False):
        return
    exe_dir = Path(sys.executable).resolve().parent
    internal_dir = (exe_dir / "_internal").resolve()
    keep_prefix = str(internal_dir).lower()

    entries = [p for p in str(os.environ.get("PATH", "")).split(os.pathsep) if p]
    filtered: list[str] = []
    for entry in entries:
        lowered = entry.lower()
        if "\\torch\\lib" in lowered and keep_prefix not in lowered:
            continue
        filtered.append(entry)

    preferred = [
        internal_dir / "ctranslate2",
    ]
    for dll_dir in reversed(preferred):
        if dll_dir.is_dir():
            dll_str = str(dll_dir)
            if dll_str not in filtered:
                filtered.insert(0, dll_str)
            try:
                os.add_dll_directory(dll_str)
            except Exception:
                pass

    os.environ["PATH"] = os.pathsep.join(filtered)


def _read_probe_result(result_path: str) -> tuple[bool, str]:
    raw = Path(result_path).read_text(encoding="utf-8", errors="replace").strip()
    if raw.startswith("1|"):
        return True, raw[2:]
    if raw.startswith("0|"):
        return False, raw[2:]
    return False, f"invalid probe output: {raw[:120]}"


def _load_ct2_ext_module():
    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir / "_internal" / "ctranslate2")
    candidates.append(Path(__file__).resolve().parent / "ctranslate2")
    for p in sys.path:
        try:
            candidates.append(Path(p) / "ctranslate2")
        except Exception:
            continue

    seen: set[str] = set()
    ext_path: Path | None = None
    for base in candidates:
        key = str(base).lower()
        if key in seen:
            continue
        seen.add(key)
        if not base.is_dir():
            continue
        matches = sorted(base.glob("_ext*.pyd"))
        if matches:
            ext_path = matches[0]
            break

    if ext_path is None:
        raise RuntimeError("ctranslate2 _ext binary not found")

    try:
        os.add_dll_directory(str(ext_path.parent))
    except Exception:
        pass
    # C-extension export symbol is tied to module basename `_ext`
    # (PyInit__ext), so module name must remain `_ext`.
    spec = importlib.util.spec_from_file_location("_ext", str(ext_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load spec for {ext_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_probe(target, args: tuple, timeout_sec: float) -> tuple[bool, str]:
    ctx = mp.get_context("spawn")
    with tempfile.NamedTemporaryFile(prefix="st_probe_", suffix=".txt", delete=False) as tf:
        result_path = tf.name
    if getattr(sys, "frozen", False):
        # In frozen builds, multiprocessing spawn can bootstrap the executable
        # in ways that interfere with native DLL loading (false probe failures).
        try:
            target(result_path, *args)
        except Exception as inline_exc:
            return False, f"inline probe failed: {inline_exc}"
        try:
            return _read_probe_result(result_path)
        except Exception:
            return False, "probe produced no result"
        finally:
            try:
                Path(result_path).unlink(missing_ok=True)
            except Exception:
                pass

    p = ctx.Process(target=target, args=(result_path, *args), daemon=True)
    try:
        p.start()
    except RuntimeError as exc:
        # Some call sites (ad-hoc scripts/tests) may not be wrapped in
        # `if __name__ == "__main__"` on Windows spawn.
        # In that case, fall back to inline probe instead of reporting a
        # false backend failure.
        if "An attempt has been made to start a new process" not in str(exc):
            return False, str(exc)
        try:
            target(result_path, *args)
        except Exception as inline_exc:
            return False, f"inline probe failed: {inline_exc}"
        try:
            return _read_probe_result(result_path)
        except Exception:
            return False, "probe produced no result"
        finally:
            try:
                Path(result_path).unlink(missing_ok=True)
            except Exception:
                pass
    p.join(timeout_sec)

    if p.is_alive():
        p.terminate()
        p.join(2.0)
        return False, "probe timeout"

    if p.exitcode not in (0, None):
        return False, f"probe crashed with exitcode={p.exitcode}"

    try:
        return _read_probe_result(result_path)
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
        _sanitize_frozen_dll_path()
        # Tokenizer loading does not require DL frameworks; keep probe logs clean.
        os.environ.setdefault("USE_TORCH", "0")
        os.environ.setdefault("TRANSFORMERS_NO_TORCH", "1")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        try:
            from transformers.utils import logging as _hf_logging

            _hf_logging.set_verbosity_error()
        except Exception:
            pass
        from transformers import AutoTokenizer

        if not os.path.isdir(model_dir):
            _write_probe_result(result_path, False, f"model_dir_not_found:{model_dir}")
            return

        tok_src = tokenizer_path if os.path.isdir(tokenizer_path) else model_dir
        ct2_ext = _load_ct2_ext_module()
        _ = ct2_ext.Translator(model_dir, device=device, compute_type=compute_type)
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
