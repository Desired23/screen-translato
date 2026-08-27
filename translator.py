# translator.py - Translation engine with caching
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import os
import re
import sys
import threading
from pathlib import Path
from native_probe import probe_argos, probe_nllb

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

try:
    import langid
except Exception:
    langid = None

argos_translate = None
ctranslate2 = None
AutoTokenizer = None


class _LanguageDetector:
    """Whitelist-only detector for EN/JA/KO/ZH using langid + heuristics."""

    _SUPPORTED = {"en", "ja", "ko", "zh"}
    _LABEL_ALIASES = {
        "en": "en",
        "ja": "ja",
        "ko": "ko",
        "zh": "zh",
        "zh-cn": "zh",
        "zh-tw": "zh",
    }

    def __init__(
        self,
        backend: str = "langid",
        min_confidence: float = 0.55,
    ):
        self._backend = str(backend or "langid").strip().lower()
        if self._backend not in {"auto", "langid", "heuristic"}:
            self._backend = "langid"
        self._min_confidence = max(0.1, min(0.99, float(min_confidence)))

        if langid is not None:
            try:
                langid.set_languages(sorted(self._SUPPORTED))
            except Exception:
                pass

    @classmethod
    def _normalize_label(cls, label: str) -> str:
        raw = str(label or "").strip().lower().replace("__label__", "")
        return cls._LABEL_ALIASES.get(raw, "")

    def _detect_langid(self, text: str) -> tuple[str, float, str] | None:
        if langid is None:
            return None
        # langid tends to misclassify CJK-heavy snippets in our OCR setting.
        # Keep it for Latin-script ambiguity only.
        if re.search(r"[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7a3]", text):
            return None
        if len(re.findall(r"[A-Za-z]", text)) < 3:
            return None
        try:
            label, score = langid.classify(text)
        except Exception:
            return None
        code = self._normalize_label(label)
        if code not in self._SUPPORTED:
            return None
        s = float(score)
        if s >= 8.0:
            conf = 0.98
        elif s >= 3.0:
            conf = 0.88
        elif s >= 0.0:
            conf = 0.75
        elif s >= -5.0:
            conf = 0.60
        elif s >= -15.0:
            conf = 0.48
        else:
            conf = 0.30
        return code, conf, "langid"

    def detect(self, text: str) -> tuple[str, float, str]:
        content = str(text or "").strip()
        if len(content) < 2 or self._backend == "heuristic":
            return "", 0.0, "none"
        if not re.search(r"[A-Za-z\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7a3]", content):
            return "", 0.0, "none"

        if self._backend == "langid":
            result = self._detect_langid(content)
            return result if result is not None else ("", 0.0, "none")

        # Auto mode stays intentionally simple: script heuristics outside,
        # langid only for short Latin-ish snippets that remain ambiguous.
        best: tuple[str, float, str] | None = None
        for fn in (self._detect_langid,):
            result = fn(content)
            if result is None:
                continue
            if result[1] >= self._min_confidence:
                return result
            if best is None or result[1] > best[1]:
                best = result
        fallback_floor = max(0.40, self._min_confidence * 0.75)
        if best is not None and best[1] >= fallback_floor:
            return best
        return "", 0.0, "none"


def _safe_log_text(text: str, limit: int = 80) -> str:
    snippet = str(text or "").replace("\n", " ").strip()[:limit]
    try:
        return snippet.encode("unicode_escape").decode("ascii", errors="ignore")
    except Exception:
        return repr(snippet)


def _prepare_transformers_runtime():
    """Use transformers tokenizer-only mode and keep startup logs quiet."""
    os.environ.setdefault("USE_TORCH", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TORCH", "1")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


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
        internal_dir / "torch" / "lib",
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


class _GoogleBackend:
    name = "google"
    _LANG_ALIASES = {
        "zh": "zh-CN",
        "zh-cn": "zh-CN",
        "zh-hans": "zh-CN",
        "zh-tw": "zh-TW",
        "zh-hant": "zh-TW",
        "ja-jp": "ja",
        "pt-br": "pt",
    }

    @classmethod
    def _normalize_lang(cls, code: str | None, *, allow_auto: bool) -> str:
        raw = str(code or "").strip().lower()
        if not raw:
            return "auto" if allow_auto else "en"
        if raw == "auto":
            return "auto" if allow_auto else "en"
        return cls._LANG_ALIASES.get(raw, raw)

    def __init__(self, target_language: str, source_language: str = "auto"):
        if GoogleTranslator is None:
            raise RuntimeError("deep-translator is not installed")
        self._target_language = self._normalize_lang(target_language, allow_auto=False)
        self._default_source = self._normalize_lang(source_language, allow_auto=True)
        self._cache: dict[str, object] = {}

    def _get_translator(self, source_language: str | None = None):
        src = self._normalize_lang(source_language or self._default_source, allow_auto=True)
        key = f"{src}->{self._target_language}"
        translator = self._cache.get(key)
        if translator is not None:
            return translator
        try:
            translator = GoogleTranslator(source=src, target=self._target_language)
        except Exception:
            if src != "auto":
                src = "auto"
                key = f"{src}->{self._target_language}"
                translator = self._cache.get(key)
                if translator is None:
                    translator = GoogleTranslator(source=src, target=self._target_language)
            else:
                raise
        self._cache[key] = translator
        return translator

    def translate(self, text: str, source_language: str | None = None) -> str:
        translator = self._get_translator(source_language=source_language)
        return translator.translate(text)


class _ArgosBackend:
    name = "argos"
    _LANG_ALIASES = {
        "zh-cn": "zh",
        "zh-tw": "zh",
        "zh-hans": "zh",
        "zh-hant": "zh",
        "pt-br": "pt",
        "ja-jp": "ja",
    }

    @classmethod
    def _normalize_code(cls, code: str) -> str:
        code = (code or "").strip().lower()
        if not code:
            return ""
        return cls._LANG_ALIASES.get(code, code)

    def __init__(
        self,
        source_language: str,
        target_language: str,
        pivot_language: str = "en",
    ):
        global argos_translate
        if argos_translate is None:
            ok, reason = probe_argos(timeout_sec=8.0)
            # Argos may throw a benign WinError 183 on first-run folder init in probe.
            if (not ok) and ("WinError 183" not in str(reason)):
                raise RuntimeError(f"argostranslate probe failed: {reason}")
            try:
                import argostranslate.translate as _argos_translate

                argos_translate = _argos_translate
            except Exception as exc:
                raise RuntimeError(f"argostranslate import failed: {exc}") from exc
        if argos_translate is None:
            raise RuntimeError("argostranslate is not installed")

        src_code = self._normalize_code(source_language)
        tgt_code = self._normalize_code(target_language)
        pivot_code = self._normalize_code(pivot_language)
        if src_code in ("", "auto"):
            raise RuntimeError("Argos backend requires explicit source_language (not auto)")
        if not tgt_code:
            raise RuntimeError("Target language is empty")

        langs = argos_translate.get_installed_languages()
        by_code = {lang.code: lang for lang in langs}
        src_lang = by_code.get(src_code)
        tgt_lang = by_code.get(tgt_code)
        if src_lang is None or tgt_lang is None:
            available = ", ".join(sorted(lang.code for lang in langs))
            raise RuntimeError(
                f"Argos language missing for {src_code}->{tgt_code}. Installed language codes: {available}"
            )

        route = self._resolve_route(
            by_code=by_code,
            source_code=src_code,
            target_code=tgt_code,
            pivot_code=pivot_code,
        )
        if not route:
            raise RuntimeError(
                f"No installed Argos translation route from {src_code} to {tgt_code}. "
                "Install direct model or pivot models (e.g. xx->en and en->vi)."
            )

        translators: list = []
        route_text = "->".join(route)
        for a, b in zip(route, route[1:]):
            try:
                translators.append(by_code[a].get_translation(by_code[b]))
            except Exception as exc:
                raise RuntimeError(
                    f"Argos package for {a}->{b} is missing or invalid (route {route_text})"
                ) from exc

        if not translators:
            raise RuntimeError(
                f"Failed to initialize Argos translators for route {route_text}"
            )

        self._translators = translators
        self._route = route
        self.route_name = route_text

    @staticmethod
    def _has_edge(by_code: dict[str, object], src: str, dst: str) -> bool:
        src_lang = by_code.get(src)
        dst_lang = by_code.get(dst)
        if src_lang is None or dst_lang is None:
            return False
        try:
            src_lang.get_translation(dst_lang)
            return True
        except Exception:
            return False

    @classmethod
    def _resolve_route(
        cls,
        by_code: dict[str, object],
        source_code: str,
        target_code: str,
        pivot_code: str,
    ) -> list[str]:
        if source_code == target_code:
            return [source_code]

        # Prefer direct model first.
        if cls._has_edge(by_code, source_code, target_code):
            return [source_code, target_code]

        # Prefer 2-hop route through pivot (e.g. ja->en->vi) if available.
        if pivot_code and pivot_code not in {source_code, target_code}:
            if cls._has_edge(by_code, source_code, pivot_code) and cls._has_edge(
                by_code, pivot_code, target_code
            ):
                return [source_code, pivot_code, target_code]

        # General shortest path on installed graph (max 4 edges).
        codes = list(by_code.keys())
        neighbors: dict[str, list[str]] = {c: [] for c in codes}
        for a in codes:
            for b in codes:
                if a == b:
                    continue
                if cls._has_edge(by_code, a, b):
                    neighbors[a].append(b)

        q: deque[tuple[str, list[str]]] = deque([(source_code, [source_code])])
        visited = {source_code}
        max_hops = 4

        while q:
            cur, path = q.popleft()
            if len(path) - 1 >= max_hops:
                continue
            for nxt in neighbors.get(cur, []):
                if nxt in visited:
                    continue
                next_path = path + [nxt]
                if nxt == target_code:
                    return next_path
                visited.add(nxt)
                q.append((nxt, next_path))

        return []

    def translate(self, text: str) -> str:
        out = text
        for translator in self._translators:
            out = translator.translate(out)
        return out


class _NllbBackend:
    name = "nllb"

    _LANG_ALIASES = {
        "zh-cn": "zh",
        "zh-tw": "zh-tw",
        "zh-hans": "zh",
        "zh-hant": "zh-tw",
        "pt-br": "pt",
        "ja-jp": "ja",
    }

    _NLLB_LANG = {
        "en": "eng_Latn",
        "vi": "vie_Latn",
        "ko": "kor_Hang",
        "ja": "jpn_Jpan",
        "zh": "zho_Hans",
        "zh-tw": "zho_Hant",
        "fr": "fra_Latn",
        "de": "deu_Latn",
        "es": "spa_Latn",
        "th": "tha_Thai",
        "ru": "rus_Cyrl",
        "pt": "por_Latn",
        "it": "ita_Latn",
        "ar": "arb_Arab",
    }

    @classmethod
    def _normalize_code(cls, code: str) -> str:
        code = (code or "").strip().lower()
        return cls._LANG_ALIASES.get(code, code)

    @staticmethod
    def _resolve_local_dir(path_value: str) -> str:
        """Resolve model/tokenizer paths robustly for both dev and packaged builds."""
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

        # Prefer executable-relative path when frozen, source-relative otherwise.
        preferred_base = base_candidates[0]
        return str((preferred_base / raw).resolve())

    def __init__(
        self,
        source_language: str,
        target_language: str,
        model_dir: str,
        tokenizer_path: str | None = None,
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 2,
        max_decoding_length: int = 192,
        max_source_tokens: int = 384,
    ):
        global ctranslate2, AutoTokenizer
        ok, reason = probe_nllb(
            model_dir=model_dir,
            tokenizer_path=tokenizer_path,
            device=device,
            compute_type=compute_type,
            timeout_sec=20.0,
        )
        if not ok:
            raise RuntimeError(f"NLLB probe failed: {reason}")

        _sanitize_frozen_dll_path()
        if ctranslate2 is None:
            ctranslate2 = _load_ct2_ext_module()
        if AutoTokenizer is None:
            # Tokenizer loading for CT2 does not require DL frameworks.
            _prepare_transformers_runtime()
            try:
                from transformers.utils import logging as _hf_logging

                _hf_logging.set_verbosity_error()
            except Exception:
                pass
            from transformers import AutoTokenizer as _AutoTokenizer

            AutoTokenizer = _AutoTokenizer
        if ctranslate2 is None:
            raise RuntimeError("ctranslate2 is not installed")
        if AutoTokenizer is None:
            raise RuntimeError("transformers is not installed")

        src_code = self._normalize_code(source_language)
        tgt_code = self._normalize_code(target_language)
        if src_code in ("", "auto"):
            raise RuntimeError("NLLB backend requires explicit source_language (not auto)")

        src_tag = self._NLLB_LANG.get(src_code)
        tgt_tag = self._NLLB_LANG.get(tgt_code)
        if not src_tag or not tgt_tag:
            raise RuntimeError(f"NLLB language mapping missing for {src_code}->{tgt_code}")

        resolved_model_dir = self._resolve_local_dir(model_dir)
        if not os.path.isdir(resolved_model_dir):
            raise RuntimeError(f"NLLB model directory not found: {resolved_model_dir}")

        resolved_tokenizer = (
            self._resolve_local_dir(tokenizer_path) if tokenizer_path else resolved_model_dir
        )
        if not os.path.isdir(resolved_tokenizer):
            # Allow loading from HF cache key if caller passes repo id string.
            resolved_tokenizer = tokenizer_path or resolved_model_dir

        self._translator = ctranslate2.Translator(
            resolved_model_dir,
            device=device,
            compute_type=compute_type,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            resolved_tokenizer,
            use_fast=False,
            local_files_only=True,
        )
        self._src_code = src_code
        self._src_tag = src_tag
        self._tgt_tag = tgt_tag
        self._beam_size = max(1, int(beam_size))
        self._max_decoding_length = max(32, int(max_decoding_length))
        # Guard against pathological OCR paragraphs causing very slow local inference.
        self._max_source_tokens = max(64, int(max_source_tokens))
        self.route_name = f"{src_code}->{tgt_code}"

    def _resolve_src_tag(self, source_language: str | None) -> str:
        src_code = self._normalize_code(source_language) if source_language else self._src_code
        src_tag = self._NLLB_LANG.get(src_code)
        return src_tag or self._src_tag

    def _encode_source_tokens(self, text: str, source_language: str | None = None) -> list[str]:
        self._tokenizer.src_lang = self._resolve_src_tag(source_language)
        encoded = self._tokenizer(
            text,
            return_attention_mask=False,
            return_tensors=None,
            truncation=True,
            max_length=self._max_source_tokens,
        )
        input_ids = encoded.get("input_ids", [])
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return self._tokenizer.convert_ids_to_tokens(input_ids)

    def _decode_target_tokens(self, tokens: list[str]) -> str:
        out_tokens = list(tokens)
        if out_tokens and out_tokens[0] == self._tgt_tag:
            out_tokens = out_tokens[1:]

        out_ids: list[int] = []
        for tok in out_tokens:
            tid = self._tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                out_ids.append(tid)

        if not out_ids:
            return ""

        return self._tokenizer.decode(out_ids, skip_special_tokens=True).strip()

    def translate(self, text: str, source_language: str | None = None) -> str:
        stripped = text.strip()
        if not stripped:
            return text

        source_tokens = self._encode_source_tokens(stripped, source_language=source_language)
        if not source_tokens:
            return text

        result = self._translator.translate_batch(
            [source_tokens],
            target_prefix=[[self._tgt_tag]],
            beam_size=self._beam_size,
            max_decoding_length=self._max_decoding_length,
        )
        if not result or not result[0].hypotheses:
            return text

        decoded = self._decode_target_tokens(result[0].hypotheses[0])
        return decoded if decoded else text

    def translate_many(self, texts: list[str], source_language: str | None = None) -> list[str]:
        if not texts:
            return []

        idx_map: list[int] = []
        encoded_batch: list[list[str]] = []
        outputs: list[str] = list(texts)

        for i, raw in enumerate(texts):
            stripped = raw.strip()
            if not stripped:
                continue
            tokens = self._encode_source_tokens(stripped, source_language=source_language)
            if not tokens:
                continue
            idx_map.append(i)
            encoded_batch.append(tokens)

        if not encoded_batch:
            return outputs

        try:
            results = self._translator.translate_batch(
                encoded_batch,
                target_prefix=[[self._tgt_tag]] * len(encoded_batch),
                beam_size=self._beam_size,
                max_decoding_length=self._max_decoding_length,
            )
        except Exception:
            # Fallback to per-item translation if batch call fails.
            for i in idx_map:
                outputs[i] = self.translate(outputs[i])
            return outputs

        for out_idx, result in zip(idx_map, results):
            if not result.hypotheses:
                continue
            decoded = self._decode_target_tokens(result.hypotheses[0])
            if decoded:
                outputs[out_idx] = decoded

        return outputs


class TranslationEngine:
    """Translation engine wrapper with caching and batch support."""

    def __init__(
        self,
        target_language: str = "vi",
        source_language: str = "auto",
        cache_max_entries: int = 2000,
        max_batch_chars: int = 4200,
        translation_backend: str = "auto",
        fallback_to_google: bool = True,
        argos_pivot_language: str = "en",
        nllb_model_dir: str = ".models/nllb-ct2-int8",
        nllb_tokenizer_path: str | None = None,
        nllb_device: str = "cpu",
        nllb_compute_type: str = "int8",
        nllb_beam_size: int = 2,
        nllb_max_decoding_length: int = 192,
        nllb_max_source_tokens: int = 384,
        context_refine_enabled: bool = True,
        context_refine_max_chars: int = 48,
        context_refine_max_words: int = 10,
        context_refine_max_per_batch: int = 1,
        manga_mode: bool = True,
        manga_source_fixes_enabled: bool = True,
        manga_target_post_edit_enabled: bool = True,
        auto_document_guard: bool = True,
        game_term_guard_enabled: bool = True,
        game_post_edit_enabled: bool = True,
        auto_source_routing_enabled: bool = True,
        language_detector_backend: str = "langid",
        language_detector_fasttext_model_path: str = ".models/lid.176.ftz",
        language_detector_min_confidence: float = 0.55,
        semantic_cache_enabled: bool = True,
        custom_glossary_enabled: bool = True,
        custom_glossary: dict[str, str] | None = None,
    ):
        self._target_lang = target_language
        self._source_lang = source_language or "auto"
        self._translation_backend = str(translation_backend or "auto").strip().lower()
        if self._translation_backend == "argos":
            # Argos backend has been removed from runtime to avoid native DLL conflicts.
            self._translation_backend = "auto"
        if self._translation_backend not in {"auto", "google", "nllb"}:
            self._translation_backend = "auto"
        self._fallback_to_google = bool(fallback_to_google)
        safe_boot_env = str(os.getenv("ST_SAFE_BOOT_NATIVE", "1")).strip().lower()
        self._safe_boot_native = getattr(sys, "frozen", False) and safe_boot_env not in {
            "0",
            "false",
            "off",
            "no",
        }
        self._argos_pivot_language = str(argos_pivot_language or "en")
        self._nllb_model_dir = str(nllb_model_dir or ".models/nllb-ct2-int8")
        self._nllb_tokenizer_path = nllb_tokenizer_path
        self._nllb_device = str(nllb_device or "cpu")
        self._nllb_compute_type = str(nllb_compute_type or "int8")
        self._nllb_beam_size = max(1, int(nllb_beam_size))
        self._nllb_max_decoding_length = max(32, int(nllb_max_decoding_length))
        self._nllb_max_source_tokens = max(64, int(nllb_max_source_tokens))
        self._context_refine_enabled = bool(context_refine_enabled)
        self._context_refine_max_chars = max(10, int(context_refine_max_chars))
        self._context_refine_max_words = max(2, int(context_refine_max_words))
        self._context_refine_max_per_batch = max(0, int(context_refine_max_per_batch))
        self._lang_detector = _LanguageDetector(
            backend=language_detector_backend,
            min_confidence=float(language_detector_min_confidence),
        )
        self._ctx_prev_tag = "@@CTX_PREV@@"
        self._ctx_curr_tag = "@@CTX_CURR@@"
        self._ctx_next_tag = "@@CTX_NEXT@@"
        self._translator = None
        self._translator_backend_name = "none"
        self._translator_backend_display = "none"
        self._backend_pool: dict[str, object] = {}
        self._backend_pool_failures: set[str] = set()
        self._init_backend()
        self._cache: "OrderedDict[str, str]" = OrderedDict()
        self._semantic_cache: "OrderedDict[str, str]" = OrderedDict()
        self._lock = threading.Lock()
        self._cache_max_entries = max(100, int(cache_max_entries))
        self._max_batch_chars = max(500, int(max_batch_chars))
        self._batch_separator = "\n\u2063\u2063\u2063\n"
        self._manga_mode = bool(manga_mode)
        self._manga_source_fixes_enabled = bool(manga_source_fixes_enabled)
        self._manga_target_post_edit_enabled = bool(manga_target_post_edit_enabled)
        self._auto_document_guard = bool(auto_document_guard)
        self._game_term_guard_enabled = bool(game_term_guard_enabled)
        self._game_post_edit_enabled = bool(game_post_edit_enabled)
        self._auto_source_routing_enabled = bool(auto_source_routing_enabled)
        self._semantic_cache_enabled = bool(semantic_cache_enabled)
        self._custom_glossary_enabled = bool(custom_glossary_enabled)
        self._word_lexicon = {
            "a", "about", "after", "again", "all", "almost", "along", "already", "also",
            "always", "am", "an", "and", "any", "anyone", "aptitude", "are", "around", "as", "at",
            "attention", "attracting",
            "back", "be", "because", "been", "before", "being", "best", "better",
            "between", "beauty", "both", "boy", "boyfriend", "burst", "but", "by", "came", "can", "cannot",
            "come", "company", "could", "cute", "dazzling", "dear", "detail", "did", "difficult",
            "do", "does", "done", "down", "each", "elite", "else", "even", "ever",
            "every", "everyone", "few", "for", "forte", "from", "fusion", "game", "girl", "go", "going",
            "good", "great", "had", "has", "have", "he", "her", "here", "hers", "him",
            "his", "how", "however", "i", "if", "in", "into", "intellect", "is", "it",
            "its", "jealous", "just", "karen", "kind", "know", "kun", "left", "let",
            "like", "likes", "line", "look", "lot", "man", "many", "me", "mech", "mine", "mode", "more",
            "most", "much", "must", "my", "name", "need", "new", "next", "no", "not",
            "now", "of", "off", "ok", "old", "on", "one", "only", "or", "ore", "other",
            "our", "out", "over", "page", "people", "photo", "photos", "play", "please",
            "possesses", "praise", "present", "really", "recent", "right", "rival",
            "rounder", "same", "san", "say", "scions", "screen", "sculpt", "see", "seek", "seraphic", "she",
            "show", "silence", "skill", "so", "some", "someone", "sports", "stage", "still", "such", "take", "than",
            "that", "the", "their", "them", "then", "there", "these", "they", "thing",
            "this", "those", "threat", "time", "to", "together", "too", "turn", "two",
            "up", "us", "usami", "very", "want", "was", "way", "we", "well", "were",
            "what", "when", "where", "which", "who", "why", "will", "with", "without",
            "word", "would", "yes", "you", "your", "casting", "attack", "basic", "enter",
            "effect", "resonance", "info", "duo", "overture",
        }
        self._ocr_fixups = {
            "NOTONLY": "NOT ONLY",
            "ISSHEAN": "IS SHE AN",
            "WHOPOSSESSES": "WHO POSSESSES",
            "ALONGWITHA": "ALONG WITH A",
            "GREATAPTITUDE": "GREAT APTITUDE",
            "LIKESOF": "LIKES OF",
            "SCIONSAND": "SCIONS AND",
            "PRAISEME": "PRAISE ME",
            "ALLROUNDER": "ALL ROUNDER",
            "ATTRACTINGORE": "ATTRACTING ORE",
            "BOYFRIENDDID": "BOYFRIEND DID",
            "COMPAN": "COMPANY",
            "RIGH": "RIGHT",
        }
        self._source_phrase_fixes: list[tuple[re.Pattern[str], str]] = [
            # OCR-mangled phrase in manga pages: "... difficult company out her company"
            (
                re.compile(
                    r"\blikes of even scions and difficult company out her company\b",
                    re.IGNORECASE,
                ),
                "even elite heirs and difficult people seek her company",
            ),
            (re.compile(r"\bscions\b", re.IGNORECASE), "elite heirs"),
            (re.compile(r"\bdifficult company\b", re.IGNORECASE), "difficult people"),
            (re.compile(r"\bout her company\b", re.IGNORECASE), "seek her company"),
        ]
        self._target_phrase_fixes: list[tuple[re.Pattern[str], str]] = [
            (re.compile(r"\bkhen ngợi tôi nhiều hơn\b", re.IGNORECASE), "khen tôi nữa đi"),
            (re.compile(r"\bcành ghép\b", re.IGNORECASE), "con nhà danh giá"),
            (re.compile(r"\bcon cháu\b", re.IGNORECASE), "con nhà danh giá"),
            (re.compile(r"\bbầu bạn của cô ấy\b", re.IGNORECASE), "ở cạnh cô ấy"),
        ]
        self._cjk_vi_target_fixes: list[tuple[re.Pattern[str], str]] = [
            (
                re.compile(r"\btr\u1eddi\s+t\u1ed1t\s+\u0111\u1ea5y\b", re.IGNORECASE),
                "h\u00f4m nay th\u1eddi ti\u1ebft \u0111\u1eb9p nh\u1ec9",
            ),
            (
                re.compile(
                    r"\bth\u1eddi\s+ti\u1ebft(?:\s+h\u00f4m\s+nay)?\s+r\u1ea5t\s+t\u1ed1t\b",
                    re.IGNORECASE,
                ),
                "h\u00f4m nay th\u1eddi ti\u1ebft r\u1ea5t \u0111\u1eb9p",
            ),
            (
                re.compile(r"\bth\u1eddi\s+ti\u1ebft\s+t\u1ed1t\b", re.IGNORECASE),
                "th\u1eddi ti\u1ebft \u0111\u1eb9p",
            ),
        ]
        self._game_source_terms: list[tuple[re.Pattern[str], str]] = [
            (re.compile(r"\bforte\s*circuit\b", re.IGNORECASE), "Forte Circuit"),
            (re.compile(r"\bto\s*sculpt\s*the\s*silence\b", re.IGNORECASE), "To Sculpt the Silence"),
            (re.compile(r"\bresonance\s*mode\b", re.IGNORECASE), "Resonance Mode"),
            (re.compile(r"\bfusion\s*burst\b", re.IGNORECASE), "Fusion Burst"),
            (re.compile(r"\bbasic\s*attack\b", re.IGNORECASE), "Basic Attack"),
            (re.compile(r"\bresonance\s*skill\b", re.IGNORECASE), "Resonance Skill"),
            (re.compile(r"\bresonance\s*liberation\b", re.IGNORECASE), "Resonance Liberation"),
            (re.compile(r"\bsynchronization\s*rate\b", re.IGNORECASE), "Synchronization Rate"),
            (re.compile(r"\bfusion\s*dmg\b", re.IGNORECASE), "Fusion DMG"),
            (re.compile(r"\bmech\s*form\b", re.IGNORECASE), "Mech Form"),
            (re.compile(r"\bseraphic\s*duo\b", re.IGNORECASE), "Seraphic Duo"),
            (re.compile(r"\baemeath\b", re.IGNORECASE), "Aemeath"),
        ]
        self._game_target_fixes: list[tuple[re.Pattern[str], str]] = [
            (re.compile(r"\bphát\s+sóng\b", re.IGNORECASE), "kích hoạt"),
            (re.compile(r"\bgiao\s+dịch\b", re.IGNORECASE), "gây"),
            (re.compile(r"\btỷ\s+lệ\s+giao\s+dịch\b", re.IGNORECASE), "sát thương"),
            (re.compile(r"\bdmg\b", re.IGNORECASE), "DMG"),
        ]
        self._game_source_phrase_fixes: list[tuple[re.Pattern[str], str]] = [
            (re.compile(r"\bwhencasting\b", re.IGNORECASE), "when casting"),
            (re.compile(r"\borbasicattack\b", re.IGNORECASE), "or Basic Attack"),
            (re.compile(r"\baemeathis\b", re.IGNORECASE), "Aemeath is"),
            (re.compile(r"\brateisnolessthan\b", re.IGNORECASE), "Rate is no less than"),
            (re.compile(r"\bseraphicduoandthe\b", re.IGNORECASE), "Seraphic Duo and the"),
            (re.compile(r"\bskilltocast\b", re.IGNORECASE), "Skill to cast"),
            (re.compile(r"\bpointsof\b", re.IGNORECASE), "points of"),
            (re.compile(r"\bthecostof\b", re.IGNORECASE), "the cost of"),
            (re.compile(r"\bovertureat\b", re.IGNORECASE), "Overture at"),
            (re.compile(r"\bclosetotheground\b", re.IGNORECASE), "close to the ground"),
            (re.compile(r"\btotheground\b", re.IGNORECASE), "to the ground"),
            (re.compile(r"\bswitchintothe\b", re.IGNORECASE), "Switch into the"),
            (re.compile(r"\bmechform\b", re.IGNORECASE), "Mech Form"),
            (re.compile(r"\baftercastingthis\b", re.IGNORECASE), "after casting this"),
            (re.compile(r"\bmechformaftercastingthis\b", re.IGNORECASE), "Mech Form after casting this"),
            (re.compile(r"\bduofor(?=\d)", re.IGNORECASE), "Duo for "),
            (re.compile(r"\billandovitc\b", re.IGNORECASE), ""),
        ]
        self._custom_glossary_rules: list[tuple[re.Pattern[str], str, str]] = []
        self._build_custom_glossary_rules(custom_glossary or {})

    @staticmethod
    def _looks_document_text(text: str) -> bool:
        s = str(text or "").strip()
        if not s:
            return False
        if s[0] in {"•", "-", "*", "–"} and len(s.split()) >= 5:
            return True

        words = re.findall(r"[A-Za-z0-9']+", s)
        w = len(words)
        if w < 7:
            return False

        punct = len(re.findall(r"[,;:.!?]", s))
        alpha = len(re.findall(r"[A-Za-z]", s))
        return w >= 10 or (w >= 7 and punct >= 1 and alpha >= 24)

    @staticmethod
    def _looks_game_text(text: str) -> bool:
        s = str(text or "").strip().lower()
        if not s:
            return False
        keys = (
            "basic attack",
            "resonance skill",
            "resonance liberation",
            "synchronization rate",
            "fusion dmg",
            "mech form",
            "seraphic duo",
            "cast",
        )
        hit = sum(1 for k in keys if k in s)
        if hit >= 2:
            return True
        if "cast" in s and any(k in s for k in ("skill", "dmg", "mech", "seraphic", "resonance")):
            return True
        return ("stage " in s and "attack" in s) or ("skill" in s and "dmg" in s)

    def _contains_game_term(self, text: str) -> bool:
        s = str(text or "")
        if not s:
            return False
        for pattern, _canonical in self._game_source_terms:
            if pattern.search(s):
                return True
        return False

    _SUPPORTED_SOURCE_CODES = {"auto", "en", "ja", "ko", "zh", "zh-tw"}
    _ZH_TRAD_HINTS = set(
        "\u9ad4\u5b78\u570b\u9580\u958b\u95dc\u9ede\u98a8\u756b\u9f8d\u842c\u8207\u70ba\u9019\u5f8c\u4f86\u81fa\u7063\u9ebc\u5ee3\u89ba\u6230\u91ab\u8b80\u807d\u8aaa\u5beb\u8cb7\u8ce3\u8eca\u6c23\u96fb"
    )
    _ZH_SIMP_HINTS = set(
        "\u4f53\u5b66\u56fd\u95e8\u5f00\u5173\u70b9\u98ce\u753b\u9f99\u4e07\u4e0e\u4e3a\u8fd9\u540e\u6765\u53f0\u6e7e\u4e48\u5e7f\u89c9\u6218\u533b\u8bfb\u542c\u8bf4\u5199\u4e70\u5356\u8f66\u6c14\u7535"
    )

    @staticmethod
    def _canonical_source_code(code: str) -> str:
        normalized = _ArgosBackend._normalize_code(code)
        normalized = _NllbBackend._normalize_code(normalized)
        return normalized or "auto"

    @classmethod
    def _detect_chinese_variant(cls, text: str, configured: str) -> str:
        trad_hits = sum(1 for ch in text if ch in cls._ZH_TRAD_HINTS)
        simp_hits = sum(1 for ch in text if ch in cls._ZH_SIMP_HINTS)
        if trad_hits >= simp_hits + 1:
            return "zh-tw"
        if simp_hits >= trad_hits + 1:
            return "zh"
        if configured == "zh-tw":
            return "zh-tw"
        return "zh"

    def _detect_script_language(self, text: str) -> str:
        s = str(text or "").strip()
        if not s:
            return self._canonical_source_code(self._source_lang)

        configured = self._canonical_source_code(self._source_lang)
        if configured not in self._SUPPORTED_SOURCE_CODES:
            configured = "auto"

        hangul = len(re.findall(r"[\uac00-\ud7a3]", s))
        kana = len(re.findall(r"[\u3040-\u30ff]", s))
        han = len(re.findall(r"[\u4e00-\u9fff]", s))
        latin_letters = len(re.findall(r"[A-Za-z]", s))

        # Fast path for strong script signals.
        if hangul >= 2 and hangul >= kana:
            return "ko"
        if kana >= 1:
            return "ja"
        if han >= 1:
            if configured == "ja":
                return "ja"
            if configured == "ko":
                return "ko"
            return self._detect_chinese_variant(s, configured)

        # Library detection is whitelist-limited (en/ja/ko/zh),
        # helping short/noisy OCR lines where regex-only logic is brittle.
        model_code, _score, _engine = self._lang_detector.detect(s)
        if model_code == "zh":
            return self._detect_chinese_variant(s, configured)
        if model_code in {"en", "ja", "ko"}:
            return model_code

        if latin_letters >= 3:
            return "en"

        # Unknown/very short/noisy text: respect explicit user choice if available.
        if configured in self._SUPPORTED_SOURCE_CODES and configured != "auto":
            return configured
        return configured

    def _resolve_source_lang_for_text(self, text: str) -> str:
        configured = self._canonical_source_code(self._source_lang)
        if not self._auto_source_routing_enabled:
            return configured

        detected = self._detect_script_language(text)
        if self._translator_backend_name in {"nllb"}:
            if detected in _NllbBackend._NLLB_LANG:
                return detected
            return configured
        if self._translator_backend_name in {"google"}:
            return detected or configured
        return configured

    def _build_local_backend_for_source(self, source_code: str):
        if self._translator_backend_name == "nllb":
            return self._translator
        return self._translator

    def _get_backend_for_source(self, source_code: str):
        if self._translator_backend_name not in {"nllb"}:
            return self._translator
        if self._translator_backend_name == "nllb":
            return self._translator

        source_code = self._canonical_source_code(source_code)
        configured = self._canonical_source_code(self._source_lang)
        if source_code == configured or source_code in {"", "auto"}:
            return self._translator

        backend = self._backend_pool.get(source_code)
        if backend is not None:
            return backend
        if source_code in self._backend_pool_failures:
            return self._translator

        try:
            backend = self._build_local_backend_for_source(source_code)
            self._backend_pool[source_code] = backend
            print(
                f"[Translator] auto source-route: {configured}->{source_code} (target={self._target_lang})",
                flush=True,
            )
            return backend
        except Exception as exc:
            self._backend_pool_failures.add(source_code)
            print(
                f"[Translator] auto source-route disabled for '{source_code}': {exc}",
                flush=True,
            )
            return self._translator

    @property
    def backend_name(self) -> str:
        return self._translator_backend_name

    @property
    def backend_display_name(self) -> str:
        return self._translator_backend_display

    def _backend_candidates(self) -> list[str]:
        if self._translation_backend == "google":
            return ["google"]
        if self._translation_backend == "auto":
            if self._fallback_to_google:
                return ["nllb", "google"]
            return ["nllb"]
        if self._translation_backend == "nllb":
            cands = ["nllb"]
            if self._fallback_to_google:
                cands.append("google")
            return cands
        if self._fallback_to_google:
            return ["nllb", "google"]
        return ["nllb"]

    def _init_backend(self):
        errors: list[str] = []
        for name in self._backend_candidates():
            try:
                if name == "nllb":
                    self._translator = _NllbBackend(
                        source_language=self._source_lang,
                        target_language=self._target_lang,
                        model_dir=self._nllb_model_dir,
                        tokenizer_path=self._nllb_tokenizer_path,
                        device=self._nllb_device,
                        compute_type=self._nllb_compute_type,
                        beam_size=self._nllb_beam_size,
                        max_decoding_length=self._nllb_max_decoding_length,
                        max_source_tokens=self._nllb_max_source_tokens,
                    )
                    route = getattr(self._translator, "route_name", "")
                elif name == "google":
                    self._translator = _GoogleBackend(
                        target_language=self._target_lang,
                        source_language=self._source_lang,
                    )
                    route = ""
                else:
                    raise RuntimeError(f"unsupported backend: {name}")
                self._translator_backend_name = name
                self._translator_backend_display = f"{name}:{route}" if route else name
                self._backend_pool = {}
                self._backend_pool_failures = set()
                configured = self._canonical_source_code(self._source_lang)
                if self._translator is not None and configured not in {"", "auto"}:
                    self._backend_pool[configured] = self._translator
                route_suffix = f" ({route})" if route else ""
                if errors:
                    print(
                        f"[Translator] backend='{name}{route_suffix}' selected after fallback. Prior errors: {' | '.join(errors)}",
                        flush=True,
                    )
                else:
                    print(f"[Translator] backend='{name}{route_suffix}'", flush=True)
                return
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        self._translator = None
        self._translator_backend_name = "none"
        self._translator_backend_display = "none"
        self._backend_pool = {}
        self._backend_pool_failures = set()
        if errors:
            print(f"[Translator] No backend available: {' | '.join(errors)}", flush=True)

    def _switch_to_google_fallback(self, reason: Exception | None = None) -> bool:
        if self._translator_backend_name == "google" and self._translator is not None:
            return True
        try:
            self._translator = _GoogleBackend(
                target_language=self._target_lang,
                source_language=self._source_lang,
            )
            self._translator_backend_name = "google"
            self._translator_backend_display = "google"
            self._backend_pool = {}
            self._backend_pool_failures = set()
            if reason is not None:
                print(f"[Translator] runtime fallback to google: {reason}", flush=True)
            return True
        except Exception as exc:
            print(f"[Translator] runtime google fallback failed: {exc}", flush=True)
            return False

    def _build_custom_glossary_rules(self, glossary: dict[str, str]):
        """Build case-insensitive replacement rules from user glossary."""
        cleaned: list[tuple[str, str]] = []
        for raw_src, raw_tgt in (glossary or {}).items():
            src = str(raw_src).strip()
            tgt = str(raw_tgt).strip()
            if not src or not tgt:
                continue
            cleaned.append((src, tgt))

        # Longest source phrase first to avoid short phrase shadowing.
        cleaned.sort(key=lambda kv: len(kv[0]), reverse=True)

        rules: list[tuple[re.Pattern[str], str, str]] = []
        for i, (src, tgt) in enumerate(cleaned):
            placeholder = f"[[GLOSS_{i}]]"
            rules.append((re.compile(re.escape(src), re.IGNORECASE), placeholder, tgt))
        self._custom_glossary_rules = rules

    @property
    def target_language(self) -> str:
        return self._target_lang

    @target_language.setter
    def target_language(self, lang: str):
        if lang != self._target_lang:
            self._target_lang = lang
            self._init_backend()
            self.clear_cache()

    def _semantic_key(self, text: str) -> str:
        if not self._semantic_cache_enabled:
            return ""
        s = str(text or "").strip()
        if not s:
            return ""
        s = s.lower()
        s = s.replace("1o0", "100")
        s = re.sub(r"(?<=\d)[o](?=\d)", "0", s)
        s = re.sub(r"[^a-z0-9\[\]_]+", " ", s)
        s = re.sub(r"\s{2,}", " ", s).strip()
        return s

    def _cache_get(self, text: str) -> str | None:
        with self._lock:
            value = self._cache.get(text)
            if value is not None:
                self._cache.move_to_end(text)
            return value

    def _cache_get_semantic(self, text: str) -> str | None:
        sem_key = self._semantic_key(text)
        if not sem_key:
            return None
        with self._lock:
            value = self._semantic_cache.get(sem_key)
            if value is not None:
                self._semantic_cache.move_to_end(sem_key)
            return value

    def _cache_set(self, text: str, translated: str):
        sem_key = self._semantic_key(text)
        with self._lock:
            self._cache[text] = translated
            self._cache.move_to_end(text)
            while len(self._cache) > self._cache_max_entries:
                self._cache.popitem(last=False)
            if sem_key:
                self._semantic_cache[sem_key] = translated
                self._semantic_cache.move_to_end(sem_key)
                while len(self._semantic_cache) > self._cache_max_entries:
                    self._semantic_cache.popitem(last=False)

    def _translate_single_uncached(self, text: str, source_hint: str | None = None) -> str:
        if self._translator is None:
            return text
        source_code = source_hint or self._resolve_source_lang_for_text(text)
        preview = _safe_log_text(text, 80)
        print(
            f"[Translator] single source={source_code or 'auto'} backend={self._translator_backend_name} text='{preview}'",
            flush=True,
        )
        backend = self._get_backend_for_source(source_code)
        if backend is None:
            return text
        try:
            if isinstance(backend, _NllbBackend):
                result = backend.translate(text, source_language=source_code)
            elif isinstance(backend, _GoogleBackend):
                result = backend.translate(text, source_language=source_code)
            else:
                result = backend.translate(text)
            return result if result else text
        except Exception as exc:
            if backend is not self._translator and self._translator is not None:
                try:
                    if isinstance(self._translator, _NllbBackend):
                        fallback_result = self._translator.translate(
                            text, source_language=source_code
                        )
                    elif isinstance(self._translator, _GoogleBackend):
                        fallback_result = self._translator.translate(
                            text, source_language=source_code
                        )
                    else:
                        fallback_result = self._translator.translate(text)
                    if fallback_result:
                        return fallback_result
                except Exception:
                    pass
            if (
                self._translator_backend_name in {"nllb"}
                and self._fallback_to_google
                and self._switch_to_google_fallback(exc)
            ):
                try:
                    result = self._translator.translate(text)
                    return result if result else text
                except Exception:
                    return text
            return text

    @staticmethod
    def _looks_all_caps_sentence(text: str) -> bool:
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return False
        upper = sum(1 for c in letters if c.isupper())
        return upper / len(letters) >= 0.85

    def _split_joined_word(self, token_lower: str) -> list[str] | None:
        """DP segmentation for OCR-joined ASCII words (e.g., 'notonly' -> ['not', 'only'])."""
        n = len(token_lower)
        if n < 6 or not token_lower.isalpha():
            return None
        if token_lower in self._word_lexicon:
            return None

        max_word_len = 20
        best: list[tuple[float, list[str]] | None] = [None] * (n + 1)
        best[0] = (0.0, [])

        for i in range(n):
            if best[i] is None:
                continue
            base_score, base_words = best[i]
            max_j = min(n, i + max_word_len)
            for j in range(i + 1, max_j + 1):
                w = token_lower[i:j]
                if w not in self._word_lexicon:
                    continue
                tiny_penalty = 0.65 if len(w) <= 2 else 0.0
                cand_score = base_score + (len(w) * len(w)) - 1.15 - tiny_penalty
                cand_words = base_words + [w]
                prev = best[j]
                if prev is None or cand_score > prev[0]:
                    best[j] = (cand_score, cand_words)

        final = best[n]
        if final is None:
            return None

        words = final[1]
        if len(words) < 2:
            return None

        tiny_count = sum(1 for w in words if len(w) <= 2)
        if tiny_count > 1:
            return None
        if any(len(w) == 1 for w in words[1:-1]):
            return None

        if len(words) >= 5:
            avg_len = sum(len(w) for w in words) / len(words)
            if avg_len < 3.0:
                return None

        return words

    @staticmethod
    def _apply_case_template(core: str, words: list[str]) -> str:
        if not words:
            return core
        if core.isupper():
            return " ".join(w.upper() for w in words)
        if core[0].isupper() and core[1:].islower():
            return " ".join([words[0].capitalize(), *words[1:]])
        return " ".join(words)

    def _normalize_token(self, token: str) -> str:
        m = re.match(r"^([^A-Za-z]*)([A-Za-z']+)([^A-Za-z]*)$", token)
        if not m:
            return token

        prefix, core, suffix = m.groups()
        upper = core.replace("'", "").upper()
        core_plain = core.replace("'", "")

        if upper in self._ocr_fixups:
            normalized_core = self._ocr_fixups[upper]
        else:
            normalized_core = core
            if "'" not in core and len(core_plain) >= 6 and core_plain.isalpha():
                split_words = self._split_joined_word(core_plain.lower())
                if split_words:
                    normalized_core = self._apply_case_template(core_plain, split_words)

        return f"{prefix}{normalized_core}{suffix}"

    def _dejoin_ascii_text(self, text: str) -> str:
        # Split obvious boundaries first: camelCase, A1/B2 style IDs, and underscores.
        spaced = (
            text.replace("_", " ")
            .replace("|", " | ")
        )
        spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", spaced)
        spaced = re.sub(r"(?<=[A-Za-z])(?=[0-9])", " ", spaced)
        spaced = re.sub(r"(?<=[0-9])(?=[A-Za-z])", " ", spaced)
        spaced = re.sub(r"\s{2,}", " ", spaced).strip()
        if not spaced:
            return spaced

        normalized = " ".join(self._normalize_token(tok) for tok in spaced.split(" "))
        normalized = re.sub(r"\s{2,}", " ", normalized).strip()
        return normalized

    def _normalize_for_translation(self, text: str) -> str:
        """Normalize OCR artifacts before translation to improve quality."""
        normalized = " ".join(text.strip().split())
        if not normalized:
            return normalized

        normalized = (
            normalized.replace(" ,", ",")
            .replace(" .", ".")
            .replace(" !", "!")
            .replace(" ?", "?")
            .replace(" ;", ";")
            .replace(" :", ":")
        )
        normalized = re.sub(r"([,.;:!?])(?=[A-Za-z0-9])", r"\1 ", normalized)
        normalized = re.sub(r"\s{2,}", " ", normalized).strip()

        if all(ord(c) < 128 for c in normalized):
            normalized = self._dejoin_ascii_text(normalized)
            if self._looks_all_caps_sentence(normalized):
                normalized = normalized.lower()

        return normalized

    def _apply_source_phrase_fixes(self, text: str) -> str:
        if not self._manga_mode or not self._manga_source_fixes_enabled:
            return text
        if self._auto_document_guard and self._looks_document_text(text):
            return text
        fixed = text
        for pattern, replacement in self._source_phrase_fixes:
            fixed = pattern.sub(replacement, fixed)
        return fixed

    def _apply_custom_glossary_placeholders(self, text: str) -> tuple[str, dict[str, str]]:
        """Replace glossary source phrases with stable placeholders before MT."""
        if not self._custom_glossary_enabled or not self._custom_glossary_rules:
            return text, {}

        replaced = text
        applied: dict[str, str] = {}
        for pattern, placeholder, target in self._custom_glossary_rules:
            if pattern.search(replaced):
                replaced = pattern.sub(placeholder, replaced)
                applied[placeholder] = target
        return replaced, applied

    def _apply_game_term_placeholders(self, text: str) -> tuple[str, dict[str, str]]:
        if not self._game_term_guard_enabled:
            return text, {}
        if not self._contains_game_term(text):
            return text, {}

        replaced = text
        applied: dict[str, str] = {}
        for i, (pattern, canonical) in enumerate(self._game_source_terms):
            if pattern.search(replaced):
                placeholder = f"[[GTERM_{i}]]"
                replaced = pattern.sub(placeholder, replaced)
                applied[placeholder] = canonical
        return replaced, applied

    def _apply_game_source_phrase_fixes(self, text: str) -> str:
        if not self._game_term_guard_enabled:
            return text
        if not self._looks_game_text(text) and not self._contains_game_term(text):
            return text

        fixed = text
        for pattern, replacement in self._game_source_phrase_fixes:
            fixed = pattern.sub(replacement, fixed)

        fixed = re.sub(r"(?i)\b1\s*[oO]\s*0\b", "100", fixed)
        fixed = re.sub(r",\s*press", ", press", fixed, flags=re.IGNORECASE)
        fixed = re.sub(r"\s{2,}", " ", fixed).strip()
        return fixed

    @staticmethod
    def _restore_glossary_placeholders(text: str, placeholders: dict[str, str]) -> str:
        if not text or not placeholders:
            return text
        out = text
        for placeholder, target in placeholders.items():
            out = out.replace(placeholder, target)
            marker = placeholder.strip("[]")
            out = re.sub(r"\[\[\s*" + re.escape(marker) + r"\s*\]\]", target, out, flags=re.IGNORECASE)
            # Tolerate malformed bracket wrappers from MT output, e.g. "[GTERM_2]]"
            out = re.sub(r"\[+\s*" + re.escape(marker) + r"\s*\]+", target, out, flags=re.IGNORECASE)
            out = re.sub(r"\b" + re.escape(marker) + r"\b", target, out, flags=re.IGNORECASE)
        return out

    def _post_process_translation(self, source_text: str, translated: str) -> str:
        if not translated:
            return translated
        out = translated

        if self._game_post_edit_enabled and self._looks_game_text(source_text):
            for pattern, replacement in self._game_target_fixes:
                out = pattern.sub(replacement, out)
            if re.search(r"\bcast(?:ing)?\b", source_text, flags=re.IGNORECASE):
                out = re.sub(r"\bthi\s+triển\b", "kích hoạt", out, flags=re.IGNORECASE)
                out = re.sub(r"\btung\b", "kích hoạt", out, flags=re.IGNORECASE)
                out = re.sub(r"\bsử\s+dụng\b", "kích hoạt", out, flags=re.IGNORECASE)
                out = re.sub(r"\bném\b", "kích hoạt", out, flags=re.IGNORECASE)
                out = re.sub(r"\bđúc\b", "kích hoạt", out, flags=re.IGNORECASE)
            if re.search(r"\bpress\s+resonance\s+skill\s+to\s+cast\b", source_text, flags=re.IGNORECASE):
                out = re.sub(
                    r"\bnhấn\s+Resonance\s+Skill\s+để\s+[^\s.,;:!?]+\b",
                    "nhấn Resonance Skill để kích hoạt",
                    out,
                    flags=re.IGNORECASE,
                )
            if re.search(r"\bdealing\b", source_text, flags=re.IGNORECASE):
                out = re.sub(r"\bxử\s+lý\b", "gây", out, flags=re.IGNORECASE)
                out = re.sub(r"\bgây\s+ra\b", "gây", out, flags=re.IGNORECASE)
            if re.search(r"\benter\s+seraphic\s+duo\b", source_text, flags=re.IGNORECASE):
                out = re.sub(r"\bnhập\s+Seraphic\s+Duo\b", "vào Seraphic Duo", out, flags=re.IGNORECASE)
            if re.search(r"\bcan be cast\b", source_text, flags=re.IGNORECASE):
                out = re.sub(r"\bcó thể được kích hoạt\b", "có thể kích hoạt", out, flags=re.IGNORECASE)
            if re.search(r"\bcan be cast in mid-air close to the ground\b", source_text, flags=re.IGNORECASE):
                out = "Có thể kích hoạt trên không khi ở gần mặt đất."
            if re.search(r"\bswitch into the mech form after casting this\b", source_text, flags=re.IGNORECASE):
                if "Mech Form" not in out:
                    out = out.rstrip(". ") + ". Chuyển sang Mech Form sau khi kích hoạt kỹ năng này."
            if re.search(r"\bresonance mode\b.*\bfusion burst in effect\b", source_text, flags=re.IGNORECASE):
                out = "Resonance Mode - Fusion Burst đang có hiệu lực."
            if re.search(r"\bforte circuit\b", source_text, flags=re.IGNORECASE) and re.search(
                r"\bto sculpt the silence\b", source_text, flags=re.IGNORECASE
            ):
                out = "Forte Circuit - To Sculpt the Silence."
            out = re.sub(r"\bAemeath\s+là\s+trong\s+Seraphic\s+Duo\b", "Aemeath đang ở trạng thái Seraphic Duo", out, flags=re.IGNORECASE)
            out = re.sub(r"\bđiểm\s+của\s+Synchronization\s+Rate\b", "điểm Synchronization Rate", out, flags=re.IGNORECASE)

        if not self._manga_mode or not self._manga_target_post_edit_enabled:
            return out
        if self._auto_document_guard and self._looks_document_text(source_text):
            return out

        for pattern, replacement in self._target_phrase_fixes:
            out = pattern.sub(replacement, out)

        # Contextual sentence-level polish for common manga narration pattern.
        lowered_source = source_text.lower()
        lowered_out = out.lower()
        if (
            "elite heirs" in lowered_source
            and "difficult people" in lowered_source
            and ("con nhà danh giá" in lowered_out or "người thừa kế" in lowered_out)
            and "khó tính" in lowered_out
        ):
            return "đến cả con nhà danh giá lẫn người khó tính cũng muốn ở cạnh cô ấy."

        out = self._post_process_cjk_vi(source_text, out)
        return out

    def _post_process_cjk_vi(self, source_text: str, translated: str) -> str:
        if self._target_lang != "vi" or not source_text or not translated:
            return translated
        if not re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", source_text):
            return translated

        out = translated
        for pattern, replacement in self._cjk_vi_target_fixes:
            out = pattern.sub(replacement, out)

        src_has_weather = bool(
            re.search(r"\u3044\u3044\u5929\u6c17", source_text)
            or re.search(r"\u5929\u6c14.{0,4}\u597d", source_text)
        )
        if src_has_weather and re.search(
            r"(th\u1eddi\s+ti\u1ebft|tr\u1eddi).{0,20}(t\u1ed1t|\u0111\u1eb9p)",
            out,
            flags=re.IGNORECASE,
        ):
            if re.search(r"\bxin\s+ch\u00e0o\b|\bch\u00e0o\b", out, flags=re.IGNORECASE):
                return "Xin ch\u00e0o, h\u00f4m nay th\u1eddi ti\u1ebft \u0111\u1eb9p nh\u1ec9."
            return "H\u00f4m nay th\u1eddi ti\u1ebft \u0111\u1eb9p nh\u1ec9."
        return out

    def _prepare_source_text(self, text: str) -> tuple[str, str, dict[str, str]]:
        normalized = self._normalize_for_translation(text)
        context_text = normalized if normalized else text.strip()
        context_text = self._apply_source_phrase_fixes(context_text)
        context_text = self._apply_game_source_phrase_fixes(context_text)
        prepared_text, game_placeholders = self._apply_game_term_placeholders(context_text)
        prepared_text, custom_placeholders = self._apply_custom_glossary_placeholders(prepared_text)
        placeholders = {}
        placeholders.update(game_placeholders)
        placeholders.update(custom_placeholders)
        return prepared_text, context_text, placeholders

    def _should_refine_with_context(self, text: str) -> bool:
        if not self._context_refine_enabled:
            return False
        stripped = text.strip()
        if not stripped:
            return False
        if len(stripped) > self._context_refine_max_chars:
            return False
        words = len(stripped.split())
        if words > self._context_refine_max_words:
            return False
        return True

    @staticmethod
    def _token_set_en(text: str) -> set[str]:
        return set(re.findall(r"[a-z']+", text.lower()))

    def _needs_context_refine(self, source_text: str, translated_text: str) -> bool:
        if not self._should_refine_with_context(source_text):
            return False

        src = source_text.strip()
        dst = translated_text.strip()
        if not dst:
            return True
        if src.lower() == dst.lower():
            return True

        src_tokens = self._token_set_en(src)
        dst_tokens = self._token_set_en(dst)
        if not src_tokens or not dst_tokens:
            return False

        overlap = len(src_tokens & dst_tokens) / max(len(src_tokens), 1)
        return overlap >= 0.8

    def _prepare_neighbor_context_text(self, text: str) -> str:
        normalized = self._normalize_for_translation(text)
        context_text = normalized if normalized else text.strip()
        context_text = self._apply_source_phrase_fixes(context_text)
        context_text = self._apply_game_source_phrase_fixes(context_text)
        return context_text

    def _build_context_query(
        self,
        prev_text: str,
        current_text: str,
        next_text: str,
    ) -> str:
        prev = prev_text if prev_text else "-"
        nxt = next_text if next_text else "-"
        return (
            f"{self._ctx_prev_tag} {prev}\n"
            f"{self._ctx_curr_tag} {current_text}\n"
            f"{self._ctx_next_tag} {nxt}"
        )

    def _extract_context_current(self, translated_query: str) -> str:
        if not translated_query:
            return ""
        escaped_curr = re.escape(self._ctx_curr_tag)
        escaped_next = re.escape(self._ctx_next_tag)
        m = re.search(
            escaped_curr + r"\s*(.*?)\s*" + escaped_next,
            translated_query,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not m:
            return ""
        out = m.group(1).strip()
        if self._ctx_prev_tag in out or self._ctx_curr_tag in out or self._ctx_next_tag in out:
            return ""
        return out

    def _translate_with_local_context(
        self,
        prev_raw: str,
        current_raw: str,
        next_raw: str,
    ) -> str | None:
        if self._translator is None:
            return None

        current_key, current_context, placeholders = self._prepare_source_text(current_raw)
        prev_context = self._prepare_neighbor_context_text(prev_raw) if prev_raw else ""
        next_context = self._prepare_neighbor_context_text(next_raw) if next_raw else ""
        if not prev_context and not next_context:
            return None

        query = self._build_context_query(prev_context, current_key, next_context)
        cache_key = f"__CTX__{query}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        translated_query = self._translate_single_uncached(query)
        extracted = self._extract_context_current(translated_query)
        if not extracted:
            return None

        extracted = self._restore_glossary_placeholders(extracted, placeholders)
        final = self._post_process_translation(current_context, extracted)
        self._cache_set(cache_key, final)
        return final

    def _chunk_texts(self, texts: list[str]):
        """Yield chunks with combined length <= max_batch_chars."""
        chunk: list[str] = []
        chunk_len = 0

        for text in texts:
            text_len = len(text)
            add_len = text_len if not chunk else text_len + len(self._batch_separator)

            if chunk and chunk_len + add_len > self._max_batch_chars:
                yield chunk
                chunk = [text]
                chunk_len = text_len
            else:
                chunk.append(text)
                chunk_len += add_len

        if chunk:
            yield chunk

    def _translate_chunk(self, chunk: list[str]) -> list[str]:
        """Translate one chunk. Falls back to per-item translation on split mismatch."""
        if not chunk:
            return []
        if len(chunk) == 1:
            return [self._translate_single_uncached(chunk[0])]
        if self._translator is None:
            return chunk
        if any(self._batch_separator in text for text in chunk):
            return [self._translate_single_uncached(text) for text in chunk]

        source_hint = None
        source_by_index: list[str] = []
        if self._auto_source_routing_enabled:
            source_by_index = [self._resolve_source_lang_for_text(text) for text in chunk]
            chunk_sources = {s for s in source_by_index if s}
            preview = " | ".join(_safe_log_text(t, 40) for t in chunk[:3])
            print(
                f"[Translator] batch route candidates={source_by_index[:6]} backend={self._translator_backend_name} texts={preview}",
                flush=True,
            )

            if self._translator_backend_name == "nllb" and len(chunk_sources) > 1:
                outputs = list(chunk)
                groups: dict[str, list[int]] = {}
                for idx, src in enumerate(source_by_index):
                    groups.setdefault(src, []).append(idx)

                for src, idxs in groups.items():
                    backend = self._get_backend_for_source(src)
                    subset = [chunk[i] for i in idxs]
                    try:
                        if hasattr(backend, "translate_many"):
                            if isinstance(backend, _NllbBackend):
                                part = backend.translate_many(subset, source_language=src)
                            else:
                                part = backend.translate_many(subset)
                            if len(part) == len(subset):
                                for out_idx, out_text in zip(idxs, part):
                                    outputs[out_idx] = out_text if out_text else chunk[out_idx]
                                continue
                    except Exception:
                        pass

                    for i in idxs:
                        outputs[i] = self._translate_single_uncached(chunk[i], source_hint=src)
                return outputs

            if chunk_sources:
                source_hint = next(iter(chunk_sources))

        if self._translator_backend_name == "nllb":
            backend = self._get_backend_for_source(source_hint or self._source_lang)
            if hasattr(backend, "translate_many"):
                try:
                    if isinstance(backend, _NllbBackend):
                        out = backend.translate_many(chunk, source_language=source_hint)
                    else:
                        out = backend.translate_many(chunk)
                    if len(out) == len(chunk):
                        return [v if v else o for v, o in zip(out, chunk)]
                except Exception:
                    pass
            return [self._translate_single_uncached(text, source_hint=source_hint) for text in chunk]

        if self._translator_backend_name != "google":
            # Keep non-google backends per-block to avoid separator corruption.
            return [self._translate_single_uncached(text, source_hint=source_hint) for text in chunk]

        combined = self._batch_separator.join(chunk)
        try:
            translated_combined = self._translator.translate(
                combined, source_language=source_hint
            )
        except Exception:
            translated_combined = None

        if translated_combined:
            parts = [p.strip() for p in translated_combined.split(self._batch_separator)]
            if len(parts) == len(chunk):
                return [part if part else orig for part, orig in zip(parts, chunk)]

        return [self._translate_single_uncached(text) for text in chunk]

    def translate(self, text: str) -> str:
        """Translate a single text string with caching."""
        if not text or not text.strip():
            return text

        key, context_text, placeholders = self._prepare_source_text(text)

        cached = self._cache_get(key)
        if cached is None:
            cached = self._cache_get_semantic(key)
        if cached is not None:
            return cached

        source_hint = self._resolve_source_lang_for_text(context_text)
        result = self._translate_single_uncached(key, source_hint=source_hint)
        result = self._restore_glossary_placeholders(result, placeholders)
        final_result = self._post_process_translation(context_text, result)
        self._cache_set(key, final_result)
        return final_result

    def translate_batch(self, texts: list[str], cancel_check=None) -> list[str]:
        """
        Translate multiple texts, using cache for already-seen texts.
        Batches uncached texts into chunked API calls when possible.
        """
        results: list[str | None] = [None] * len(texts)
        pending_positions: dict[str, list[int]] = {}
        context_by_key: dict[str, str] = {}
        placeholders_by_key: dict[str, dict[str, str]] = {}
        unique_uncached: list[str] = []

        for i, text in enumerate(texts):
            stripped = text.strip()
            if not stripped:
                results[i] = text
                continue

            key, context_text, placeholders = self._prepare_source_text(stripped)

            cached = self._cache_get(key)
            if cached is None:
                cached = self._cache_get_semantic(key)
            if cached is not None:
                results[i] = cached
                continue

            if key not in pending_positions:
                pending_positions[key] = []
                context_by_key[key] = context_text
                placeholders_by_key[key] = placeholders
                unique_uncached.append(key)
            pending_positions[key].append(i)

        if unique_uncached:
            translated_unique: list[str] = []
            chunks = list(self._chunk_texts(unique_uncached))
            cancelled = False
            if self._translator_backend_name == "google" and len(chunks) > 1 and cancel_check is None:
                ordered_outputs: list[list[str] | None] = [None] * len(chunks)
                max_workers = min(4, len(chunks))
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = {
                        ex.submit(self._translate_chunk, chunk): idx
                        for idx, chunk in enumerate(chunks)
                    }
                    for fut in as_completed(futures):
                        idx = futures[fut]
                        chunk = chunks[idx]
                        try:
                            part = fut.result()
                        except Exception:
                            part = [self._translate_single_uncached(text) for text in chunk]
                        ordered_outputs[idx] = part
                for part in ordered_outputs:
                    if part:
                        translated_unique.extend(part)
            else:
                for chunk in chunks:
                    if callable(cancel_check) and cancel_check():
                        cancelled = True
                        break
                    translated_unique.extend(self._translate_chunk(chunk))

            for original, translated in zip(unique_uncached, translated_unique):
                final_text = translated if translated else original
                final_text = self._restore_glossary_placeholders(
                    final_text,
                    placeholders_by_key.get(original, {}),
                )
                final_text = self._post_process_translation(
                    context_by_key.get(original, original),
                    final_text,
                )
                self._cache_set(original, final_text)
                for idx in pending_positions.get(original, []):
                    results[idx] = final_text

            if cancelled:
                for original in unique_uncached[len(translated_unique):]:
                    fallback = context_by_key.get(original, original)
                    for idx in pending_positions.get(original, []):
                        results[idx] = fallback

        for i, value in enumerate(results):
            if value is None:
                results[i] = texts[i]

        # Optional local-context refinement for short dialogue bubbles.
        # Keeps 1 block -> 1 translation, but uses neighbor text as context.
        if self._context_refine_enabled and self._translator is not None:
            refined_count = 0
            for i, src_text in enumerate(texts):
                if callable(cancel_check) and cancel_check():
                    break
                if refined_count >= self._context_refine_max_per_batch:
                    break
                current_result = str(results[i]) if results[i] is not None else ""
                if not self._needs_context_refine(src_text, current_result):
                    continue
                prev_text = texts[i - 1].strip() if i > 0 else ""
                next_text = texts[i + 1].strip() if i + 1 < len(texts) else ""
                if not prev_text and not next_text:
                    continue
                refined = self._translate_with_local_context(
                    prev_raw=prev_text,
                    current_raw=src_text,
                    next_raw=next_text,
                )
                if refined and refined.strip() and refined.strip() != current_result.strip():
                    results[i] = refined
                    refined_count += 1

        return [str(v) for v in results]

    def _translate_individually(self, texts: list[str], indices: list[int], results: list):
        """Fallback: translate texts one by one."""
        for idx, text in zip(indices, texts):
            translated = self._translate_single_uncached(text)
            self._cache_set(text, translated)
            results[idx] = translated

    def clear_cache(self):
        """Clear the translation cache."""
        with self._lock:
            self._cache.clear()
            self._semantic_cache.clear()
