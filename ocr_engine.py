# ocr_engine.py - multi-backend OCR engine (EasyOCR, WinRT, RapidOCR, PaddleOCR)
import warnings
# suppress torch pin_memory warning when no GPU is available
warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning)

from dataclasses import dataclass
import sys
import asyncio
import numpy as np


# mapping from language code to EasyOCR language code(s)
# only load the exact language needed — no English combo (saves 600ms)
EASYOCR_LANG_MAP = {
    "en": ["en"],
    "ko": ["ko"],      # ko-only: ~839ms vs ko+en: ~1822ms
    "ja": ["ja"],
    "zh-CN": ["ch_sim"],
    "zh-TW": ["ch_tra"],
    "vi": ["en"],
    "fr": ["fr"],
    "de": ["de"],
    "es": ["es"],
    "th": ["th"],
    "ru": ["ru"],
    "pt": ["pt"],
    "it": ["it"],
    "ar": ["ar"],
}

# Backend name constants
BACKEND_RAPIDOCR = "rapidocr"
BACKEND_PADDLEOCR = "paddleocr"
BACKEND_WINRT = "winrt"
BACKEND_EASYOCR = "easyocr"
BACKEND_NONE = "none"


@dataclass
class TextBlock:
    """Represents a detected text block with position."""
    text: str
    bbox: list  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] - four corners
    confidence: float

    @property
    def x(self) -> int:
        """Left coordinate."""
        return int(min(p[0] for p in self.bbox))

    @property
    def y(self) -> int:
        """Top coordinate."""
        return int(min(p[1] for p in self.bbox))

    @property
    def width(self) -> int:
        """Width of bounding box."""
        return int(max(p[0] for p in self.bbox) - self.x)

    @property
    def height(self) -> int:
        """Height of bounding box."""
        return int(max(p[1] for p in self.bbox) - self.y)

    @property
    def center_y(self) -> float:
        """Vertical center of bounding box."""
        return self.y + self.height / 2.0


class _BaseEngine:
    """Abstract base class for different OCR backends."""

    def detect(self, image: np.ndarray, min_confidence: float) -> list[TextBlock]:
        raise NotImplementedError


def _merge_group(group: list["TextBlock"]) -> "TextBlock":
    """Merge a list of TextBlocks on the same line into a single block.

    Blocks are sorted left-to-right. Spaces are inserted only where the
    visual gap between adjacent blocks is > 0.3 × avg char width.
    """
    group.sort(key=lambda b: b.bbox[0][0])

    parts: list[str] = []
    for i, g in enumerate(group):
        parts.append(g.text)
        if i < len(group) - 1:
            right_edge = max(p[0] for p in g.bbox)
            left_edge  = min(p[0] for p in group[i + 1].bbox)
            gap = left_edge - right_edge
            avg_char_w = g.width / max(len(g.text), 1)
            if gap > avg_char_w * 0.3:
                parts.append(" ")

    all_x = [p[0] for g in group for p in g.bbox]
    all_y = [p[1] for g in group for p in g.bbox]
    return TextBlock(
        text="".join(parts),
        bbox=[
            [min(all_x), min(all_y)],
            [max(all_x), min(all_y)],
            [max(all_x), max(all_y)],
            [min(all_x), max(all_y)],
        ],
        confidence=min(g.confidence for g in group),
    )


def merge_same_line_blocks(blocks: list["TextBlock"]) -> list["TextBlock"]:
    """Merge OCR blocks that share the same visual line — O(n log n).

    Algorithm:
    1. Sort blocks by center_y  → O(n log n)
    2. Single linear pass: compare each block only to the *last* block in
       the current group → O(n)
    Total: O(n log n) vs the previous O(n²) nested-loop approach.
    """
    if not blocks:
        return blocks

    avg_h = sum(b.height for b in blocks) / len(blocks)
    threshold = avg_h * 0.5

    # Sort once — avoids the O(n²) nested comparison
    sorted_blocks = sorted(blocks, key=lambda b: b.center_y)

    lines: list[TextBlock] = []
    current_group: list[TextBlock] = [sorted_blocks[0]]

    for block in sorted_blocks[1:]:
        # Compare only against the anchor (first block of current group)
        if abs(block.center_y - current_group[0].center_y) <= threshold:
            current_group.append(block)
        else:
            lines.append(_merge_group(current_group))
            current_group = [block]

    lines.append(_merge_group(current_group))

    # Re-sort merged lines top-to-bottom
    lines.sort(key=lambda b: b.y)
    return lines


class _EasyOCREngine(_BaseEngine):
    """Wrapper around EasyOCR (original engine)."""

    def __init__(self, lang: str):
        self._lang = lang
        self._reader = None
        self._loaded_lang = None
        self._init_failed = False
        self._OCR_SCALE = 0.5  # downscale factor

    def _ensure_reader(self):
        if self._init_failed:
            return
        if self._reader is not None and self._loaded_lang == self._lang:
            return
        codes = EASYOCR_LANG_MAP.get(self._lang, ["en"])
        try:
            import easyocr
            self._reader = easyocr.Reader(codes, gpu=False, verbose=False)
            self._loaded_lang = self._lang
            print(f"[OCR] Loaded EasyOCR ({codes})", flush=True)
        except Exception as e:
            self._init_failed = True
            msg = str(e)
            print(f"[OCR] EasyOCR init failed: {msg}", flush=True)
            # common DLL problem
            if "c10.dll" in msg or "DLL load failed" in msg:
                print("[OCR]  * Torch/CPU issue detected – try installing CPU-only wheel:\n" +
                      "    pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu",
                      flush=True)

    def detect(self, image: np.ndarray, min_confidence: float = 0.3) -> list[TextBlock]:
        self._ensure_reader()
        if self._reader is None:
            return []
        try:
            small = image[::2, ::2, :]
            results = self._reader.readtext(small)
        except Exception as e:
            print(f"[OCR] EasyOCR error: {e}", flush=True)
            return []
        if not results:
            return []
        inv = 1.0 / self._OCR_SCALE
        blocks: list[TextBlock] = []
        for bbox, text, conf in results:
            if conf >= min_confidence and text.strip():
                scaled = [[p[0] * inv, p[1] * inv] for p in bbox]
                blocks.append(TextBlock(text=text.strip(), bbox=scaled, confidence=conf))
        blocks.sort(key=lambda b: (b.y, b.x))
        return blocks


class _WinRTEngine(_BaseEngine):
    """OCR using Windows Runtime (WinRT) API. Very fast when language pack exists."""

    def __init__(self, lang: str):
        self._lang_tag = lang
        self._engine = None
        self._imaging = None
        self._streams = None
        self._ocr = None
        self._init_failed = False
        self._init()  # try to initialize immediately

    def _init(self):
        if sys.platform != "win32":
            return
        try:
            import winrt.windows.media.ocr as winrt_ocr
            import winrt.windows.graphics.imaging as imaging
            import winrt.windows.storage.streams as streams
            from winrt.windows.globalization import Language
        except ImportError:
            # WinRT not installed at all
            return

        self._ocr = winrt_ocr
        self._imaging = imaging
        self._streams = streams

        # try to create engine for the desired language; the call may
        # hang indefinitely if the language pack isn't present, so run it
        # in a separate thread with a short timeout.
        def make_engine(q):
            try:
                q.put(winrt_ocr.OcrEngine.try_create_from_language(Language(self._lang_tag)))
            except Exception as e:
                q.put(e)

        import threading, queue
        q: "queue.Queue[object]" = queue.Queue()
        th = threading.Thread(target=make_engine, args=(q,))
        th.daemon = True
        th.start()
        th.join(2.0)  # wait up to 2 seconds
        if q.empty():
            print(f"[OCR] WinRT try_create_from_language({self._lang_tag}) timed out (pack missing?)", flush=True)
            return
        result = q.get()
        if isinstance(result, Exception):
            print(f"[OCR] WinRT init raised", result, flush=True)
            return
        self._engine = result
        if self._engine is None:
            # most likely the language pack wasn't installed
            print(f"[OCR] WinRT engine unavailable for '{self._lang_tag}'", flush=True)

    def detect(self, image: np.ndarray, min_confidence: float = 0.3) -> list[TextBlock]:
        if self._engine is None:
            return []
        # convert numpy -> PIL -> bytes -> SoftwareBitmap
        from PIL import Image
        pil = Image.fromarray(image)
        img_bytes = pil.convert("RGBA").tobytes()
        w, h = pil.size
        writer = self._streams.DataWriter()
        writer.write_bytes(list(img_bytes))
        buf = writer.detach_buffer()
        bitmap = self._imaging.SoftwareBitmap.create_copy_from_buffer(
            buf,
            self._imaging.BitmapPixelFormat.RGBA8,
            w,
            h,
            self._imaging.BitmapAlphaMode.PREMULTIPLIED,
        )
        # the recognize_async method is awaitable
        async def _recognize():
            return await self._engine.recognize_async(bitmap)
        try:
            result = asyncio.run(_recognize())
        except Exception as e:
            print(f"[OCR] WinRT recognition failed: {e}", flush=True)
            return []
        blocks: list[TextBlock] = []
        for line in result.lines:
            rect = line.bounding_rect
            x, y, w0, h0 = rect.x, rect.y, rect.width, rect.height
            bbox = [[x, y], [x + w0, y], [x + w0, y + h0], [x, y + h0]]
            text = line.text or ""
            blocks.append(TextBlock(text=text, bbox=bbox, confidence=1.0))
        return blocks


class _RapidEngine(_BaseEngine):
    """OCR via rapidocr-onnxruntime wrapper. Much faster than EasyOCR on CPU."""

    def __init__(self, lang: str):
        self._lang = lang
        self._rapid = None
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._rapid = RapidOCR()
            print("[OCR] Loaded RapidOCR", flush=True)
        except Exception as e:
            msg = str(e)
            print(f"[OCR] RapidOCR unavailable: {msg}", flush=True)
            if "DLL load failed" in msg:
                print("[OCR]  * ONNX Runtime failure – try reinstalling cpu-only or matching GPU runtime.", flush=True)

    def detect(self, image: np.ndarray, min_confidence: float = 0.20) -> list[TextBlock]:
        if self._rapid is None:
            return []
        try:
            results = self._rapid(image)
        except Exception as e:
            print(f"[OCR] RapidOCR error: {e}", flush=True)
            return []
        blocks: list[TextBlock] = []
        # rapid returns list with [boxes, txts, confidences] per page
        if results and results[0]:
            for box, text, conf in results[0]:
                # confidence may occasionally be returned as a string
                try:
                    conf_val = float(conf)
                except Exception:
                    conf_val = 0.0
                if conf_val >= min_confidence and text.strip():
                    # box is 4x2 array
                    bbox = [[float(x), float(y)] for x, y in box]
                    blocks.append(TextBlock(text=text.strip(), bbox=bbox, confidence=conf_val))
        blocks.sort(key=lambda b: (b.y, b.x))
        return blocks


class _PaddleEngine(_BaseEngine):
    """Fallback engine using paddleocr if installed (usually slower than rapid)."""

    def __init__(self, lang: str):
        self._lang = lang
        self._model = None
        self._error_count = 0          # consecutive error counter
        self._disabled = False         # permanently disabled after too many errors
        try:
            import os
            # Bypass network connectivity check (PaddleOCR 3.x)
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            from paddleocr import PaddleOCR
            paddle_lang = "korean" if lang.startswith("ko") else "en"
            # PaddleOCR 3.x removed show_log; try with, then without
            for kwargs in [
                {"lang": paddle_lang, "use_angle_cls": False},
                {"lang": paddle_lang},
                {},
            ]:
                try:
                    self._model = PaddleOCR(**kwargs)
                    break
                except TypeError:
                    continue
            if self._model is not None:
                print("[OCR] Loaded PaddleOCR", flush=True)
            else:
                print("[OCR] PaddleOCR: no compatible init signature found", flush=True)
        except Exception as e:
            msg = str(e)
            print(f"[OCR] PaddleOCR unavailable: {msg}", flush=True)
            if "c10.dll" in msg or "DLL load failed" in msg:
                print("[OCR]  * Paddle uses torch under the hood – ensure torch can import or install CPU-only wheel.", flush=True)

    def detect(self, image: np.ndarray, min_confidence: float = 0.45) -> list[TextBlock]:
        if self._model is None or self._disabled:
            return []
        try:
            # PaddleOCR 3.x uses predict() API; cls arg is no longer accepted
            results = self._model.predict(image)
            self._error_count = 0   # reset on success
        except (TypeError, AttributeError):
            try:
                results = self._model.ocr(image)
                self._error_count = 0
            except Exception as e:
                self._error_count += 1
                print(f"[OCR] PaddleOCR error: {e}", flush=True)
                if self._error_count >= 3:
                    self._disabled = True
                    print("[OCR] PaddleOCR disabled after 3 consecutive errors", flush=True)
                return []
        except Exception as e:
            self._error_count += 1
            print(f"[OCR] PaddleOCR error: {e}", flush=True)
            if self._error_count >= 3:
                self._disabled = True
                print("[OCR] PaddleOCR disabled after 3 consecutive errors", flush=True)
            return []
        blocks: list[TextBlock] = []
        try:
            # PaddleOCR 2.x format: results is list[list[[bbox,(text,conf)]]]
            if results and results[0] and isinstance(results[0], list):
                first = results[0][0]
                # check if it's the v2.x tuple structure
                if isinstance(first, (list, tuple)) and len(first) == 2 and isinstance(first[1], (list, tuple)):
                    for line in results[0]:
                        bbox, (text, conf) = line
                        if conf >= min_confidence and text.strip():
                            blocks.append(TextBlock(text=text.strip(), bbox=bbox, confidence=conf))
                else:
                    # maybe v3.x wraps differently — fall through to attribute access
                    raise ValueError("unexpected format")
            else:
                # PaddleOCR 3.x format: results is list of Result-like objects
                for page in (results or []):
                    page_results = getattr(page, "rec_res", None) or getattr(page, "get", lambda k, d=None: None)("rec_res") or []
                    bboxes = getattr(page, "dt_boxes", []) or []
                    for bbox, (text, conf) in zip(bboxes, page_results):
                        if conf >= min_confidence and text.strip():
                            bbox_list = [[float(p[0]), float(p[1])] for p in bbox]
                            blocks.append(TextBlock(text=text.strip(), bbox=bbox_list, confidence=conf))
        except Exception as e:
            print(f"[OCR] PaddleOCR parse error: {e}. Raw results type: {type(results)}", flush=True)
        blocks.sort(key=lambda b: (b.y, b.x))
        return blocks


def _avg_confidence(blocks: list[TextBlock]) -> float:
    """Compute average confidence of a list of TextBlocks. Returns 0.0 if empty."""
    if not blocks:
        return 0.0
    return sum(b.confidence for b in blocks) / len(blocks)


def _make_engine(name: str, lang: str) -> "_BaseEngine | None":
    """Instantiate and return a backend engine by name, or None if unavailable."""
    if name == BACKEND_RAPIDOCR:
        e = _RapidEngine(lang)
        return e if e._rapid is not None else None
    if name == BACKEND_PADDLEOCR:
        e = _PaddleEngine(lang)
        return e if e._model is not None else None
    if name == BACKEND_WINRT:
        e = _WinRTEngine(lang)
        return e if e._engine is not None else None
    if name == BACKEND_EASYOCR:
        e = _EasyOCREngine(lang)
        e._ensure_reader()
        return e if e._reader is not None else None
    return None


class OCREngine:
    """
    Unified front-end that runs a primary backend and optionally a fallback.

    Strategy (configurable via config dict):
      - primary_backend   : engine to try first        (default: "rapidocr")
      - fallback_backend  : engine when conf too low    (default: "paddleocr")
      - confidence_thresh : avg-confidence gate         (default: 0.75)
      - winrt_enabled     : allow WinRT as primary/fb   (default: False)
      - easyocr_enabled   : allow EasyOCR as primary/fb (default: False)

    If primary fails to load, falls back automatically.
    If fallback also fails, returns primary result (even if empty).
    """

    def __init__(self, source_language: str = "en", config: dict | None = None):
        self._source_language = source_language
        self._config = config or {}
        self._primary: _BaseEngine | None = None
        self._fallback: _BaseEngine | None = None
        self._primary_name: str = BACKEND_NONE
        self._fallback_name: str = BACKEND_NONE
        self._confidence_thresh: float = self._config.get("confidence_thresh", 0.75)
        self._select_backends()

    # ── Internal helpers ────────────────────────────────────────────────

    def _select_backends(self):
        """Instantiate primary and fallback engines according to config.

        Default cascade (when winrt_enabled=True):  WinRT → RapidOCR → EasyOCR
        Default cascade (when winrt_enabled=False): RapidOCR → EasyOCR
        PaddleOCR is NOT in any auto-cascade (broken oneDNN on most systems).
        """
        self._primary = None
        self._fallback = None
        self._primary_name = BACKEND_NONE
        self._fallback_name = BACKEND_NONE
        self._confidence_thresh = self._config.get("confidence_thresh", 0.75)

        wanted_primary  = self._config.get("primary_backend",  BACKEND_RAPIDOCR)
        wanted_fallback = self._config.get("fallback_backend", BACKEND_NONE)
        winrt_ok   = self._config.get("winrt_enabled",   False)
        easyocr_ok = self._config.get("easyocr_enabled", False)

        # Respect opt-in guards
        if wanted_primary == BACKEND_WINRT and not winrt_ok:
            print("[OCR] WinRT requested but winrt_enabled=false — skipping", flush=True)
            wanted_primary = BACKEND_RAPIDOCR
        if wanted_primary == BACKEND_EASYOCR and not easyocr_ok:
            wanted_primary = BACKEND_RAPIDOCR

        if wanted_fallback == BACKEND_WINRT and not winrt_ok:
            wanted_fallback = BACKEND_NONE
        if wanted_fallback == BACKEND_EASYOCR and not easyocr_ok:
            wanted_fallback = BACKEND_NONE

        # ── Load primary ──────────────────────────────────────────────
        engine = _make_engine(wanted_primary, self._source_language)
        if engine is not None:
            self._primary      = engine
            self._primary_name = wanted_primary
            print(f"[OCR] Primary backend: {wanted_primary}", flush=True)
        else:
            # Auto-cascade: WinRT → RapidOCR → EasyOCR  (no PaddleOCR)
            print(f"[OCR] Primary '{wanted_primary}' unavailable — cascade", flush=True)
            cascade = []
            if winrt_ok:
                cascade.append(BACKEND_WINRT)
            cascade.append(BACKEND_RAPIDOCR)
            if easyocr_ok:
                cascade.append(BACKEND_EASYOCR)
            for name in cascade:
                if name == wanted_primary:
                    continue  # already tried
                e = _make_engine(name, self._source_language)
                if e is not None:
                    self._primary      = e
                    self._primary_name = name
                    print(f"[OCR] Primary auto-selected: {name}", flush=True)
                    break

        # ── Load fallback (different from primary) ────────────────────
        if (wanted_fallback
                and wanted_fallback != BACKEND_NONE
                and wanted_fallback != BACKEND_PADDLEOCR   # skip broken paddle
                and wanted_fallback != self._primary_name):
            fe = _make_engine(wanted_fallback, self._source_language)
            if fe is not None:
                self._fallback      = fe
                self._fallback_name = wanted_fallback
                print(f"[OCR] Fallback backend: {wanted_fallback}", flush=True)
            else:
                print(f"[OCR] Fallback '{wanted_fallback}' unavailable", flush=True)
        elif wanted_fallback == BACKEND_PADDLEOCR:
            print("[OCR] PaddleOCR skipped (broken oneDNN)", flush=True)


    # ── Public API ──────────────────────────────────────────────────────

    def set_source_language(self, lang: str):
        if lang != self._source_language:
            self._source_language = lang
            self._select_backends()

    def update_config(self, config: dict):
        """Update engine config (e.g. after settings change) and reload backends."""
        self._config = config
        self._select_backends()

    @property
    def backend_name(self) -> str:
        """Human-readable name shown in overlay title bar."""
        if self._primary_name == BACKEND_NONE:
            return "<none>"
        if self._fallback_name and self._fallback_name != BACKEND_NONE:
            return f"{self._primary_name}+{self._fallback_name}"
        return self._primary_name

    @property
    def primary_name(self) -> str:
        return self._primary_name

    @property
    def fallback_name(self) -> str:
        return self._fallback_name

    def detect(self, image: np.ndarray, min_confidence: float = 0.3) -> list[TextBlock]:
        """
        Run OCR using primary backend.
        Fallback is triggered ONLY when primary returns 0 blocks —
        NOT based on confidence (which causes false fallbacks on normal text).
        """
        if self._primary is None:
            return []

        primary_blocks = self._primary.detect(image, min_confidence)

        # Log actual confidence so we can tune if needed
        if primary_blocks:
            avg = sum(b.confidence for b in primary_blocks) / len(primary_blocks)
            print(f"[OCR] {self._primary_name} conf={avg:.2f}  blocks={len(primary_blocks)}", flush=True)

        # Only fall back when primary found nothing at all
        if not primary_blocks and self._fallback is not None:
            print(f"[OCR] No blocks → fallback ({self._fallback_name})", flush=True)
            fallback_blocks = self._fallback.detect(image, min_confidence)
            if fallback_blocks:
                avg = sum(b.confidence for b in fallback_blocks) / len(fallback_blocks)
                print(f"[OCR] {self._fallback_name} conf={avg:.2f}  blocks={len(fallback_blocks)}", flush=True)
            return fallback_blocks

        return primary_blocks

    def warm_up(self):
        """Pre-warm both backends to avoid cold-start latency on first real frame."""
        import numpy as np
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        if self._primary is not None:
            print(f"[OCR] Warming up {self._primary_name}...", flush=True)
            try:
                self._primary.detect(dummy)
            except Exception:
                pass
        if self._fallback is not None:
            print(f"[OCR] Warming up {self._fallback_name}...", flush=True)
            try:
                self._fallback.detect(dummy)
            except Exception:
                pass
        print("[OCR] Warm-up done ✅", flush=True)
