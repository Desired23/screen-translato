# async_pipeline.py - Non-blocking OCR + translate pipeline
#
# Design: runs everything in a single background QThread.
# A new frame cancels in-flight work via a threading.Event().
# Results are emitted via pyqtSignal (thread-safe in Qt).

import threading
import time
import re
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal


@dataclass
class PipelineResult:
    blocks: list        # list[TextBlock]
    translated: list    # list[str]
    elapsed_ms: float


class _Worker(QThread):
    """Single background thread: OCR → merge → translate → emit."""

    result_ready = pyqtSignal(object)   # PipelineResult
    error = pyqtSignal(str)

    def __init__(self, ocr_engine, translator):
        super().__init__()
        self._ocr = ocr_engine
        self._translator = translator

        self._image: np.ndarray | None = None
        self._cancel = threading.Event()
        self._processing = threading.Event()
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    # ── Called from main thread ───────────────────────────────────────

    def submit(self, image: np.ndarray):
        """Enqueue a new frame. Cancels in-flight work immediately."""
        with self._condition:
            self._image = image
            self._cancel.set()   # signal cancellation to running work
            self._condition.notify()

    def wake(self):
        """Wake thread if it's blocked waiting for new frames."""
        with self._condition:
            self._condition.notify_all()

    def is_processing(self) -> bool:
        return self._processing.is_set()

    # ── QThread.run ───────────────────────────────────────────────────

    def run(self):
        """Spin loop: pick up the latest image, process, repeat."""
        while not self.isInterruptionRequested():
            with self._condition:
                while self._image is None and not self.isInterruptionRequested():
                    self._condition.wait(timeout=0.25)
                if self.isInterruptionRequested():
                    break
                image = self._image
                self._image = None
                self._cancel.clear()

            self._process(image)

    def _process(self, image: np.ndarray):
        self._processing.set()
        t0 = time.perf_counter()
        try:
            if self._cancel.is_set():
                return

            # ── OCR ───────────────────────────────────────────────────
            t_ocr = time.perf_counter()
            blocks = self._ocr_smart(image)
            print(f"[Perf] OCR={( time.perf_counter()-t_ocr)*1000:.0f}ms  blocks={len(blocks)}", flush=True)

            if self._cancel.is_set():
                return

            if not blocks:
                self.result_ready.emit(PipelineResult([], [], 0.0))
                return

            # ── Merge same-line blocks ────────────────────────────────
            if self._cancel.is_set():
                return
            from ocr_engine import merge_same_line_blocks, merge_paragraph_blocks
            line_blocks = merge_same_line_blocks(blocks)
            blocks = line_blocks
            cfg = getattr(self._ocr, "_config", {})
            if cfg.get("paragraph_merge_enabled", True):
                if cfg.get("auto_document_line_mode_enabled", True) and self._looks_like_document_frame(
                    line_blocks, cfg
                ):
                    if cfg.get("auto_document_translate_by_paragraph", True):
                        doc_blocks = merge_paragraph_blocks(
                            line_blocks,
                            max_lines_per_group=int(cfg.get("document_paragraph_max_lines_per_group", 8)),
                            min_overlap_ratio=float(cfg.get("document_paragraph_min_overlap_ratio", 0.30)),
                            center_distance_factor=float(cfg.get("document_paragraph_center_distance_factor", 0.85)),
                            max_width_expand_ratio=float(cfg.get("document_paragraph_max_width_expand_ratio", 3.00)),
                        )
                        if doc_blocks:
                            collapse_ratio = len(doc_blocks) / max(len(line_blocks), 1)
                            if len(line_blocks) >= 10 and collapse_ratio < 0.06:
                                print(
                                    f"[OCR] Document paragraph merge over-collapsed ({len(line_blocks)} -> {len(doc_blocks)}), "
                                    "fallback to line blocks",
                                    flush=True,
                                )
                                blocks = line_blocks
                            else:
                                print(
                                    f"[OCR] Document paragraph merge ({len(line_blocks)} -> {len(doc_blocks)})",
                                    flush=True,
                                )
                                blocks = doc_blocks
                        else:
                            print(
                                f"[OCR] Document text pattern detected ({len(line_blocks)} lines) -> line-level mode",
                                flush=True,
                            )
                            blocks = line_blocks
                    else:
                        print(
                            f"[OCR] Document text pattern detected ({len(line_blocks)} lines) -> line-level mode",
                            flush=True,
                        )
                        blocks = line_blocks
                elif cfg.get("auto_ui_line_mode_enabled", True) and self._looks_like_ui_game_frame(
                    line_blocks, cfg
                ):
                    if cfg.get("ui_panel_merge_enabled", True):
                        panel_blocks = self._merge_ui_panel_blocks(line_blocks, cfg)
                        if len(panel_blocks) != len(line_blocks):
                            print(
                                f"[OCR] UI/game panel merge ({len(line_blocks)} -> {len(panel_blocks)})",
                                flush=True,
                            )
                        else:
                            print(
                                f"[OCR] UI/game text pattern detected ({len(line_blocks)} lines) -> line-level mode",
                                flush=True,
                            )
                        blocks = panel_blocks
                    else:
                        print(
                            f"[OCR] UI/game text pattern detected ({len(line_blocks)} lines) -> line-level mode",
                            flush=True,
                        )
                        blocks = line_blocks
                else:
                    merged_blocks = merge_paragraph_blocks(
                        line_blocks,
                        max_lines_per_group=int(cfg.get("paragraph_max_lines_per_group", 4)),
                        min_overlap_ratio=float(cfg.get("paragraph_min_overlap_ratio", 0.35)),
                        center_distance_factor=float(cfg.get("paragraph_center_distance_factor", 0.55)),
                        max_width_expand_ratio=float(cfg.get("paragraph_max_width_expand_ratio", 1.8)),
                    )
                    # Safety guard: if too many lines collapse into too few blocks,
                    # keep line-level blocks to avoid cross-bubble paragraph merge.
                    collapse_threshold = max(2, len(line_blocks) // 4)
                    if len(line_blocks) >= 8 and len(merged_blocks) <= collapse_threshold:
                        print(
                            f"[OCR] Paragraph merge over-collapsed ({len(line_blocks)} -> {len(merged_blocks)}), "
                            "fallback to line blocks",
                            flush=True,
                        )
                        blocks = line_blocks
                    else:
                        blocks = merged_blocks

            # ── Translate ─────────────────────────────────────────────
            if self._cancel.is_set():
                return
            texts = [b.text for b in blocks]
            t_tr = time.perf_counter()
            translated = self._translator.translate_batch(texts, cancel_check=self._cancel.is_set)
            print(f"[Perf] Translate={( time.perf_counter()-t_tr)*1000:.0f}ms", flush=True)

            if self._cancel.is_set():
                return

            elapsed = (time.perf_counter() - t0) * 1000
            self.result_ready.emit(PipelineResult(blocks, translated, elapsed))

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
        finally:
            self._processing.clear()

    @staticmethod
    def _is_joined_token(text: str) -> bool:
        token = text.strip()
        if " " in token:
            return False
        letters = [c for c in token if c.isalpha()]
        if len(letters) < 6:
            return False
        upper_ratio = sum(1 for c in letters if c.isupper()) / max(len(letters), 1)
        return upper_ratio >= 0.8

    def _joined_ratio(self, blocks: list) -> float:
        if not blocks:
            return 1.0
        joined = sum(1 for b in blocks if self._is_joined_token(getattr(b, "text", "")))
        return joined / len(blocks)

    def _quality_score(self, blocks: list) -> float:
        if not blocks:
            return -1.0
        avg_conf = sum(getattr(b, "confidence", 0.0) for b in blocks) / len(blocks)
        joined = sum(1 for b in blocks if self._is_joined_token(getattr(b, "text", "")))
        spaced = sum(1 for b in blocks if " " in getattr(b, "text", "").strip())
        return (avg_conf * 1.2) + (len(blocks) * 0.015) + (spaced * 0.02) - (joined * 0.05)

    @staticmethod
    def _looks_ui_like_text(text: str) -> bool:
        s = text.strip()
        if not s:
            return False
        if re.search(r"[0-9]", s) and re.search(r"[A-Za-z]", s):
            return True
        if re.search(r"[a-z][A-Z]", s):
            return True
        if any(ch in s for ch in ("-", "/", "_", ":", "%")):
            return True
        return False

    @staticmethod
    def _looks_paragraph_like_frame(blocks: list) -> bool:
        if not blocks:
            return False

        punct_count = 0
        long_count = 0
        total_words = 0

        for b in blocks:
            t = getattr(b, "text", "").strip()
            if not t:
                continue
            words = len(t.split())
            total_words += words
            if words >= 6:
                long_count += 1
            if words >= 4 and re.search(r"[.,;:!?)]$", t):
                punct_count += 1

        n = max(len(blocks), 1)
        avg_words = total_words / n
        long_ratio = long_count / n
        punct_ratio = punct_count / n

        return (
            avg_words >= 4.6
            or long_ratio >= 0.30
            or (long_ratio >= 0.20 and punct_ratio >= 0.12)
        )

    @staticmethod
    def _is_bullet_like(text: str) -> bool:
        s = text.strip()
        if not s:
            return False
        if s[0] in {"•", "-", "*", "–"}:
            return True
        return bool(re.match(r"^\d+[\.)]\s+", s))

    def _looks_like_document_frame(self, blocks: list, cfg: dict) -> bool:
        if not blocks:
            return False
        if len(blocks) < int(cfg.get("auto_document_line_mode_min_blocks", 4)):
            return False

        total_words = 0
        long_count = 0
        sentence_like = 0
        bullet_like = 0
        tech_like = 0
        joined_like = 0
        alpha_lines = 0

        for b in blocks:
            t = getattr(b, "text", "").strip()
            if not t:
                continue
            words = len(t.split())
            total_words += words
            if words >= 6:
                long_count += 1
            if words >= 4 and re.search(r"[.,;:!?)]$", t):
                sentence_like += 1
            if self._is_bullet_like(t):
                bullet_like += 1
            if self._looks_ui_like_text(t):
                tech_like += 1
            if self._is_joined_token(t):
                joined_like += 1
            if re.search(r"[A-Za-z]", t):
                alpha_lines += 1

        n = max(len(blocks), 1)
        if alpha_lines / n < 0.5:
            return False

        avg_words = total_words / n
        long_ratio = long_count / n
        sentence_ratio = sentence_like / n
        bullet_ratio = bullet_like / n
        tech_ratio = tech_like / n
        joined_ratio = joined_like / n

        if tech_ratio >= 0.40 and joined_ratio >= 0.20:
            return False

        return (
            avg_words >= float(cfg.get("auto_document_line_mode_avg_words", 4.8))
            or long_ratio >= float(cfg.get("auto_document_line_mode_long_ratio", 0.28))
            or sentence_ratio >= float(cfg.get("auto_document_line_mode_sentence_ratio", 0.14))
            or bullet_ratio >= float(cfg.get("auto_document_line_mode_bullet_ratio", 0.08))
        )

    def _looks_like_ui_game_frame(self, blocks: list, cfg: dict) -> bool:
        if not blocks:
            return False
        if len(blocks) < int(cfg.get("auto_ui_line_mode_min_blocks", 12)):
            return False
        if self._looks_paragraph_like_frame(blocks):
            return False

        short_count = 0
        tech_count = 0
        for b in blocks:
            t = getattr(b, "text", "").strip()
            if not t:
                continue
            if len(t.split()) <= 4:
                short_count += 1
            if self._looks_ui_like_text(t):
                tech_count += 1

        n = max(len(blocks), 1)
        short_ratio = short_count / n
        tech_ratio = tech_count / n
        joined_ratio = self._joined_ratio(blocks)

        return (
            short_ratio >= float(cfg.get("auto_ui_line_mode_short_ratio", 0.55))
            and (
                tech_ratio >= float(cfg.get("auto_ui_line_mode_tech_ratio", 0.28))
                or (
                    tech_ratio >= 0.12
                    and joined_ratio >= float(cfg.get("auto_ui_line_mode_joined_ratio", 0.22))
                )
            )
        )

    def _merge_ui_panel_blocks(self, line_blocks: list, cfg: dict) -> list:
        """Merge nearby UI/game lines inside the same panel, not across whole frame."""
        if not line_blocks:
            return line_blocks
        from ocr_engine import merge_paragraph_blocks

        merged = merge_paragraph_blocks(
            line_blocks,
            max_lines_per_group=int(cfg.get("ui_panel_merge_max_lines_per_group", 3)),
            min_overlap_ratio=float(cfg.get("ui_panel_merge_min_overlap_ratio", 0.45)),
            center_distance_factor=float(cfg.get("ui_panel_merge_center_distance_factor", 0.40)),
            max_width_expand_ratio=float(cfg.get("ui_panel_merge_max_width_expand_ratio", 1.45)),
        )

        if not merged:
            return line_blocks

        # Keep panel-merge conservative to avoid accidental across-panel merge.
        if len(merged) <= max(2, len(line_blocks) // 3):
            return line_blocks
        return merged

    @staticmethod
    def _is_dark_ui_image(image: np.ndarray) -> bool:
        if image is None or image.size == 0:
            return False
        if image.ndim == 3:
            gray = image[..., :3].astype(np.float32).mean(axis=2)
            r = image[..., 0].astype(np.float32)
            g = image[..., 1].astype(np.float32)
            b = image[..., 2].astype(np.float32)
            # yellow-ish highlight text ratio
            yellow_ratio = float(np.mean((r > 140) & (g > 130) & (b < 120)))
        else:
            gray = image.astype(np.float32)
            yellow_ratio = 0.0

        dark_ratio = float(np.mean(gray < 80))
        return dark_ratio >= 0.35 and yellow_ratio >= 0.005

    @staticmethod
    def _enhance_dark_ui_image(image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return image
        arr = image.astype(np.float32)
        p10 = np.percentile(arr, 10)
        p95 = np.percentile(arr, 95)
        if p95 - p10 > 1:
            arr = (arr - p10) * (255.0 / (p95 - p10))
        arr = np.clip((arr - 16.0) * 1.22, 0, 255)
        # mild gamma to lift bright glyphs on dark background
        arr = np.power(arr / 255.0, 0.88) * 255.0
        return np.clip(arr, 0, 255).astype(np.uint8)

    def _ocr_smart(self, image: np.ndarray) -> list:
        """
        Smart OCR strategy:
          1. Try det + BATCH rec via RapidOCR internals (fast path)
          2. If too many blocks and still slow → WinRT fallback (O(1))
          3. Ultimate fallback: standard ocr_engine.detect()
        """
        # Access internal RapidOCR det/rec components
        primary = getattr(self._ocr, '_primary', None)
        rapid   = getattr(primary, '_rapid', None)

        if rapid is None or not hasattr(rapid, 'text_det') or not hasattr(rapid, 'text_rec'):
            return self._ocr.detect(image) or []

        try:
            # ── Stage 1: Detection (fixed cost ~80-150ms) ─────────────
            t_det = time.perf_counter()
            dt_boxes, _ = rapid.text_det(image)
            det_ms = (time.perf_counter() - t_det) * 1000

            if dt_boxes is None or len(dt_boxes) == 0:
                print(f"[OCR] det={det_ms:.0f}ms  no boxes", flush=True)
                return []

            boxes = sorted(
                list(dt_boxes),
                key=lambda b: (
                    float(np.min(np.array(b)[:, 1])),
                    float(np.min(np.array(b)[:, 0])),
                ),
            )
            max_boxes = int(getattr(self._ocr, "_config", {}).get("ocr_smart_max_boxes", 80))
            if max_boxes > 0:
                boxes = boxes[:max_boxes]

            # ── Stage 2: Crop each box ────────────────────────────────
            crops, valid_boxes = [], []
            for box in boxes:
                box_arr = np.array(box)
                xs, ys = box_arr[:, 0], box_arr[:, 1]
                x1, y1 = max(0, int(xs.min())), max(0, int(ys.min()))
                x2, y2 = min(image.shape[1], int(xs.max())), min(image.shape[0], int(ys.max()))
                crop = image[y1:y2, x1:x2]
                if crop.size > 0 and crop.shape[0] > 2 and crop.shape[1] > 2:
                    crops.append(crop)
                    valid_boxes.append(box)

            if not crops:
                return []

            # ── Stage 3: BATCH recognition (1 call for all crops) ─────
            # RapidOCR text_rec(list_of_crops) is much faster than
            # calling text_rec([1 crop]) N times (avoids N*preprocessing).
            t_rec = time.perf_counter()
            try:
                rec_results, _ = rapid.text_rec(crops)   # batch call ✅
            except Exception as e:
                print(f"[OCR] Batch rec failed: {e}", flush=True)
                rec_results = None
            rec_ms = (time.perf_counter() - t_rec) * 1000

            print(f"[OCR] det={det_ms:.0f}ms  rec_batch(x{len(crops)})={rec_ms:.0f}ms", flush=True)

            # ── WinRT fallback when rec is super slow ─────────────────
            total_ocr_ms = det_ms + rec_ms
            if total_ocr_ms > 1500 and len(crops) > 12:
                winrt_blocks = self._try_winrt(image)
                if winrt_blocks:
                    print(f"[OCR] WinRT fallback returned {len(winrt_blocks)} blocks", flush=True)
                    return winrt_blocks

            if not rec_results:
                return []

            # ── Build TextBlocks ──────────────────────────────────────
            from ocr_engine import TextBlock
            MIN_CONF = 0.20
            blocks = []
            for box, rec in zip(valid_boxes, rec_results):
                if rec is None:
                    continue
                if isinstance(rec, (list, tuple)) and len(rec) == 2:
                    text, conf = rec
                else:
                    text, conf = str(rec), 0.0
                text = str(text).strip()
                conf = float(conf) if conf is not None else 0.0
                if conf >= MIN_CONF and text:
                    bbox = [[float(x), float(y)] for x, y in box]
                    blocks.append(TextBlock(text=text, bbox=bbox, confidence=conf))

            blocks.sort(key=lambda b: (b.y, b.x))
            # If the fast path yields many OCR-joined tokens (common in manga),
            # run the backend's quality-aware detect() once and keep the better result.
            if self._joined_ratio(blocks) >= 0.35:
                fallback_blocks = self._ocr.detect(image) or []
                if self._quality_score(fallback_blocks) > self._quality_score(blocks) + 0.03:
                    print("[OCR] Smart path switched to quality pass result", flush=True)
                    return fallback_blocks
            cfg = getattr(self._ocr, "_config", {})
            if cfg.get("ocr_dark_ui_retry_enabled", True) and self._is_dark_ui_image(image):
                avg_conf = (
                    sum(getattr(b, "confidence", 0.0) for b in blocks) / max(len(blocks), 1)
                    if blocks
                    else 0.0
                )
                joined_ratio = self._joined_ratio(blocks)
                if (
                    avg_conf <= float(cfg.get("ocr_dark_ui_retry_confidence_thresh", 0.84))
                    or joined_ratio >= float(cfg.get("ocr_dark_ui_retry_joined_ratio_thresh", 0.26))
                ):
                    enhanced = self._enhance_dark_ui_image(image)
                    retry_blocks = self._ocr.detect(enhanced) or []
                    margin = float(cfg.get("ocr_dark_ui_retry_score_margin", 0.02))
                    if self._quality_score(retry_blocks) > self._quality_score(blocks) + margin:
                        print("[OCR] Dark-UI retry accepted enhanced OCR result", flush=True)
                        return retry_blocks
            return blocks

        except Exception as e:
            print(f"[OCR] Smart OCR failed ({e}), fallback", flush=True)
            return self._ocr.detect(image) or []

    def _try_winrt(self, image: np.ndarray) -> list:
        """Try Windows OCR API — O(1) regardless of block count."""
        try:
            from ocr_engine import _WinRTEngine, TextBlock
            if not hasattr(self, '_winrt'):
                self._winrt = _WinRTEngine('en')   # cached per worker
            return self._winrt.detect(image) or []
        except Exception as e:
            print(f"[OCR] WinRT unavailable: {e}", flush=True)
            return []




class AsyncTranslationPipeline(QObject):
    """
    Non-blocking translation pipeline — wraps _Worker (a QThread).

    Usage
    -----
    pipeline = AsyncTranslationPipeline(ocr_engine, translator, parent=self)
    pipeline.result_ready.connect(self._on_result)
    pipeline.submit(image)   # non-blocking, returns immediately
    pipeline.shutdown()      # call on close
    """

    result_ready = pyqtSignal(object)   # PipelineResult
    pipeline_error = pyqtSignal(str)

    # Class-level attribute so paintEvent check works even before __init__ fully runs
    _current_task = None

    def __init__(self, ocr_engine, translator, parent=None, drop_frames_when_busy: bool = True):
        super().__init__(parent)
        self._ocr = ocr_engine
        self._translator = translator
        self._drop_frames_when_busy = bool(drop_frames_when_busy)

        self._worker = _Worker(ocr_engine, translator)
        self._worker.result_ready.connect(self.result_ready)
        self._worker.error.connect(self.pipeline_error)
        self._worker.start()

        # Fast-sample to skip identical frames
        self._last_sample: np.ndarray | None = None

    # ── Public API ────────────────────────────────────────────────────

    def submit(self, image: np.ndarray):
        """Non-blocking. Drops similar frames, cancels in-flight work."""
        if self._drop_frames_when_busy and self._worker.is_processing():
            return
        img_sample = self._fast_sample(image)
        if self._last_sample is not None:
            try:
                if img_sample.shape == self._last_sample.shape:
                    diff = np.abs(
                        img_sample.astype(np.int16) - self._last_sample.astype(np.int16)
                    )
                    mad = float(np.mean(diff))
                    changed_ratio = float(np.mean(diff >= 3))
                    # Robust static-frame detection (ignore tiny compositor noise)
                    if mad < 4.5 and changed_ratio < 0.02:
                        return
            except ValueError:
                pass
        self._last_sample = img_sample
        self._worker.submit(image)

    def shutdown(self):
        """Stop the background thread cleanly."""
        self._worker.requestInterruption()
        self._worker.wake()
        self._worker.quit()
        self._worker.wait(2000)

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _fast_sample(img: np.ndarray) -> np.ndarray:
        """~0.1ms downsample for rough image similarity check."""
        if img.ndim == 3:
            gray = img[::8, ::8, :].astype(np.uint16).mean(axis=2)
        else:
            gray = img[::8, ::8].astype(np.uint16)
        return (gray // 4).astype(np.uint8)
