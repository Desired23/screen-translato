# async_pipeline.py - Non-blocking OCR + translate pipeline
#
# Design: runs everything in a single background QThread.
# A new frame cancels in-flight work via a threading.Event().
# Results are emitted via pyqtSignal (thread-safe in Qt).

import threading
import time
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
        self._lock = threading.Lock()

    # ── Called from main thread ───────────────────────────────────────

    def submit(self, image: np.ndarray):
        """Enqueue a new frame. Cancels in-flight work immediately."""
        with self._lock:
            self._image = image
            self._cancel.set()   # signal cancellation to running work

    # ── QThread.run ───────────────────────────────────────────────────

    def run(self):
        """Spin loop: pick up the latest image, process, repeat."""
        while not self.isInterruptionRequested():
            # Wait for work
            with self._lock:
                image = self._image
                self._image = None
                self._cancel.clear()

            if image is None:
                self.msleep(20)   # short idle sleep
                continue

            self._process(image)

    def _process(self, image: np.ndarray):
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
            from ocr_engine import merge_same_line_blocks
            blocks = merge_same_line_blocks(blocks)

            # ── Translate ─────────────────────────────────────────────
            if self._cancel.is_set():
                return
            texts = [b.text for b in blocks]
            t_tr = time.perf_counter()
            translated = self._translator.translate_batch(texts)
            print(f"[Perf] Translate={( time.perf_counter()-t_tr)*1000:.0f}ms", flush=True)

            if self._cancel.is_set():
                return

            elapsed = (time.perf_counter() - t0) * 1000
            self.result_ready.emit(PipelineResult(blocks, translated, elapsed))

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

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

            MAX_BOXES = 20
            boxes = list(dt_boxes[:MAX_BOXES])

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

    def __init__(self, ocr_engine, translator, parent=None):
        super().__init__(parent)
        self._ocr = ocr_engine
        self._translator = translator

        self._worker = _Worker(ocr_engine, translator)
        self._worker.result_ready.connect(self.result_ready)
        self._worker.error.connect(self.pipeline_error)
        self._worker.start()

        # Fast-sample to skip identical frames
        self._last_sample: np.ndarray | None = None

    # ── Public API ────────────────────────────────────────────────────

    def submit(self, image: np.ndarray):
        """Non-blocking. Drops similar frames, cancels in-flight work."""
        img_sample = self._fast_sample(image)
        if self._last_sample is not None:
            try:
                if img_sample.shape == self._last_sample.shape:
                    mad = np.mean(np.abs(img_sample - self._last_sample))
                    # 3.0 absolute difference threshold to ignore minor noise/tiny animations
                    if mad < 3.0:
                        return
            except ValueError:
                pass
        self._last_sample = img_sample
        self._worker.submit(image)

    def shutdown(self):
        """Stop the background thread cleanly."""
        self._worker.requestInterruption()
        self._worker.quit()
        self._worker.wait(2000)

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _fast_sample(img: np.ndarray) -> np.ndarray:
        """~0.1ms downsample for rough image similarity check."""
        return img[::8, ::8, 0].astype(np.float32)
