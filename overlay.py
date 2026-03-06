# overlay.py - Transparent overlay window with live translation
import ctypes
import traceback
import numpy as np
from PyQt6.QtCore import Qt, QTimer, QPoint, QRect
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QCursor
from PyQt6.QtWidgets import QWidget, QApplication

from capture import ScreenCapture
from ocr_engine import OCREngine, TextBlock
from translator import TranslationEngine
from renderer import TextRenderer
from async_pipeline import AsyncTranslationPipeline, PipelineResult
from config import load_config



class OverlayWindow(QWidget):
    """
    Transparent overlay window for screen translation.
    - Frameless, stays on top
    - Resizable by dragging edges
    - Movable by dragging title bar
    - Runs OCR + translation loop on captured screen beneath
    """

    TITLE_BAR_HEIGHT = 24
    BORDER_WIDTH = 4
    MIN_SIZE = 150

    def __init__(self, config: dict | None = None):
        super().__init__()

        self._config = config or load_config()
        self._capture = ScreenCapture()
        source_lang = self._config.get("source_language", "en")
        self._ocr = OCREngine(source_language=source_lang, config=self._config)
        self._translator = TranslationEngine(
            target_language=self._config.get("target_language", "vi"),
            source_language=source_lang,
        )
        self._renderer = TextRenderer(
            font_family=self._config.get("font_family", "Segoe UI"),
            text_color=self._config.get("text_color", "#000000"),
            bg_color=self._config.get("background_color", "#ffffff"),
            font_size_min=self._config.get("font_size_min", 10),
            font_size_max=self._config.get("font_size_max", 36),
        )

        # Translation results
        self._text_blocks: list[TextBlock] = []
        self._translated_texts: list[str] = []

        # Async pipeline (replaces TranslationWorker)
        self._pipeline = AsyncTranslationPipeline(
            ocr_engine=self._ocr,
            translator=self._translator,
            parent=self,
        )
        self._pipeline.result_ready.connect(self._on_result)
        self._pipeline.pipeline_error.connect(
            lambda msg: print(f"[Pipeline] Error: {msg}", flush=True)
        )

        # Whether the OS excludes our window from screen capture
        # (set in showEvent — requires Win10 Build 2004+)
        self._overlay_excluded: bool = False
        # Last dirty-frame hash (with overlay visible) for Mode B dedup
        self._last_dirty_hash: str = ""
        # Drag/resize state
        self._dragging = False
        self._resizing = False
        self._resize_edge = None
        self._drag_start = QPoint()
        self._geometry_start = QRect()

        self._setup_ui()
        self._setup_timer()
        # Pre-warm OCR backends in a background thread so app starts quickly
        import threading
        threading.Thread(target=self._ocr.warm_up, daemon=True, name="ocr-warmup").start()

    def _setup_ui(self):
        """Configure the overlay window."""
        self.setWindowTitle("Screen Translator")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool  # Don't show in taskbar
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        self.setMinimumSize(self.MIN_SIZE, self.MIN_SIZE)

        # Default size and position - center of screen
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.geometry()
            w, h = 600, 400
            x = (geo.width() - w) // 2
            y = (geo.height() - h) // 2
            self.setGeometry(x, y, w, h)
        else:
            self.setGeometry(200, 200, 600, 400)

    def _setup_timer(self):
        """Set up the periodic translation timer."""
        interval = self._config.get("capture_interval_ms", 300)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._timer.setInterval(interval)

    def showEvent(self, event):
        """Start translation loop when shown."""
        super().showEvent(event)
        self._timer.start()
        # Attempt to make this window invisible to screen-capture APIs
        # (mss, BitBlt) while staying visible to the user.
        # WDA_EXCLUDEFROMCAPTURE (0x11) requires Windows 10 Build 2004+.
        # Check the return value — False means the call failed; we must
        # fall back to the opacity-toggle approach to avoid a feedback loop
        # where OCR reads its own translated output.
        self._overlay_excluded = False
        try:
            hwnd = int(self.winId())
            WDA_EXCLUDEFROMCAPTURE = 0x00000011
            ok = ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
            self._overlay_excluded = bool(ok)
            print(f"[Capture] WDA_EXCLUDEFROMCAPTURE: {'OK — no opacity flicker' if self._overlay_excluded else 'FAILED — using opacity fallback'}", flush=True)
        except Exception as e:
            print(f"[Capture] WDA_EXCLUDEFROMCAPTURE unavailable: {e}", flush=True)
        QTimer.singleShot(300, self._on_tick)

    def hideEvent(self, event):
        """Stop translation loop when hidden."""
        super().hideEvent(event)
        self._timer.stop()
        self._text_blocks = []
        self._translated_texts = []

    def closeEvent(self, event):
        """Clean up resources."""
        self._timer.stop()
        self._pipeline.shutdown()
        self._capture.close()
        super().closeEvent(event)

    # ─── Translation Pipeline ─────────────────────────────────────────

    def _on_tick(self):
        """Capture → pipeline every timer tick (non-blocking).

        Two modes depending on whether the OS excludes our window from capture:

        A. _overlay_excluded=True  (Win10 2004+)
           └─ Capture directly every tick. OS makes overlay invisible to mss,
             so the image is always clean. pipeline.submit() deduplicates via hash.

        B. _overlay_excluded=False (older Windows / API failure)
           └─ Quick-capture WITH overlay visible, hash check first.
             Only hide overlay when hash changed (content update), then
             capture clean image and submit. This way flicker ONLY happens
             on real content changes, never on static screens.
        """
        if not self.isVisible():
            return

        try:
            geo = self.geometry()
            H_MARGIN = 4
            BOTTOM_MARGIN = 20
            cx = geo.x() + H_MARGIN
            cy = geo.y() + self.TITLE_BAR_HEIGHT
            cw = geo.width() - H_MARGIN * 2
            ch = geo.height() - self.TITLE_BAR_HEIGHT - BOTTOM_MARGIN

            if ch <= 10 or cw <= 10:
                return

            if self._overlay_excluded:
                # ── Mode A: direct capture, no flicker ever ────────────────
                image = self._capture.capture_region(cx, cy, cw, ch)
                self._pipeline.submit(image)  # deduped by hash inside
            else:
                # ── Mode B: hash check first, hide only on change ────────
                # Capture with overlay visible — slightly dirty but fast.
                # Compare dirty-to-dirty (not dirty-to-clean) to avoid a
                # permanent mismatch loop when overlay text changes the image.
                dirty = self._capture.capture_region(cx, cy, cw, ch)
                dirty_hash = self._pipeline._fast_hash(dirty)
                if dirty_hash == self._last_dirty_hash:
                    return  # nothing changed — skip entirely, no flicker

                self._last_dirty_hash = dirty_hash

                # Content changed — hide overlay to get a clean frame
                self.setWindowOpacity(0)
                QApplication.processEvents()
                clean = self._capture.capture_region(cx, cy, cw, ch)
                self.setWindowOpacity(1.0)
                self._pipeline.submit(clean)

        except Exception:
            traceback.print_exc()
            self.setWindowOpacity(1.0)

    def _on_result(self, result: PipelineResult):
        """Receive results from async pipeline and repaint."""
        if result.elapsed_ms > 0:
            print(f"[Perf] pipeline={result.elapsed_ms:.0f}ms  blocks={len(result.blocks)}", flush=True)
        self._text_blocks = result.blocks
        self._translated_texts = result.translated
        self.update()

    # ─── Painting ─────────────────────────────────────────────────────

    def paintEvent(self, event):
        """Draw the overlay UI and translated text."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w, h = self.width(), self.height()

        # ── Title bar (compact, subtle) ──
        title_rect = QRect(0, 0, w, self.TITLE_BAR_HEIGHT)
        title_bg = QColor("#1a1a2e")
        title_bg.setAlpha(200)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(title_bg))
        painter.drawRoundedRect(title_rect, 4, 4)
        # Square off bottom corners of title bar
        painter.drawRect(QRect(0, self.TITLE_BAR_HEIGHT - 4, w, 4))

        # Title text — thin, small
        font = QFont("Segoe UI", 8)
        font.setWeight(QFont.Weight.Thin)
        painter.setFont(font)
        painter.setPen(QPen(QColor("#aaaaaa")))

        backend = getattr(self._ocr, "backend_name", "?")
        is_busy = self._pipeline._worker.isRunning() and self._pipeline._worker._image is not None
        status = "⏳" if is_busy else f"✅ {len(self._text_blocks)}"
        painter.drawText(
            QRect(8, 0, w - 50, self.TITLE_BAR_HEIGHT),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            f"Translator [{backend}]  {status}",
        )

        # Close button (small)
        close_rect = QRect(w - 28, 3, 22, 18)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(200, 60, 60, 150)))
        painter.drawRoundedRect(close_rect, 3, 3)
        close_font = QFont("Segoe UI", 8)
        painter.setFont(close_font)
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(close_rect, Qt.AlignmentFlag.AlignCenter, "✕")

        # ── Content area - nearly transparent ──
        content_rect = QRect(0, self.TITLE_BAR_HEIGHT, w, h - self.TITLE_BAR_HEIGHT)
        content_bg = QColor(0, 0, 0, 1)  # Nearly transparent
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(content_bg))
        painter.drawRect(content_rect)

        # ── Border — white, thin ──
        painter.setPen(QPen(QColor("#ffffff"), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(QRect(0, 0, w - 1, h - 1), 4, 4)

        # ── Render translated text ──
        if self._text_blocks and self._translated_texts:
            self._renderer.render(
                painter,
                self._text_blocks,
                self._translated_texts,
                offset_x=0,
                offset_y=self.TITLE_BAR_HEIGHT,
            )

        # ── Resize handle (subtle) ──
        handle_color = QColor(255, 255, 255, 80)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(handle_color))
        painter.drawRect(QRect(w - 10, h - 10, 8, 2))
        painter.drawRect(QRect(w - 10, h - 6, 8, 2))
        painter.drawRect(QRect(w - 4, h - 10, 2, 8))

        painter.end()

    # ─── Mouse Interaction ────────────────────────────────────────────

    def mousePressEvent(self, event):
        """Handle mouse press for dragging and resizing."""
        if event.button() != Qt.MouseButton.LeftButton:
            return

        pos = event.pos()

        # Check close button
        close_rect = QRect(self.width() - 28, 3, 22, 18)
        if close_rect.contains(pos):
            self.hide()
            return

        # Check if on resize edge
        edge = self._get_resize_edge(pos)
        if edge:
            self._resizing = True
            self._resize_edge = edge
            self._drag_start = event.globalPosition().toPoint()
            self._geometry_start = self.geometry()
            return

        # Check if on title bar (drag area)
        if pos.y() <= self.TITLE_BAR_HEIGHT:
            self._dragging = True
            self._drag_start = event.globalPosition().toPoint()
            self._geometry_start = self.geometry()

    def mouseMoveEvent(self, event):
        """Handle mouse move for dragging and resizing."""
        pos = event.pos()

        if self._dragging:
            delta = event.globalPosition().toPoint() - self._drag_start
            self.move(self._geometry_start.topLeft() + delta)
            return

        if self._resizing:
            delta = event.globalPosition().toPoint() - self._drag_start
            geo = QRect(self._geometry_start)

            if "right" in self._resize_edge:
                geo.setWidth(max(self.MIN_SIZE, self._geometry_start.width() + delta.x()))
            if "bottom" in self._resize_edge:
                geo.setHeight(max(self.MIN_SIZE, self._geometry_start.height() + delta.y()))
            if "left" in self._resize_edge:
                new_left = self._geometry_start.left() + delta.x()
                new_width = self._geometry_start.width() - delta.x()
                if new_width >= self.MIN_SIZE:
                    geo.setLeft(new_left)
            if "top" in self._resize_edge:
                new_top = self._geometry_start.top() + delta.y()
                new_height = self._geometry_start.height() - delta.y()
                if new_height >= self.MIN_SIZE:
                    geo.setTop(new_top)

            self.setGeometry(geo)
            return

        # Update cursor based on hover position
        edge = self._get_resize_edge(pos)
        if edge:
            cursors = {
                "right": Qt.CursorShape.SizeHorCursor,
                "left": Qt.CursorShape.SizeHorCursor,
                "bottom": Qt.CursorShape.SizeVerCursor,
                "top": Qt.CursorShape.SizeVerCursor,
                "bottom-right": Qt.CursorShape.SizeFDiagCursor,
                "top-left": Qt.CursorShape.SizeFDiagCursor,
                "bottom-left": Qt.CursorShape.SizeBDiagCursor,
                "top-right": Qt.CursorShape.SizeBDiagCursor,
            }
            self.setCursor(cursors.get(edge, Qt.CursorShape.ArrowCursor))
        elif pos.y() <= self.TITLE_BAR_HEIGHT:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        self._dragging = False
        self._resizing = False
        self._resize_edge = None

    def _get_resize_edge(self, pos: QPoint) -> str | None:
        """Determine which resize edge the mouse is near."""
        b = self.BORDER_WIDTH
        w, h = self.width(), self.height()

        on_left = pos.x() <= b
        on_right = pos.x() >= w - b
        on_top = pos.y() <= b
        on_bottom = pos.y() >= h - b

        if on_bottom and on_right:
            return "bottom-right"
        if on_bottom and on_left:
            return "bottom-left"
        if on_top and on_right:
            return "top-right"
        if on_top and on_left:
            return "top-left"
        if on_right:
            return "right"
        if on_left:
            return "left"
        if on_bottom:
            return "bottom"
        if on_top and pos.y() > self.TITLE_BAR_HEIGHT:
            return "top"

        return None

    def keyPressEvent(self, event):
        """Handle key presses."""
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
