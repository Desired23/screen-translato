# overlay.py - Transparent overlay window with live translation
import ctypes
import ctypes.wintypes
import traceback
import numpy as np
from PyQt6.sip import voidptr
from PyQt6.QtCore import Qt, QTimer, QPoint, QRect
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QCursor
from PyQt6.QtWidgets import QWidget, QApplication

from capture import ScreenCapture
from ocr_engine import OCREngine, TextBlock
from translator import TranslationEngine
from renderer import TextRenderer
from async_pipeline import AsyncTranslationPipeline, PipelineResult
from config import load_config



class OverlayContentWindow(QWidget):
    def __init__(self, parent_overlay):
        super().__init__()
        self._parent = parent_overlay
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def paintEvent(self, event):
        if not self._parent._text_blocks or not self._parent._translated_texts:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w, h = self.width(), self.height()
        t = self._parent.TITLE_BAR_HEIGHT
        
        # Content background: nearly transparent to allow reading but keep click-through
        content_rect = QRect(0, t, w, h - t)
        painter.fillRect(content_rect, QColor(0, 0, 0, 1))

        self._parent._renderer.render(
            painter,
            self._parent._text_blocks,
            self._parent._translated_texts,
            offset_x=0,
            offset_y=t,
        )
        painter.end()


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
        # Last dirty-frame sample (with overlay visible) for Mode B dedup
        self._last_dirty_sample: np.ndarray | None = None
        # Note: Mouse dragging/resizing state handles are no longer needed
        # Update: We now use a dual-window approach to prevent PyQt6 nativeEvent crashes 
        # while keeping the center fully click-through and the borders resizable.
        self._content = OverlayContentWindow(self)
        self._drag_start: QPoint | None = None
        self._resize_edge: str | None = None
        
        # Manual Capture / Freeze mode state
        self._is_frozen: bool = False

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
        if hasattr(self, "_content"):
            self._content.setGeometry(self.geometry())
            self._content.show()
        self._timer.start()
        # We explicitly set `_overlay_excluded` to False and omit SetWindowDisplayAffinity.
        # This allows screen capture tools (OBS, ShareX) to see the translated text.
        # The app will naturally fall back to "Mode B: opacity toggle" to prevent self-reading.
        self._overlay_excluded = False
        print("[Capture] Screen capture visibility enabled — using opacity fallback for OCR", flush=True)
        QTimer.singleShot(300, self._on_tick)

    def hideEvent(self, event):
        """Stop translation loop when hidden."""
        super().hideEvent(event)
        if hasattr(self, "_content"):
            self._content.hide()
        self._timer.stop()
        self._text_blocks = []
        self._translated_texts = []

    def closeEvent(self, event):
        """Clean up resources."""
        if hasattr(self, "_content"):
            self._content.close()
        self._timer.stop()
        self._pipeline.shutdown()
        self._capture.close()
        super().closeEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_content"):
            self._content.setGeometry(self.geometry())

    def moveEvent(self, event):
        super().moveEvent(event)
        if hasattr(self, "_content"):
            self._content.setGeometry(self.geometry())

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

        if self._is_frozen:
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
                dirty_sample = self._pipeline._fast_sample(dirty)
                if self._last_dirty_sample is not None:
                    try:
                        if dirty_sample.shape == self._last_dirty_sample.shape:
                            mad = np.mean(np.abs(dirty_sample - self._last_dirty_sample))
                            if mad < 3.0: # 3.0 threshold for minor noise/animation
                                return  # nothing changed — skip entirely, no flicker
                    except ValueError:
                        pass

                self._last_dirty_sample = dirty_sample

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
        
        # Ensure overlay is visible even if we previously hid it in Mode B
        self.setWindowOpacity(1.0)
        
        self.update()
        if hasattr(self, "_content"):
            self._content.update()

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

        # ── Toggle Freeze Button ──
        # Placed to the left of the close button
        freeze_rect = QRect(w - 56, 3, 22, 18)
        painter.setBrush(QBrush(QColor(100, 150, 255, 150) if self._is_frozen else QColor(100, 100, 100, 100)))
        painter.drawRoundedRect(freeze_rect, 3, 3)
        painter.setPen(QPen(QColor("#ffffff")))
        freeze_icon = "▶" if self._is_frozen else "⏸"
        painter.drawText(freeze_rect, Qt.AlignmentFlag.AlignCenter, freeze_icon)
        
        # ── Refresh Button ──
        # Placed to the left of the freeze button
        refresh_rect = QRect(w - 84, 3, 22, 18)
        painter.setBrush(QBrush(QColor(100, 100, 100, 100)))
        painter.drawRoundedRect(refresh_rect, 3, 3)
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(refresh_rect, Qt.AlignmentFlag.AlignCenter, "🔄")

        # ── Close button (small) ──
        close_rect = QRect(w - 28, 3, 22, 18)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(200, 60, 60, 150)))
        painter.drawRoundedRect(close_rect, 3, 3)
        close_font = QFont("Segoe UI", 8)
        painter.setFont(close_font)
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(close_rect, Qt.AlignmentFlag.AlignCenter, "✕")

        # ── Border — white, thin ──
        painter.setPen(QPen(QColor("#ffffff"), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(QRect(0, 0, w - 1, h - 1), 4, 4)

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
        """Handle mouse press inside the client area."""
        if event.button() != Qt.MouseButton.LeftButton:
            return

        x, y = event.pos().x(), event.pos().y()
        w, h = self.width(), self.height()
        b = self.BORDER_WIDTH
        t = self.TITLE_BAR_HEIGHT

        close_rect = QRect(w - 28, 3, 22, 18)
        if close_rect.contains(event.pos()):
            self.hide()
            return
            
        freeze_rect = QRect(w - 56, 3, 22, 18)
        if freeze_rect.contains(event.pos()):
            self._is_frozen = not self._is_frozen
            self.update()
            
            # If we just initiated a freeze, we force one immediate capture to lock the current frame
            if self._is_frozen:
                # Bypass the 'is_frozen' check temporarily to grab the exact moment they clicked 'freeze'
                self._is_frozen = False
                self._on_tick() # This will submit the current frame to the pipeline
                self._is_frozen = True # Re-engage freeze mode immediately
                
            return
            
        refresh_rect = QRect(w - 84, 3, 22, 18)
        if refresh_rect.contains(event.pos()):
            # Clear text blocks and force a scan
            self._text_blocks = []
            self._translated_texts = []
            
            # Briefly disable freeze to allow the tick to process
            was_frozen = self._is_frozen
            self._is_frozen = False
            self._on_tick() # Forces an immediate capture and submit
            self._is_frozen = was_frozen
            
            self.update()
            if hasattr(self, "_content"):
                self._content.update()
            return

        self._drag_start = event.globalPosition().toPoint()
        self._start_geo = self.geometry()

        on_left = x <= b
        on_right = x >= w - b
        on_top = y <= b
        on_bottom = y >= h - b

        if on_top and on_left: self._resize_edge = "top_left"
        elif on_top and on_right: self._resize_edge = "top_right"
        elif on_bottom and on_left: self._resize_edge = "bottom_left"
        elif on_bottom and on_right: self._resize_edge = "bottom_right"
        elif on_left: self._resize_edge = "left"
        elif on_right: self._resize_edge = "right"
        elif on_bottom: self._resize_edge = "bottom"
        elif y <= t: self._resize_edge = "title"
        else: self._resize_edge = None

    def mouseMoveEvent(self, event):
        x, y = event.pos().x(), event.pos().y()
        w, h = self.width(), self.height()
        b = self.BORDER_WIDTH

        if not getattr(self, "_resize_edge", None):
            if x <= b and y <= b: self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            elif x >= w - b and y >= h - b: self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            elif x >= w - b and y <= b: self.setCursor(Qt.CursorShape.SizeBDiagCursor)
            elif x <= b and y >= h - b: self.setCursor(Qt.CursorShape.SizeBDiagCursor)
            elif x <= b or x >= w - b: self.setCursor(Qt.CursorShape.SizeHorCursor)
            elif y <= b or y >= h - b: self.setCursor(Qt.CursorShape.SizeVerCursor)
            else: self.setCursor(Qt.CursorShape.ArrowCursor)

        if getattr(self, "_drag_start", None) and self._resize_edge:
            d = event.globalPosition().toPoint() - self._drag_start
            g = self._start_geo
            if self._resize_edge == "title":
                self.move(g.topLeft() + d)
            elif isinstance(self._resize_edge, str):
                new_g = QRect(g)
                if "left" in self._resize_edge:
                    new_g.setLeft(min(g.left() + d.x(), g.right() - self.MIN_SIZE))
                if "right" in self._resize_edge:
                    new_g.setRight(max(g.right() + d.x(), g.left() + self.MIN_SIZE))
                if "top" in self._resize_edge:
                    new_g.setTop(min(g.top() + d.y(), g.bottom() - self.MIN_SIZE))
                if "bottom" in self._resize_edge:
                    new_g.setBottom(max(g.bottom() + d.y(), g.top() + self.MIN_SIZE))
                self.setGeometry(new_g)

    def mouseReleaseEvent(self, event):
        self._drag_start = None
        self._resize_edge = None
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def keyPressEvent(self, event):
        """Handle key presses."""
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
