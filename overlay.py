# overlay.py - Transparent overlay window with live translation
import time
import traceback
import numpy as np
from PyQt6.QtCore import Qt, QTimer, QPoint, QRect, QThread, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QCursor
from PyQt6.QtWidgets import QWidget, QApplication

from capture import ScreenCapture
from ocr_engine import OCREngine, TextBlock
from translator import TranslationEngine
from renderer import TextRenderer
from config import load_config


class TranslationWorker(QThread):
    """Background thread for OCR + translation to avoid blocking UI."""

    result_ready = pyqtSignal(list, list)  # text_blocks, translated_texts

    def __init__(self, ocr_engine: OCREngine, translator: TranslationEngine):
        super().__init__()
        self._ocr = ocr_engine
        self._translator = translator
        self._image: np.ndarray | None = None
        self._running = True

    def set_image(self, image: np.ndarray):
        """Set the image to process."""
        self._image = image

    def run(self):
        """Process the image: OCR → translate."""
        if self._image is None:
            return

        try:
            # OCR
            blocks = self._ocr.detect(self._image)
            if not blocks:
                self.result_ready.emit([], [])
                return

            # Extract texts and batch translate
            texts = [b.text for b in blocks]
            translated = self._translator.translate_batch(texts)

            self.result_ready.emit(blocks, translated)
        except Exception:
            traceback.print_exc()
            self.result_ready.emit([], [])


class OverlayWindow(QWidget):
    """
    Transparent overlay window for screen translation.
    - Frameless, stays on top
    - Resizable by dragging edges
    - Movable by dragging title bar
    - Runs OCR + translation loop on captured screen beneath
    """

    TITLE_BAR_HEIGHT = 32
    BORDER_WIDTH = 6
    MIN_SIZE = 150

    def __init__(self, config: dict | None = None):
        super().__init__()

        self._config = config or load_config()
        self._capture = ScreenCapture()
        self._ocr = OCREngine()
        self._translator = TranslationEngine(self._config.get("target_language", "vi"))
        self._renderer = TextRenderer(
            font_family=self._config.get("font_family", "Segoe UI"),
            text_color=self._config.get("text_color", "#ffffff"),
            bg_color=self._config.get("background_color", "#1a1a2e"),
            font_size_min=self._config.get("font_size_min", 10),
            font_size_max=self._config.get("font_size_max", 36),
        )

        # Translation results
        self._text_blocks: list[TextBlock] = []
        self._translated_texts: list[str] = []
        self._is_translating = False
        self._worker: TranslationWorker | None = None

        # Drag/resize state
        self._dragging = False
        self._resizing = False
        self._resize_edge = None
        self._drag_start = QPoint()
        self._geometry_start = QRect()

        self._setup_ui()
        self._setup_timer()

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
        interval = self._config.get("capture_interval_ms", 1500)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._run_translation_cycle)
        self._timer.setInterval(interval)

    def showEvent(self, event):
        """Start translation loop when shown."""
        super().showEvent(event)
        self._timer.start()
        # Run first cycle immediately after a short delay
        QTimer.singleShot(300, self._run_translation_cycle)

    def hideEvent(self, event):
        """Stop translation loop when hidden."""
        super().hideEvent(event)
        self._timer.stop()
        self._text_blocks = []
        self._translated_texts = []

    def closeEvent(self, event):
        """Clean up resources."""
        self._timer.stop()
        self._capture.close()
        super().closeEvent(event)

    # ─── Translation Pipeline ─────────────────────────────────────────

    def _run_translation_cycle(self):
        """One cycle: capture → OCR → translate → repaint."""
        if self._is_translating or not self.isVisible():
            return

        self._is_translating = True

        try:
            # Get overlay geometry (global screen coordinates)
            geo = self.geometry()
            content_y = geo.y() + self.TITLE_BAR_HEIGHT
            content_h = geo.height() - self.TITLE_BAR_HEIGHT

            if content_h <= 10 or geo.width() <= 10:
                self._is_translating = False
                return

            # Temporarily hide to capture what's beneath
            self.setWindowOpacity(0)
            QApplication.processEvents()
            # Small delay to ensure the window is fully hidden
            time.sleep(0.05)

            # Capture the screen region
            image = self._capture.capture_region(
                geo.x(), content_y, geo.width(), content_h
            )

            # Show overlay again
            self.setWindowOpacity(1.0)
            QApplication.processEvents()

            # Run OCR + translation in background thread
            self._worker = TranslationWorker(self._ocr, self._translator)
            self._worker.result_ready.connect(self._on_translation_done)
            self._worker.set_image(image)
            self._worker.start()

        except Exception:
            traceback.print_exc()
            self.setWindowOpacity(1.0)
            self._is_translating = False

    def _on_translation_done(self, blocks: list, translated: list):
        """Handle translation results from worker thread."""
        self._text_blocks = blocks
        self._translated_texts = translated
        self._is_translating = False
        self.update()  # Trigger repaint

    # ─── Painting ─────────────────────────────────────────────────────

    def paintEvent(self, event):
        """Draw the overlay UI and translated text."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w, h = self.width(), self.height()
        border_color = QColor(self._config.get("border_color", "#e94560"))
        title_color = QColor(self._config.get("title_bar_color", "#16213e"))

        # ── Title bar ──
        title_rect = QRect(0, 0, w, self.TITLE_BAR_HEIGHT)
        title_bg = QColor(title_color)
        title_bg.setAlpha(220)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(title_bg))
        painter.drawRoundedRect(title_rect, 6, 6)
        # Square off bottom corners of title bar
        painter.drawRect(QRect(0, self.TITLE_BAR_HEIGHT - 6, w, 6))

        # Title text
        font = QFont("Segoe UI", 10, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QPen(QColor("#ffffff")))

        status = "⏳ Đang dịch..." if self._is_translating else f"✅ {len(self._text_blocks)} khối text"
        painter.drawText(
            QRect(12, 0, w - 80, self.TITLE_BAR_HEIGHT),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            f"🌐 Screen Translator  —  {status}",
        )

        # Close button
        close_rect = QRect(w - 36, 4, 28, 24)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#e94560")))
        painter.drawRoundedRect(close_rect, 4, 4)
        painter.setPen(QPen(QColor("#ffffff"), 2))
        painter.drawText(close_rect, Qt.AlignmentFlag.AlignCenter, "✕")

        # ── Content area - semi-transparent background ──
        content_rect = QRect(0, self.TITLE_BAR_HEIGHT, w, h - self.TITLE_BAR_HEIGHT)
        content_bg = QColor(0, 0, 0, 1)  # Nearly transparent
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(content_bg))
        painter.drawRect(content_rect)

        # ── Border ──
        painter.setPen(QPen(border_color, 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(QRect(1, 1, w - 2, h - 2), 6, 6)

        # ── Render translated text ──
        if self._text_blocks and self._translated_texts:
            self._renderer.render(
                painter,
                self._text_blocks,
                self._translated_texts,
                offset_x=0,
                offset_y=self.TITLE_BAR_HEIGHT,
            )

        # ── Resize handles (visual indicator) ──
        handle_color = QColor(border_color)
        handle_color.setAlpha(150)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(handle_color))
        # Bottom-right corner handle
        painter.drawRect(QRect(w - 12, h - 12, 10, 3))
        painter.drawRect(QRect(w - 12, h - 8, 10, 3))
        painter.drawRect(QRect(w - 6, h - 12, 3, 10))

        painter.end()

    # ─── Mouse Interaction ────────────────────────────────────────────

    def mousePressEvent(self, event):
        """Handle mouse press for dragging and resizing."""
        if event.button() != Qt.MouseButton.LeftButton:
            return

        pos = event.pos()

        # Check close button
        close_rect = QRect(self.width() - 36, 4, 28, 24)
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
