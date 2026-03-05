# renderer.py - Renders translated text on the overlay
from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QPainter, QFont, QColor, QFontMetrics, QPen, QBrush
from ocr_engine import TextBlock


class TextRenderer:
    """Renders translated text blocks onto a QPainter surface."""

    def __init__(
        self,
        font_family: str = "Segoe UI",
        text_color: str = "#ffffff",
        bg_color: str = "#1a1a2e",
        font_size_min: int = 10,
        font_size_max: int = 36,
    ):
        self._font_family = font_family
        self._text_color = QColor(text_color)
        self._bg_color = QColor(bg_color)
        self._font_size_min = font_size_min
        self._font_size_max = font_size_max

    def render(
        self,
        painter: QPainter,
        text_blocks: list[TextBlock],
        translated_texts: list[str],
        offset_x: int = 0,
        offset_y: int = 0,
    ):
        """
        Render translated texts on the painter at the positions of original text blocks.
        
        Args:
            painter: QPainter to draw on
            text_blocks: Original OCR-detected text blocks
            translated_texts: Corresponding translated texts
            offset_x: X offset (for overlay positioning)
            offset_y: Y offset (for overlay positioning, e.g. title bar height)
        """
        if not text_blocks or not translated_texts:
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        for block, translated in zip(text_blocks, translated_texts):
            if not translated or not translated.strip():
                continue

            # Calculate draw rect from bounding box
            rect = QRectF(
                block.x + offset_x,
                block.y + offset_y,
                block.width,
                block.height,
            )

            # Expand rect slightly for padding
            padding = 3
            rect = rect.adjusted(-padding, -padding, padding, padding)

            # Calculate optimal font size to fit the text in the bounding box
            font_size = self._calculate_font_size(translated, rect)
            font = QFont(self._font_family, font_size)
            font.setWeight(QFont.Weight.Medium)

            # Draw background rectangle (to cover original text)
            bg_color = QColor(self._bg_color)
            bg_color.setAlpha(230)
            painter.setPen(QPen(QColor(self._bg_color.darker(120)), 1))
            painter.setBrush(QBrush(bg_color))
            painter.drawRoundedRect(rect, 3, 3)

            # Draw translated text
            painter.setFont(font)
            painter.setPen(QPen(self._text_color))
            painter.drawText(
                rect.adjusted(padding, padding, -padding, -padding),
                0x0001 | 0x0080 | 0x0100,  # AlignLeft | WordWrap | TextWordWrap
                translated,
            )

    def _calculate_font_size(self, text: str, rect: QRectF) -> int:
        """Calculate the best font size to fit text within the bounding rectangle."""
        target_height = rect.height() - 6  # padding

        # Start with a size proportional to the height
        best_size = max(self._font_size_min, min(int(target_height * 0.7), self._font_size_max))

        # Try to fit within width
        font = QFont(self._font_family, best_size)
        metrics = QFontMetrics(font)
        text_width = metrics.horizontalAdvance(text)

        if text_width > rect.width() - 6:
            # Scale down to fit width
            ratio = (rect.width() - 6) / max(text_width, 1)
            best_size = max(self._font_size_min, int(best_size * ratio))

        return best_size
