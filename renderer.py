# renderer.py - Renders translated text on the overlay
# Optimizations: binary-search font fit + LRU-like cache for font sizes
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPainter, QFont, QColor, QFontMetrics, QPen, QBrush
from ocr_engine import TextBlock


class TextRenderer:
    """Renders translated text blocks onto a QPainter surface."""

    def __init__(
        self,
        font_family: str = "Segoe UI",
        text_color: str = "#000000",
        bg_color: str = "#ffffff",
        font_size_min: int = 6,
        font_size_max: int = 36,
    ):
        self._font_family = font_family
        self._text_color = QColor(text_color)
        self._bg_color = QColor(bg_color)
        self._font_size_min = font_size_min
        self._font_size_max = font_size_max

        # (text_hash, box_w_bucket, box_h_bucket) → pixel_size
        # bucket size = 10px to increase hit rate across nearby sizes
        self._font_cache: dict[tuple, int] = {}
        self._CACHE_MAX = 500
        self._CACHE_EVICT = 100

    # ── Font fitting ──────────────────────────────────────────────────

    def _fit_text_in_box(self, text: str, box_w: float, box_h: float) -> int:
        """
        Binary-search the largest pixel size where `text` fits inside
        (box_w × box_h) with word-wrap. Result is cached.

        Complexity: O(log N) binary search (~5 steps) vs O(N) linear (~24 steps).
        Cache hit: O(1), skips search entirely.
        """
        if box_w <= 0 or box_h <= 0:
            return self._font_size_min

        # Cache key: (text hash, box dims rounded to 10px bucket)
        key = (hash(text), int(box_w) // 10, int(box_h) // 10)
        if key in self._font_cache:
            return self._font_cache[key]

        lo = self._font_size_min
        hi = self._font_size_max
        best = lo

        while lo <= hi:
            mid = (lo + hi) // 2
            font = QFont(self._font_family, -1)
            font.setPixelSize(mid)
            fm = QFontMetrics(font)
            br = fm.boundingRect(
                0, 0, int(box_w), 0,
                Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignLeft,
                text,
            )
            if br.height() <= box_h and br.width() <= box_w:
                best = mid    # fits → try bigger
                lo = mid + 1
            else:
                hi = mid - 1  # doesn't fit → try smaller

        # Store in cache (evict oldest entries when full)
        if len(self._font_cache) >= self._CACHE_MAX:
            oldest = list(self._font_cache.keys())[: self._CACHE_EVICT]
            for k in oldest:
                del self._font_cache[k]
        self._font_cache[key] = best

        return best

    # ── Rendering ─────────────────────────────────────────────────────

    def render(
        self,
        painter: QPainter,
        text_blocks: list[TextBlock],
        translated_texts: list[str],
        offset_x: int = 0,
        offset_y: int = 0,
    ):
        """
        Render each translated text at its original block position.

        Per block:
          1. Compute rect from bbox + offset
          2. Binary-search best font size (or hit cache)
          3. Draw white background + centered black text
        """
        if not text_blocks or not translated_texts:
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        PAD = 3

        for block, trans_text in zip(text_blocks, translated_texts):
            if not trans_text or not trans_text.strip():
                continue

            pts = block.bbox
            bx = min(p[0] for p in pts) + offset_x
            by = min(p[1] for p in pts) + offset_y
            bw = max(p[0] for p in pts) - min(p[0] for p in pts)
            bh = max(p[1] for p in pts) - min(p[1] for p in pts)

            if bw < 10 or bh < 8:
                continue

            rect = QRectF(bx - PAD, by - PAD, bw + PAD * 2, bh + PAD * 2)

            pixel_size = self._fit_text_in_box(trans_text, bw, bh)
            font = QFont(self._font_family, -1)
            font.setPixelSize(pixel_size)
            font.setWeight(QFont.Weight.Normal)

            # Background
            bg = QColor(255, 255, 255, 235)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(bg))
            painter.drawRoundedRect(rect, 3, 3)

            # Text
            painter.setFont(font)
            painter.setPen(QPen(QColor(0, 0, 0)))
            painter.drawText(
                rect,
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                trans_text,
            )
