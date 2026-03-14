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
        Optimized with overlap resolution and content height auto-sizing.
        """
        if not text_blocks or not translated_texts:
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # We use MARGIN to prevent overlap, no internal PAD needed anymore
        PAD = 0
        MARGIN = 4
        MIN_READABLE_FONT = 13  # Minimum font size for readability
        MIN_BOX_WIDTH = 40      # Minimum width to prevent tall skinny boxes

        target_items = []

        # 1. Compute target rects and font sizes
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

            # Ensure minimal readability for width
            bw = max(bw, MIN_BOX_WIDTH)

            # Fit text but clamp to MIN_READABLE_FONT
            pixel_size = max(MIN_READABLE_FONT, self._fit_text_in_box(trans_text, bw, bh))

            font = QFont(self._font_family, -1)
            font.setPixelSize(pixel_size)
            font.setWeight(QFont.Weight.Normal)

            # Recalculate actual height needed with this font
            fm = QFontMetrics(font)
            br = fm.boundingRect(
                0, 0, int(bw), 0,
                Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignLeft,
                trans_text
            )

            # Actual bounding box needed (with padding)
            needed_h = max(bh, br.height())
            rect = QRectF(bx - PAD, by - PAD, bw + PAD * 2, needed_h + PAD * 2)

            target_items.append({
                "rect": rect,
                "text": trans_text,
                "font": font
            })

        # 2. Sort top-to-bottom
        target_items.sort(key=lambda item: item["rect"].y())

        # 3. Resolve overlaps
        for i in range(len(target_items)):
            while True:
                collided = False
                r_i = target_items[i]["rect"]
                for j in range(i):
                    r_j = target_items[j]["rect"]
                    # Add margin around r_j for separation
                    r_j_margin = r_j.adjusted(-MARGIN, -MARGIN, MARGIN, MARGIN)

                    if r_i.intersects(r_j_margin):
                        # Push r_i down just below r_j_margin
                        push_down = r_j_margin.bottom() - r_i.top()
                        r_i.translate(0, push_down)
                        collided = True
                if not collided:
                    break

        # 4. Draw
        for item in target_items:
            rect = item["rect"]
            font = item["font"]
            trans_text = item["text"]

            # Background
            bg = QColor(self._bg_color)
            if bg.alpha() == 255:
                bg.setAlpha(240)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(bg))
            painter.drawRoundedRect(rect, 4, 4)

            # Text
            painter.setFont(font)
            painter.setPen(QPen(QColor(self._text_color)))
            painter.drawText(
                rect,
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                trans_text,
            )
