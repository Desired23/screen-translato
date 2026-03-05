# ocr_engine.py - OCR engine using EasyOCR (supports Python 3.13)
from dataclasses import dataclass
import numpy as np


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


class OCREngine:
    """EasyOCR-based text detection and recognition."""

    def __init__(self):
        self._engine = None

    def _ensure_engine(self):
        """Lazy-load the OCR engine (heavy import, downloads models on first run)."""
        if self._engine is None:
            import easyocr
            # Support common languages - English + script auto detection
            self._engine = easyocr.Reader(
                ["en", "vi", "ja", "ko", "zh_sim"],
                gpu=False,  # Use CPU for compatibility
                verbose=False,
            )

    def detect(self, image: np.ndarray, min_confidence: float = 0.3) -> list[TextBlock]:
        """
        Run OCR on an image and return detected text blocks.
        
        Args:
            image: numpy array (RGB) of the image
            min_confidence: minimum confidence threshold
            
        Returns:
            List of TextBlock with text, bounding box, and confidence
        """
        self._ensure_engine()

        results = self._engine.readtext(image)

        if not results:
            return []

        blocks = []
        for bbox, text, confidence in results:
            if confidence >= min_confidence and text.strip():
                # EasyOCR returns bbox as [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                blocks.append(TextBlock(
                    text=text.strip(),
                    bbox=bbox,
                    confidence=confidence,
                ))

        # Sort by position: top-to-bottom, then left-to-right
        blocks.sort(key=lambda b: (b.y, b.x))
        return blocks
