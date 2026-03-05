# capture.py - Fast screen capture using mss
import mss
import numpy as np
from PIL import Image


class ScreenCapture:
    """Captures a specific region of the screen using mss (fastest method)."""

    def __init__(self):
        self._sct = mss.mss()

    def capture_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        Capture a screen region and return as numpy array (RGB).
        
        Args:
            x: Left coordinate
            y: Top coordinate
            width: Width of the capture area
            height: Height of the capture area
            
        Returns:
            numpy array of shape (height, width, 3) in RGB format
        """
        monitor = {
            "left": x,
            "top": y,
            "width": width,
            "height": height,
        }
        screenshot = self._sct.grab(monitor)
        # Convert BGRA to RGB numpy array
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        return np.array(img)

    def capture_region_pil(self, x: int, y: int, width: int, height: int) -> Image.Image:
        """Capture a screen region and return as PIL Image."""
        monitor = {
            "left": x,
            "top": y,
            "width": width,
            "height": height,
        }
        screenshot = self._sct.grab(monitor)
        return Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

    def close(self):
        """Release resources."""
        self._sct.close()
