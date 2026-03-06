import sys
import ctypes
import ctypes.wintypes
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtWidgets import QApplication, QWidget

class TestWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setGeometry(100, 100, 400, 400)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

    def nativeEvent(self, eventType, message):
        try:
            msg = ctypes.wintypes.MSG.from_address(message.__int__())
            if msg.message == 0x0084: # NCHITTEST
                print("NCHITTEST intercepted!", flush=True)
                return True, -1 # HTTRANSPARENT
        except Exception as e:
            print(f"Error: {e}", flush=True)
            pass
        return super().nativeEvent(eventType, message)

app = QApplication(sys.argv)
w = TestWindow()
w.show()
sys.exit(app.exec())
