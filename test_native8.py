import sys
import ctypes
import ctypes.wintypes
from PyQt6.sip import voidptr
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtWidgets import QApplication, QWidget

class TestWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(100, 100, 400, 400)

    def nativeEvent(self, eventType, message):
        try:
            evt_bytes = bytes(eventType)
            if evt_bytes in (b"windows_generic_MSG", b"windows_dispatcher_MSG"):
                msg_ptr = int(message)
                if not msg_ptr:
                    return False, 0
                msg = ctypes.wintypes.MSG.from_address(msg_ptr)
                if msg.message == 0x0084:  # WM_NCHITTEST
                    return True, voidptr(-1)
        except Exception as e:
            print(f"Error: {e}", flush=True)
        return super().nativeEvent(eventType, message)

app = QApplication(sys.argv)
w = TestWindow()
w.show()
sys.exit(app.exec())
