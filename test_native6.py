import sys
import ctypes
import ctypes.wintypes
from PyQt6.QtWidgets import QApplication, QWidget

class TestWindow(QWidget):
    def nativeEvent(self, eventType, message):
        try:
            evt_bytes = bytes(eventType)
            if evt_bytes in (b"windows_generic_MSG", b"windows_dispatcher_MSG"):
                msg_ptr = int(message)
                if msg_ptr:
                    msg = ctypes.cast(msg_ptr, ctypes.POINTER(ctypes.wintypes.MSG)).contents
                    if msg.message == 0x0084:
                        return False, 0
        except Exception as e:
            print(f"Error: {e}", flush=True)
        return super().nativeEvent(eventType, message)

app = QApplication(sys.argv)
w = TestWindow()
w.show()
sys.exit(app.exec())
