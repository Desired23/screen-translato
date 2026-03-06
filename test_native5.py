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
                msg = ctypes.wintypes.MSG.from_address(msg_ptr)
                if msg.message == 0x0084:  # WM_NCHITTEST
                    # TRY TO CRASH IT HERE
                    pos = self.mapFromGlobal(ctypes.wintypes.POINT(0,0))
                    # Wait, QPoint not imported. Let's return False, 0
                    return False, 0
        except Exception as e:
            pass
        return super().nativeEvent(eventType, message)

app = QApplication(sys.argv)
w = TestWindow()
w.show()
sys.exit(app.exec())
