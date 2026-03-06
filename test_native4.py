import sys
from PyQt6.sip import voidptr
from PyQt6.QtWidgets import QApplication, QWidget

class TestWindow(QWidget):
    def nativeEvent(self, eventType, message):
        print("nativeEvent triggered", flush=True)
        return True, voidptr(-1)

app = QApplication(sys.argv)
w = TestWindow()
w.show()
sys.exit(app.exec())
