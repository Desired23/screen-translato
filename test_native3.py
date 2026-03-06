import sys
import sip
from PyQt6.QtWidgets import QApplication, QWidget

class TestWindow(QWidget):
    def nativeEvent(self, eventType, message):
        print("nativeEvent triggered", flush=True)
        return True, sip.voidptr(-1)

app = QApplication(sys.argv)
w = TestWindow()
w.show()
sys.exit(app.exec())
