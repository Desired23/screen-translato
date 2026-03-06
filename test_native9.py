import sys
from PyQt6.sip import voidptr
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QWidget

class TestWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(100, 100, 400, 400)

    def nativeEvent(self, eventType, message):
        return True, voidptr(-1)

app = QApplication(sys.argv)
w = TestWindow()
w.show()
sys.exit(app.exec())
