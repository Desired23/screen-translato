import sys
from PyQt6.QtWidgets import QApplication, QWidget

class TestWindow(QWidget):
    def nativeEvent(self, eventType, message):
        return True, -1

app = QApplication(sys.argv)
w = TestWindow()
w.show()
sys.exit(app.exec())
