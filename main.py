# main.py - Entry point with system tray and global hotkey
import sys
import keyboard
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon, QAction, QFont, QColor, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QSystemTrayIcon,
    QMenu,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QSpinBox,
    QPushButton,
    QGroupBox,
    QMessageBox,
    QWidget,
)

from config import load_config, save_config
from overlay import OverlayWindow


LANGUAGES = {
    "vi": "Tiếng Việt",
    "en": "English",
    "ja": "日本語",
    "ko": "한국어",
    "zh-CN": "中文 (简体)",
    "zh-TW": "中文 (繁體)",
    "fr": "Français",
    "de": "Deutsch",
    "es": "Español",
    "th": "ไทย",
    "ru": "Русский",
    "pt": "Português",
    "it": "Italiano",
    "ar": "العربية",
}


class HotkeyEdit(QLineEdit):
    """A line edit that captures keyboard shortcuts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setPlaceholderText("Nhấn phím tắt...")
        self.setStyleSheet("""
            QLineEdit {
                background: #2a2a4a;
                border: 2px solid #e94560;
                border-radius: 6px;
                color: #ffffff;
                padding: 8px 12px;
                font-size: 13px;
                font-weight: bold;
            }
            QLineEdit:focus {
                border-color: #00d2ff;
            }
        """)
        self._recording = False

    def mousePressEvent(self, event):
        """Start recording on click."""
        self._recording = True
        self.setText("⌨ Đang ghi phím...")
        self.setStyleSheet(self.styleSheet().replace("#e94560", "#00d2ff"))
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        """Capture the key combination."""
        if not self._recording:
            return

        key = event.key()
        modifiers = event.modifiers()

        if key in (Qt.Key.Key_Control, Qt.Key.Key_Shift, Qt.Key.Key_Alt, Qt.Key.Key_Meta):
            return  # Wait for the actual key

        parts = []
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            parts.append("ctrl")
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            parts.append("shift")
        if modifiers & Qt.KeyboardModifier.AltModifier:
            parts.append("alt")

        key_name = Qt.Key(key).name.decode() if isinstance(Qt.Key(key).name, bytes) else Qt.Key(key).name
        key_name = key_name.replace("Key_", "").lower()
        parts.append(key_name)

        hotkey = "+".join(parts)
        self.setText(hotkey)
        self._recording = False
        self.setStyleSheet(self.styleSheet().replace("#00d2ff", "#e94560"))


class SettingsDialog(QDialog):
    """Settings dialog for configuring the app."""

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._config = config.copy()
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("⚙ Cài đặt Screen Translator")
        self.setFixedSize(420, 480)
        self.setStyleSheet("""
            QDialog {
                background: #0f0f23;
                color: #ffffff;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                color: #e94560;
                border: 1px solid #2a2a4a;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 20px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
            QLabel {
                color: #cccccc;
                font-size: 12px;
            }
            QComboBox, QSpinBox {
                background: #1a1a2e;
                border: 1px solid #2a2a4a;
                border-radius: 6px;
                color: #ffffff;
                padding: 6px 10px;
                min-height: 28px;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            QComboBox QAbstractItemView {
                background: #1a1a2e;
                color: #ffffff;
                selection-background-color: #e94560;
            }
            QPushButton {
                border-radius: 6px;
                padding: 10px 24px;
                font-weight: bold;
                font-size: 13px;
                min-height: 20px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # ── Hotkey group ──
        hotkey_group = QGroupBox("⌨ Phím tắt")
        hk_layout = QVBoxLayout(hotkey_group)
        hk_layout.addWidget(QLabel("Click vào ô rồi nhấn tổ hợp phím mong muốn:"))
        self._hotkey_edit = HotkeyEdit()
        self._hotkey_edit.setText(self._config.get("hotkey", "ctrl+shift+t"))
        hk_layout.addWidget(self._hotkey_edit)
        layout.addWidget(hotkey_group)

        # ── Translation group ──
        trans_group = QGroupBox("🌐 Dịch thuật")
        tl = QVBoxLayout(trans_group)

        tl.addWidget(QLabel("Ngôn ngữ đích:"))
        self._lang_combo = QComboBox()
        for code, name in LANGUAGES.items():
            self._lang_combo.addItem(f"{name} ({code})", code)
        current_lang = self._config.get("target_language", "vi")
        idx = self._lang_combo.findData(current_lang)
        if idx >= 0:
            self._lang_combo.setCurrentIndex(idx)
        tl.addWidget(self._lang_combo)

        tl.addWidget(QLabel("Chu kỳ dịch (ms):"))
        self._interval_spin = QSpinBox()
        self._interval_spin.setRange(500, 5000)
        self._interval_spin.setSingleStep(100)
        self._interval_spin.setValue(self._config.get("capture_interval_ms", 1500))
        self._interval_spin.setSuffix(" ms")
        tl.addWidget(self._interval_spin)

        layout.addWidget(trans_group)

        # ── Buttons ──
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Hủy")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background: #2a2a4a;
                color: #cccccc;
            }
            QPushButton:hover {
                background: #3a3a5a;
            }
        """)
        cancel_btn.clicked.connect(self.reject)

        save_btn = QPushButton("💾 Lưu")
        save_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #e94560, stop:1 #ff6b6b);
                color: #ffffff;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff6b6b, stop:1 #e94560);
            }
        """)
        save_btn.clicked.connect(self._save)

        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)

    def _save(self):
        """Save settings and close."""
        self._config["hotkey"] = self._hotkey_edit.text()
        self._config["target_language"] = self._lang_combo.currentData()
        self._config["capture_interval_ms"] = self._interval_spin.value()
        save_config(self._config)
        self.accept()

    def get_config(self) -> dict:
        return self._config


def create_tray_icon() -> QPixmap:
    """Create a simple tray icon programmatically."""
    size = 64
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(0, 0, 0, 0))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    # Background circle
    gradient_start = QColor("#e94560")
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(gradient_start)
    painter.drawEllipse(2, 2, size - 4, size - 4)

    # "T" letter
    font = QFont("Segoe UI", 32, QFont.Weight.Bold)
    painter.setFont(font)
    painter.setPen(QColor("#ffffff"))
    painter.drawText(0, 0, size, size, Qt.AlignmentFlag.AlignCenter, "T")

    painter.end()
    return pixmap


class ScreenTranslatorApp:
    """Main application controller."""

    def __init__(self):
        self._app = QApplication(sys.argv)
        self._app.setQuitOnLastWindowClosed(False)

        self._config = load_config()
        self._overlay: OverlayWindow | None = None
        self._hotkey_registered = False

        self._setup_tray()
        self._register_hotkey()

    def _setup_tray(self):
        """Set up system tray icon and menu."""
        icon = QIcon(create_tray_icon())
        self._tray = QSystemTrayIcon(icon, self._app)

        menu = QMenu()
        menu.setStyleSheet("""
            QMenu {
                background: #0f0f23;
                color: #ffffff;
                border: 1px solid #2a2a4a;
                border-radius: 8px;
                padding: 4px;
            }
            QMenu::item {
                padding: 8px 24px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background: #e94560;
            }
        """)

        toggle_action = QAction(f"🌐 Bật/Tắt Overlay ({self._config['hotkey']})", self._app)
        toggle_action.triggered.connect(self._toggle_overlay)

        settings_action = QAction("⚙ Cài đặt", self._app)
        settings_action.triggered.connect(self._show_settings)

        quit_action = QAction("❌ Thoát", self._app)
        quit_action.triggered.connect(self._quit)

        menu.addAction(toggle_action)
        menu.addSeparator()
        menu.addAction(settings_action)
        menu.addSeparator()
        menu.addAction(quit_action)

        self._tray.setContextMenu(menu)
        self._tray.setToolTip("Screen Translator - Nhấn " + self._config["hotkey"])
        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()

        # Show notification
        self._tray.showMessage(
            "Screen Translator",
            f"Ứng dụng đang chạy! Nhấn {self._config['hotkey']} để bật overlay dịch.",
            QSystemTrayIcon.MessageIcon.Information,
            3000,
        )

    def _register_hotkey(self):
        """Register global hotkey."""
        try:
            if self._hotkey_registered:
                keyboard.unhook_all_hotkeys()
                self._hotkey_registered = False

            hotkey = self._config.get("hotkey", "ctrl+shift+t")
            keyboard.add_hotkey(hotkey, self._on_hotkey_pressed)
            self._hotkey_registered = True
        except Exception as e:
            print(f"Failed to register hotkey: {e}")

    def _on_hotkey_pressed(self):
        """Handle global hotkey press (called from keyboard thread)."""
        # Use QTimer to safely call from the main thread
        QTimer.singleShot(0, self._toggle_overlay)

    def _toggle_overlay(self):
        """Toggle the overlay window visibility."""
        if self._overlay is None:
            self._overlay = OverlayWindow(self._config)

        if self._overlay.isVisible():
            self._overlay.hide()
        else:
            self._overlay.show()
            self._overlay.activateWindow()

    def _show_settings(self):
        """Show settings dialog."""
        dialog = SettingsDialog(self._config)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            old_hotkey = self._config.get("hotkey")
            self._config = dialog.get_config()

            # Re-register hotkey if changed
            if self._config.get("hotkey") != old_hotkey:
                self._register_hotkey()
                # Update tray menu
                self._setup_tray()

            # Recreate overlay with new settings if it exists
            if self._overlay is not None:
                was_visible = self._overlay.isVisible()
                geo = self._overlay.geometry()
                self._overlay.close()
                self._overlay = OverlayWindow(self._config)
                self._overlay.setGeometry(geo)
                if was_visible:
                    self._overlay.show()

    def _on_tray_activated(self, reason):
        """Handle tray icon activation."""
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self._toggle_overlay()

    def _quit(self):
        """Quit the application."""
        if self._overlay:
            self._overlay.close()
        keyboard.unhook_all_hotkeys()
        self._tray.hide()
        self._app.quit()

    def run(self) -> int:
        """Run the application."""
        return self._app.exec()


def main():
    # Ensure high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = ScreenTranslatorApp()
    sys.exit(app.run())


if __name__ == "__main__":
    main()
