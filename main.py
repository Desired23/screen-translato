# main.py - Entry point with system tray and global hotkey
import sys

# IMPORTANT: Import torch BEFORE PyQt6 to avoid DLL conflict on Windows
# PyQt6 modifies DLL search paths which prevents torch's c10.dll from loading
try:
    import torch  # noqa: F401 - pre-load torch DLLs
except (ImportError, OSError):
    pass

from pynput import keyboard as pynput_keyboard
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
    QDoubleSpinBox,
    QCheckBox,
    QPushButton,
    QGroupBox,
    QMessageBox,
    QWidget,
)

from config import load_config, save_config
from overlay import OverlayWindow


# Source languages (for OCR - what's on screen)
SOURCE_LANGUAGES = {
    "en": "English",
    "ko": "한국어",
    "ja": "日本語",
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

# Target languages (translation output)
TARGET_LANGUAGES = {
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

        # ── Language group ──
        lang_group = QGroupBox("🌐 Ngôn ngữ")
        ll = QVBoxLayout(lang_group)

        ll.addWidget(QLabel("Ngôn ngữ gốc (trên màn hình):"))
        self._source_lang_combo = QComboBox()
        for code, name in SOURCE_LANGUAGES.items():
            self._source_lang_combo.addItem(f"{name} ({code})", code)
        current_source = self._config.get("source_language", "ko")
        idx = self._source_lang_combo.findData(current_source)
        if idx >= 0:
            self._source_lang_combo.setCurrentIndex(idx)
        ll.addWidget(self._source_lang_combo)

        ll.addWidget(QLabel("Ngôn ngữ dịch sang:"))
        self._target_lang_combo = QComboBox()
        for code, name in TARGET_LANGUAGES.items():
            self._target_lang_combo.addItem(f"{name} ({code})", code)
        current_target = self._config.get("target_language", "vi")
        idx = self._target_lang_combo.findData(current_target)
        if idx >= 0:
            self._target_lang_combo.setCurrentIndex(idx)
        ll.addWidget(self._target_lang_combo)

        layout.addWidget(lang_group)

        # ── Settings group ──
        settings_group = QGroupBox("⚙ Cài đặt")
        sl = QVBoxLayout(settings_group)

        sl.addWidget(QLabel("Chu kỳ dịch (ms):"))
        self._interval_spin = QSpinBox()
        self._interval_spin.setRange(500, 5000)
        self._interval_spin.setSingleStep(100)
        self._interval_spin.setValue(self._config.get("capture_interval_ms", 1500))
        self._interval_spin.setSuffix(" ms")
        sl.addWidget(self._interval_spin)

        layout.addWidget(settings_group)

        # ── OCR Engine group ──
        ocr_group = QGroupBox("🔍 OCR Engine")
        ocr_group.setStyleSheet("""
            QCheckBox {
                color: #cccccc;
                font-size: 12px;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 1px solid #2a2a4a;
                background: #1a1a2e;
            }
            QCheckBox::indicator:checked {
                background: #e94560;
                border-color: #e94560;
            }
        """)
        ol = QVBoxLayout(ocr_group)

        # Info label
        info_row = QHBoxLayout()
        info_label = QLabel("Primary: RapidOCR  →  Fallback: PaddleOCR")
        info_label.setStyleSheet("color: #00d2ff; font-size: 11px; font-style: italic;")
        info_row.addWidget(info_label)
        ol.addLayout(info_row)

        # Confidence threshold
        ol.addWidget(QLabel("Ngưỡng confidence fallback (0.60 → 0.95):"))
        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.60, 0.95)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setDecimals(2)
        self._conf_spin.setValue(self._config.get("confidence_thresh", 0.75))
        self._conf_spin.setToolTip(
            "Nếu avg confidence của RapidOCR < ngưỡng này → chạy PaddleOCR"
        )
        ol.addWidget(self._conf_spin)

        # Checkbox: WinRT
        self._winrt_cb = QCheckBox("Bật WinRT OCR (chỉ EN và một số ngôn ngữ có Language Pack)")
        self._winrt_cb.setChecked(bool(self._config.get("winrt_enabled", False)))
        self._winrt_cb.setToolTip(
            "WinRT rất nhanh nhưng accuracy thấp với CJK. Tắt mặc định."
        )
        ol.addWidget(self._winrt_cb)

        # Checkbox: EasyOCR
        self._easyocr_cb = QCheckBox("Bật EasyOCR (ngôn ngữ hiếm: Thai, Arabic, v.v.)")
        self._easyocr_cb.setChecked(bool(self._config.get("easyocr_enabled", False)))
        self._easyocr_cb.setToolTip(
            "EasyOCR chậm (~800ms) nhưng hỗ trợ nhiều ngôn ngữ nhất. Chỉ bật khi cần."
        )
        ol.addWidget(self._easyocr_cb)

        layout.addWidget(ocr_group)

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
        self._config["source_language"] = self._source_lang_combo.currentData()
        self._config["target_language"] = self._target_lang_combo.currentData()
        self._config["capture_interval_ms"] = self._interval_spin.value()
        self._config["confidence_thresh"] = self._conf_spin.value()
        self._config["winrt_enabled"] = self._winrt_cb.isChecked()
        self._config["easyocr_enabled"] = self._easyocr_cb.isChecked()
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
        self._hotkey_listener = None
        self._pressed_keys = set()

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
        self._tray.activated.connect(lambda _reason: self._toggle_overlay())
        self._tray.show()

        # Show notification
        self._tray.showMessage(
            "Screen Translator",
            f"Ứng dụng đang chạy! Nhấn {self._config['hotkey']} để bật overlay dịch.",
            QSystemTrayIcon.MessageIcon.Information,
            3000,
        )

    def _parse_hotkey(self, hotkey_str: str):
        """Parse hotkey string like 'ctrl+shift+t' into pynput keys."""
        key_map = {
            'ctrl': pynput_keyboard.Key.ctrl_l,
            'shift': pynput_keyboard.Key.shift_l,
            'alt': pynput_keyboard.Key.alt_l,
        }
        parts = hotkey_str.lower().split('+')
        keys = set()
        for part in parts:
            part = part.strip()
            if part in key_map:
                keys.add(key_map[part])
            elif len(part) == 1:
                keys.add(pynput_keyboard.KeyCode.from_char(part))
            else:
                # Try as Key attribute (e.g., 'f1', 'space')
                try:
                    keys.add(getattr(pynput_keyboard.Key, part))
                except AttributeError:
                    keys.add(pynput_keyboard.KeyCode.from_char(part))
        return keys

    def _register_hotkey(self):
        """Register global hotkey using pynput (no admin required)."""
        try:
            # Stop existing listener
            if self._hotkey_listener:
                self._hotkey_listener.stop()
                self._hotkey_listener = None

            hotkey = self._config.get("hotkey", "ctrl+shift+t")
            self._target_keys = self._parse_hotkey(hotkey)
            self._pressed_keys = set()

            def on_press(key):
                self._pressed_keys.add(key)
                # Normalize: check both left/right modifiers
                if self._check_hotkey_match():
                    QTimer.singleShot(0, self._toggle_overlay)

            def on_release(key):
                self._pressed_keys.discard(key)

            self._hotkey_listener = pynput_keyboard.Listener(
                on_press=on_press,
                on_release=on_release,
            )
            self._hotkey_listener.daemon = True
            self._hotkey_listener.start()
            print(f"[Hotkey] Registered: {hotkey}", flush=True)
        except Exception as e:
            print(f"[Hotkey] Failed to register: {e}", flush=True)

    def _check_hotkey_match(self) -> bool:
        """Check if currently pressed keys match the target hotkey."""
        for target in self._target_keys:
            matched = False
            for pressed in self._pressed_keys:
                if target == pressed:
                    matched = True
                    break
                # Handle left/right modifier variants
                if isinstance(target, pynput_keyboard.Key):
                    name = target.name
                    if name.endswith('_l'):
                        try:
                            right = getattr(pynput_keyboard.Key, name[:-2] + '_r')
                            if pressed == right:
                                matched = True
                                break
                        except AttributeError:
                            pass
                # Handle KeyCode case-insensitive match
                if isinstance(target, pynput_keyboard.KeyCode) and isinstance(pressed, pynput_keyboard.KeyCode):
                    if target.char and pressed.char and target.char.lower() == pressed.char.lower():
                        matched = True
                        break
            if not matched:
                return False
        return True

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
        try:
            if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
                self._toggle_overlay()
        except TypeError:
            # PyQt6 C++ enum conversion issue - just toggle on any activation
            self._toggle_overlay()

    def _quit(self):
        """Quit the application."""
        if self._overlay:
            self._overlay.close()
        if self._hotkey_listener:
            self._hotkey_listener.stop()
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
