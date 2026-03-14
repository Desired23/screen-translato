# main.py - Entry point with system tray and global hotkey
import os
import sys
import ctypes
import ctypes.wintypes
import traceback
import threading
import faulthandler
import multiprocessing as mp
from datetime import datetime
from pathlib import Path


def _setup_frozen_logging():
    """Redirect stdout/stderr to AppData log file in packaged builds."""
    if not getattr(sys, "frozen", False):
        return
    try:
        appdata = os.getenv("APPDATA") or os.path.expanduser("~")
        log_dir = os.path.join(appdata, "ScreenTranslator", "logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = os.path.join(log_dir, f"app-{timestamp}-p{os.getpid()}.log")
        log_file = open(log_path, "w", encoding="utf-8", buffering=1)
        sys.stdout = log_file
        sys.stderr = log_file
        print(f"[Startup] Logging to {log_path}", flush=True)
    except Exception:
        pass


_setup_frozen_logging()


def _install_global_exception_hooks():
    """Keep crashes visible in frozen builds by forcing stack traces to log."""

    def _sys_hook(exc_type, exc_value, exc_tb):
        try:
            print("[Fatal] Unhandled exception in main thread", flush=True)
            traceback.print_exception(exc_type, exc_value, exc_tb)
        except Exception:
            pass

    def _thread_hook(args):
        try:
            print(
                f"[Fatal] Unhandled exception in thread {getattr(args, 'thread', None)}",
                flush=True,
            )
            traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback)
        except Exception:
            pass

    sys.excepthook = _sys_hook
    if hasattr(threading, "excepthook"):
        threading.excepthook = _thread_hook

    try:
        faulthandler.enable(file=sys.stderr, all_threads=True)
    except Exception:
        pass


_install_global_exception_hooks()

# Avoid eager torch import in packaged builds.
# It can trigger native access violations on some machines before UI startup.
if not getattr(sys, "frozen", False):
    try:
        import torch  # noqa: F401
    except Exception:
        pass

from pynput import keyboard as pynput_keyboard
from PyQt6.QtCore import Qt, QTimer, qInstallMessageHandler
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
    QColorDialog,
    QScrollArea,
    QWidget,
)

from config import load_config, save_config
from overlay import OverlayWindow

_SINGLE_INSTANCE_MUTEX = None


def _acquire_single_instance_lock() -> bool:
    """Prevent multiple running instances that duplicate tray icons."""
    global _SINGLE_INSTANCE_MUTEX
    if os.name != "nt":
        return True
    error_already_exists = 183
    mutex_names = [
        "Local\\ScreenTranslatorSingletonMutex",
        "Global\\ScreenTranslatorSingletonMutex",
    ]
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = (
        ctypes.wintypes.LPVOID,
        ctypes.wintypes.BOOL,
        ctypes.wintypes.LPCWSTR,
    )
    kernel32.CreateMutexW.restype = ctypes.wintypes.HANDLE
    kernel32.CloseHandle.argtypes = (ctypes.wintypes.HANDLE,)
    kernel32.CloseHandle.restype = ctypes.wintypes.BOOL

    for mutex_name in mutex_names:
        ctypes.set_last_error(0)
        handle = kernel32.CreateMutexW(None, False, mutex_name)
        last_error = ctypes.get_last_error()
        if not handle:
            continue
        if last_error == error_already_exists:
            kernel32.CloseHandle(handle)
            return False
        _SINGLE_INSTANCE_MUTEX = handle
        return True

    print("[Startup] Failed to create single-instance mutex.", flush=True)
    return True


# Source languages (for OCR - what's on screen)
SOURCE_LANGUAGES = {
    "en": "English",
    "ko": "Korean",
    "ja": "Japanese",
    "zh-CN": "Chinese (Simplified)",
    "zh-TW": "Chinese (Traditional)",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "th": "Thai",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
    "ar": "Arabic",
}
# Target languages (translation output)
TARGET_LANGUAGES = {
    "vi": "Vietnamese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "zh-CN": "Chinese (Simplified)",
    "zh-TW": "Chinese (Traditional)",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "th": "Thai",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
    "ar": "Arabic",
}

class HotkeyEdit(QLineEdit):
    """A line edit that captures keyboard shortcuts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setPlaceholderText("Press hotkey...")
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
        self.setText("Recording...")
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
        self._color_values: dict[str, str] = {}
        self._color_buttons: dict[str, QPushButton] = {}
        self._setup_ui()

    @staticmethod
    def _normalize_hex_color(value: str, default: str) -> str:
        color = QColor(str(value or ""))
        if not color.isValid():
            color = QColor(default)
        return color.name(QColor.NameFormat.HexRgb)

    def _runtime_path(self, raw_path: str) -> Path:
        path = Path(str(raw_path or "").strip()).expanduser()
        if path.is_absolute():
            return path
        bases: list[Path] = []
        if getattr(sys, "frozen", False):
            bases.append(Path(sys.executable).resolve().parent)
        bases.append(Path(__file__).resolve().parent)
        bases.append(Path.cwd())
        for base in bases:
            candidate = (base / path).resolve()
            if candidate.exists():
                return candidate
        return (bases[0] / path).resolve() if bases else path.resolve()

    def _refresh_color_button(self, key: str):
        btn = self._color_buttons.get(key)
        color = self._color_values.get(key, "#ffffff")
        if btn is None:
            return
        btn.setText(color.upper())
        btn.setStyleSheet(
            f"QPushButton {{ background: {color}; color: {'#000000' if QColor(color).lightness() > 150 else '#ffffff'}; "
            "border: 1px solid #2a2a4a; border-radius: 6px; padding: 6px 10px; font-weight: bold; }"
        )

    def _pick_color(self, key: str):
        initial = QColor(self._color_values.get(key, "#ffffff"))
        color = QColorDialog.getColor(initial, self, "Pick Color")
        if not color.isValid():
            return
        self._color_values[key] = color.name(QColor.NameFormat.HexRgb)
        self._refresh_color_button(key)

    def _validate_nllb_before_save(self) -> bool:
        backend = self._translation_backend_combo.currentData()
        if backend != "nllb":
            return True

        model_dir = self._runtime_path(self._config.get("nllb_model_dir", ".models/nllb-ct2-int8"))
        tok_path = self._runtime_path(self._config.get("nllb_tokenizer_path", ".models/nllb-ct2-int8"))
        errors: list[str] = []

        if not model_dir.is_dir():
            errors.append(f"- Model dir not found: {model_dir}")
        elif not (model_dir / "model.bin").exists():
            errors.append(f"- Missing file: {model_dir / 'model.bin'}")

        tokenizer_ok = False
        if tok_path.is_dir():
            tokenizer_ok = any(
                (tok_path / name).exists()
                for name in ("tokenizer.json", "sentencepiece.bpe.model")
            )
        elif tok_path.is_file():
            tokenizer_ok = tok_path.name in {"tokenizer.json", "sentencepiece.bpe.model"}
        if not tokenizer_ok:
            errors.append(
                "- Tokenizer path must contain tokenizer.json or sentencepiece.bpe.model "
                f"(current: {tok_path})"
            )

        if not errors:
            return True

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("NLLB Validation Failed")
        msg.setText("NLLB backend is not ready. Choose an action:")
        msg.setInformativeText("\n".join(errors))
        switch_btn = msg.addButton("Switch to AUTO and Save", QMessageBox.ButtonRole.AcceptRole)
        cancel_btn = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        msg.exec()
        if msg.clickedButton() == switch_btn:
            idx = self._translation_backend_combo.findData("auto")
            if idx >= 0:
                self._translation_backend_combo.setCurrentIndex(idx)
            self._config["translation_backend"] = "auto"
            return True
        return False

    def _setup_ui(self):
        self.setWindowTitle("Settings - Screen Translator")
        self.resize(480, 760)
        self.setMinimumSize(440, 680)
        self.setStyleSheet("""
            QDialog {
                background: #0f0f23;
                color: #ffffff;
            }
            QScrollArea#settingsScrollArea,
            QWidget#settingsScrollViewport,
            QWidget#settingsScrollContent {
                background: #0f0f23;
                border: none;
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
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background: #1a1a2e;
                border: 1px solid #2a2a4a;
                border-radius: 6px;
                color: #ffffff;
                padding: 6px 10px;
                min-height: 28px;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #00d2ff;
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
            QCheckBox {
                color: #d8d8d8;
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
            QPushButton {
                border-radius: 6px;
                padding: 10px 24px;
                font-weight: bold;
                font-size: 13px;
                min-height: 20px;
            }
        """)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(14, 14, 14, 14)
        outer_layout.setSpacing(12)

        scroll = QScrollArea(self)
        scroll.setObjectName("settingsScrollArea")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_content = QWidget()
        scroll_content.setObjectName("settingsScrollContent")
        scroll.viewport().setObjectName("settingsScrollViewport")
        layout = QVBoxLayout(scroll_content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        scroll.setWidget(scroll_content)
        outer_layout.addWidget(scroll, 1)

        hotkey_group = QGroupBox("Hotkey")
        hk_layout = QVBoxLayout(hotkey_group)
        hk_layout.addWidget(QLabel("Click the box and press your desired hotkey:"))
        self._hotkey_edit = HotkeyEdit()
        self._hotkey_edit.setText(self._config.get("hotkey", "ctrl+shift+t"))
        hk_layout.addWidget(self._hotkey_edit)
        layout.addWidget(hotkey_group)

        lang_group = QGroupBox("Language")
        ll = QVBoxLayout(lang_group)

        ll.addWidget(QLabel("Source language (on screen):"))
        self._source_lang_combo = QComboBox()
        for code, name in SOURCE_LANGUAGES.items():
            self._source_lang_combo.addItem(f"{name} ({code})", code)
        current_source = self._config.get("source_language", "ko")
        idx = self._source_lang_combo.findData(current_source)
        if idx >= 0:
            self._source_lang_combo.setCurrentIndex(idx)
        ll.addWidget(self._source_lang_combo)

        ll.addWidget(QLabel("Target language:"))
        self._target_lang_combo = QComboBox()
        for code, name in TARGET_LANGUAGES.items():
            self._target_lang_combo.addItem(f"{name} ({code})", code)
        current_target = self._config.get("target_language", "vi")
        idx = self._target_lang_combo.findData(current_target)
        if idx >= 0:
            self._target_lang_combo.setCurrentIndex(idx)
        ll.addWidget(self._target_lang_combo)

        ll.addWidget(QLabel("Translation backend:"))
        self._translation_backend_combo = QComboBox()
        self._translation_backend_combo.addItem("Auto (NLLB -> Argos -> Google)", "auto")
        self._translation_backend_combo.addItem("NLLB (Offline)", "nllb")
        self._translation_backend_combo.addItem("Google (Online)", "google")
        self._translation_backend_combo.addItem("Argos (Offline)", "argos")
        current_backend = self._config.get("translation_backend", "auto")
        idx = self._translation_backend_combo.findData(current_backend)
        if idx >= 0:
            self._translation_backend_combo.setCurrentIndex(idx)
        ll.addWidget(self._translation_backend_combo)

        self._translation_fallback_cb = QCheckBox("Allow fallback to Google if primary backend fails")
        self._translation_fallback_cb.setChecked(
            bool(self._config.get("translation_fallback_to_google", True))
        )
        ll.addWidget(self._translation_fallback_cb)

        ll.addWidget(QLabel("Argos pivot language:"))
        self._argos_pivot_combo = QComboBox()
        self._argos_pivot_combo.addItem("English (en)", "en")
        self._argos_pivot_combo.addItem("Japanese (ja)", "ja")
        self._argos_pivot_combo.addItem("Korean (ko)", "ko")
        self._argos_pivot_combo.addItem("Chinese (zh)", "zh")
        current_pivot = self._config.get("argos_pivot_language", "en")
        idx = self._argos_pivot_combo.findData(current_pivot)
        if idx >= 0:
            self._argos_pivot_combo.setCurrentIndex(idx)
        ll.addWidget(self._argos_pivot_combo)

        self._context_refine_cb = QCheckBox("Context refine for short bubbles")
        self._context_refine_cb.setChecked(
            bool(self._config.get("translation_context_refine_enabled", True))
        )
        ll.addWidget(self._context_refine_cb)

        ll.addWidget(QLabel("Context refine max chars:"))
        self._context_max_chars_spin = QSpinBox()
        self._context_max_chars_spin.setRange(16, 120)
        self._context_max_chars_spin.setSingleStep(4)
        self._context_max_chars_spin.setValue(
            int(self._config.get("translation_context_refine_max_chars", 48))
        )
        ll.addWidget(self._context_max_chars_spin)

        ll.addWidget(QLabel("Context refine max words:"))
        self._context_max_words_spin = QSpinBox()
        self._context_max_words_spin.setRange(3, 20)
        self._context_max_words_spin.setSingleStep(1)
        self._context_max_words_spin.setValue(
            int(self._config.get("translation_context_refine_max_words", 10))
        )
        ll.addWidget(self._context_max_words_spin)

        ll.addWidget(QLabel("Context refine max blocks per frame:"))
        self._context_max_per_batch_spin = QSpinBox()
        self._context_max_per_batch_spin.setRange(0, 12)
        self._context_max_per_batch_spin.setSingleStep(1)
        self._context_max_per_batch_spin.setValue(
            int(self._config.get("translation_context_refine_max_per_batch", 1))
        )
        ll.addWidget(self._context_max_per_batch_spin)

        self._game_term_guard_cb = QCheckBox("Preserve game terms (Basic Attack, Resonance Skill, ...)")
        self._game_term_guard_cb.setChecked(
            bool(self._config.get("translation_game_term_guard_enabled", True))
        )
        ll.addWidget(self._game_term_guard_cb)

        self._game_post_edit_cb = QCheckBox("Game post-edit (fix cast/dealing wording)")
        self._game_post_edit_cb.setChecked(
            bool(self._config.get("translation_game_post_edit_enabled", True))
        )
        ll.addWidget(self._game_post_edit_cb)

        self._auto_source_routing_cb = QCheckBox(
            "Auto source routing by script (EN/JA/KO/ZH)"
        )
        self._auto_source_routing_cb.setChecked(
            bool(self._config.get("translation_auto_source_routing_enabled", True))
        )
        ll.addWidget(self._auto_source_routing_cb)

        self._semantic_cache_cb = QCheckBox(
            "Semantic cache for near-identical OCR lines"
        )
        self._semantic_cache_cb.setChecked(
            bool(self._config.get("translation_semantic_cache_enabled", True))
        )
        ll.addWidget(self._semantic_cache_cb)

        layout.addWidget(lang_group)

        settings_group = QGroupBox("Runtime")
        sl = QVBoxLayout(settings_group)

        sl.addWidget(QLabel("Capture interval (ms):"))
        self._interval_spin = QSpinBox()
        self._interval_spin.setRange(500, 5000)
        self._interval_spin.setSingleStep(100)
        self._interval_spin.setValue(self._config.get("capture_interval_ms", 1500))
        self._interval_spin.setSuffix(" ms")
        sl.addWidget(self._interval_spin)

        self._drop_frames_busy_cb = QCheckBox("Drop new frames while pipeline is busy")
        self._drop_frames_busy_cb.setChecked(
            bool(self._config.get("drop_frames_when_busy", True))
        )
        sl.addWidget(self._drop_frames_busy_cb)

        self._auto_ui_line_mode_cb = QCheckBox("Auto UI/Game mode: force line-level merge")
        self._auto_ui_line_mode_cb.setChecked(
            bool(self._config.get("auto_ui_line_mode_enabled", True))
        )
        sl.addWidget(self._auto_ui_line_mode_cb)

        self._ui_panel_merge_cb = QCheckBox("UI/Game panel merge (keep related lines together)")
        self._ui_panel_merge_cb.setChecked(
            bool(self._config.get("ui_panel_merge_enabled", True))
        )
        sl.addWidget(self._ui_panel_merge_cb)

        self._auto_document_line_mode_cb = QCheckBox("Auto document mode: keep line-level blocks")
        self._auto_document_line_mode_cb.setChecked(
            bool(self._config.get("auto_document_line_mode_enabled", True))
        )
        sl.addWidget(self._auto_document_line_mode_cb)

        self._document_paragraph_cb = QCheckBox("Document mode: translate by paragraph")
        self._document_paragraph_cb.setChecked(
            bool(self._config.get("auto_document_translate_by_paragraph", True))
        )
        sl.addWidget(self._document_paragraph_cb)

        self._dark_ui_retry_cb = QCheckBox("Dark UI OCR retry (better yellow/bright text)")
        self._dark_ui_retry_cb.setChecked(
            bool(self._config.get("ocr_dark_ui_retry_enabled", True))
        )
        sl.addWidget(self._dark_ui_retry_cb)

        layout.addWidget(settings_group)

        display_group = QGroupBox("Display")
        dl = QVBoxLayout(display_group)
        dl.addWidget(QLabel("Overlay title bar color:"))
        self._color_values["title_bar_color"] = self._normalize_hex_color(
            self._config.get("title_bar_color", "#1a1a2e"), "#1a1a2e"
        )
        self._title_bar_btn = QPushButton()
        self._color_buttons["title_bar_color"] = self._title_bar_btn
        self._title_bar_btn.clicked.connect(lambda: self._pick_color("title_bar_color"))
        self._refresh_color_button("title_bar_color")
        dl.addWidget(self._title_bar_btn)

        dl.addWidget(QLabel("Overlay border color:"))
        self._color_values["border_color"] = self._normalize_hex_color(
            self._config.get("border_color", "#ffffff"), "#ffffff"
        )
        self._border_btn = QPushButton()
        self._color_buttons["border_color"] = self._border_btn
        self._border_btn.clicked.connect(lambda: self._pick_color("border_color"))
        self._refresh_color_button("border_color")
        dl.addWidget(self._border_btn)

        dl.addWidget(QLabel("Translation text color:"))
        self._color_values["text_color"] = self._normalize_hex_color(
            self._config.get("text_color", "#000000"), "#000000"
        )
        self._text_btn = QPushButton()
        self._color_buttons["text_color"] = self._text_btn
        self._text_btn.clicked.connect(lambda: self._pick_color("text_color"))
        self._refresh_color_button("text_color")
        dl.addWidget(self._text_btn)

        dl.addWidget(QLabel("Translation background color:"))
        self._color_values["background_color"] = self._normalize_hex_color(
            self._config.get("background_color", "#ffffff"), "#ffffff"
        )
        self._bg_btn = QPushButton()
        self._color_buttons["background_color"] = self._bg_btn
        self._bg_btn.clicked.connect(lambda: self._pick_color("background_color"))
        self._refresh_color_button("background_color")
        dl.addWidget(self._bg_btn)
        layout.addWidget(display_group)

        ocr_group = QGroupBox("OCR")
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

        info_row = QHBoxLayout()
        info_label = QLabel("Primary: RapidOCR -> Fallback: PaddleOCR")
        info_label.setStyleSheet("color: #00d2ff; font-size: 11px; font-style: italic;")
        info_row.addWidget(info_label)
        ol.addLayout(info_row)

        ol.addWidget(QLabel("Fallback confidence threshold (0.60 -> 0.95):"))
        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.60, 0.95)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setDecimals(2)
        self._conf_spin.setValue(self._config.get("confidence_thresh", 0.75))
        self._conf_spin.setToolTip(
            "If RapidOCR average confidence < threshold -> run PaddleOCR"
        )
        ol.addWidget(self._conf_spin)

        self._winrt_cb = QCheckBox("Enable WinRT OCR (requires Windows language packs)")
        self._winrt_cb.setChecked(bool(self._config.get("winrt_enabled", False)))
        self._winrt_cb.setToolTip(
            "WinRT is very fast but may be less accurate for CJK text."
        )
        ol.addWidget(self._winrt_cb)

        self._easyocr_cb = QCheckBox("Enable EasyOCR (rare languages, slower)")
        self._easyocr_cb.setChecked(bool(self._config.get("easyocr_enabled", False)))
        self._easyocr_cb.setToolTip(
            "EasyOCR is slower but supports many languages."
        )
        ol.addWidget(self._easyocr_cb)

        layout.addWidget(ocr_group)
        layout.addStretch(1)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
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

        save_btn = QPushButton("Save")
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
        outer_layout.addLayout(btn_layout)
    def _save(self):
        """Save settings and close."""
        self._config["hotkey"] = self._hotkey_edit.text()
        self._config["source_language"] = self._source_lang_combo.currentData()
        self._config["target_language"] = self._target_lang_combo.currentData()
        self._config["translation_backend"] = self._translation_backend_combo.currentData()
        self._config["translation_fallback_to_google"] = self._translation_fallback_cb.isChecked()
        self._config["argos_pivot_language"] = self._argos_pivot_combo.currentData()
        self._config["translation_context_refine_enabled"] = self._context_refine_cb.isChecked()
        self._config["translation_context_refine_max_chars"] = self._context_max_chars_spin.value()
        self._config["translation_context_refine_max_words"] = self._context_max_words_spin.value()
        self._config["translation_context_refine_max_per_batch"] = self._context_max_per_batch_spin.value()
        self._config["translation_game_term_guard_enabled"] = self._game_term_guard_cb.isChecked()
        self._config["translation_game_post_edit_enabled"] = self._game_post_edit_cb.isChecked()
        self._config["translation_auto_source_routing_enabled"] = (
            self._auto_source_routing_cb.isChecked()
        )
        self._config["translation_semantic_cache_enabled"] = self._semantic_cache_cb.isChecked()
        self._config["capture_interval_ms"] = self._interval_spin.value()
        self._config["drop_frames_when_busy"] = self._drop_frames_busy_cb.isChecked()
        self._config["auto_ui_line_mode_enabled"] = self._auto_ui_line_mode_cb.isChecked()
        self._config["ui_panel_merge_enabled"] = self._ui_panel_merge_cb.isChecked()
        self._config["auto_document_line_mode_enabled"] = self._auto_document_line_mode_cb.isChecked()
        self._config["auto_document_translate_by_paragraph"] = self._document_paragraph_cb.isChecked()
        self._config["ocr_dark_ui_retry_enabled"] = self._dark_ui_retry_cb.isChecked()
        self._config["confidence_thresh"] = self._conf_spin.value()
        self._config["winrt_enabled"] = self._winrt_cb.isChecked()
        self._config["easyocr_enabled"] = self._easyocr_cb.isChecked()
        self._config["title_bar_color"] = self._color_values.get("title_bar_color", "#1a1a2e")
        self._config["border_color"] = self._color_values.get("border_color", "#ffffff")
        self._config["text_color"] = self._color_values.get("text_color", "#000000")
        self._config["background_color"] = self._color_values.get("background_color", "#ffffff")
        if not self._validate_nllb_before_save():
            return
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
        self._qt_message_handler = None

        self._config = load_config()
        self._overlay: OverlayWindow | None = None
        self._tray: QSystemTrayIcon | None = None
        self._hotkey_listener = None
        self._pressed_keys = set()

        self._setup_qt_logging()
        self._setup_tray()
        self._register_hotkey()
        self._setup_autotest_hooks()

    def _setup_autotest_hooks(self):
        """Optional test hooks for packaged smoke tests."""
        auto_toggle = str(os.getenv("ST_AUTOTEST_TOGGLE_ON_START", "")).strip() == "1"
        if auto_toggle:
            QTimer.singleShot(
                1200,
                lambda: self._safe_call(self._toggle_overlay, "autotest-toggle"),
            )

        quit_after_raw = str(os.getenv("ST_AUTOTEST_QUIT_MS", "")).strip()
        if not quit_after_raw:
            return
        try:
            quit_after_ms = int(quit_after_raw)
        except ValueError:
            quit_after_ms = 0
        if quit_after_ms > 0:
            QTimer.singleShot(
                quit_after_ms,
                lambda: self._safe_call(self._quit, "autotest-quit"),
            )

    def _setup_qt_logging(self):
        """Route Qt warnings/errors into the same app log file."""

        def _handler(mode, context, message):
            level_map = {
                0: "QtDebug",
                1: "QtWarning",
                2: "QtCritical",
                3: "QtFatal",
                4: "QtInfo",
            }
            try:
                level = level_map.get(int(mode), "Qt")
            except Exception:
                level = "Qt"
            print(f"[{level}] {message}", flush=True)

        self._qt_message_handler = _handler
        qInstallMessageHandler(self._qt_message_handler)

    def _safe_call(self, fn, context: str):
        """Protect tray/UI callbacks from killing the process."""
        try:
            fn()
        except Exception:
            print(f"[UI] Callback failure: {context}", flush=True)
            traceback.print_exc()
            if self._tray is not None:
                self._tray.showMessage(
                    "Screen Translator",
                    f"Internal error in {context}. Check logs in %APPDATA%\\ScreenTranslator\\logs.",
                    QSystemTrayIcon.MessageIcon.Critical,
                    5000,
                )

    def _setup_tray(self):
        """Set up system tray icon and menu."""
        if self._tray is not None:
            try:
                self._tray.activated.disconnect()
            except Exception:
                pass
            self._tray.hide()
            self._tray.deleteLater()
            self._tray = None

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

        toggle_action = QAction(f"Toggle Overlay ({self._config['hotkey']})", self._app)
        toggle_action.triggered.connect(
            lambda _checked=False: self._safe_call(self._toggle_overlay, "tray-toggle")
        )

        settings_action = QAction("Settings", self._app)
        settings_action.triggered.connect(
            lambda _checked=False: self._safe_call(self._show_settings, "tray-settings")
        )

        quit_action = QAction("Quit", self._app)
        quit_action.triggered.connect(
            lambda _checked=False: self._safe_call(self._quit, "tray-quit")
        )

        menu.addAction(toggle_action)
        menu.addSeparator()
        menu.addAction(settings_action)
        menu.addSeparator()
        menu.addAction(quit_action)

        self._tray.setContextMenu(menu)
        self._tray.setToolTip("Screen Translator - Press " + self._config["hotkey"])
        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()
        print("[Tray] Icon initialized", flush=True)

        # Show notification
        self._tray.showMessage(
            "Screen Translator",
            f"App is running. Press {self._config['hotkey']} to toggle overlay.",
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
            self._freeze_keys = self._parse_hotkey("ctrl+shift+m")
            self._pressed_keys = set()

            def on_press(key):
                self._pressed_keys.add(key)
                # Normalize: check both left/right modifiers
                if self._check_hotkey_match(self._target_keys):
                    QTimer.singleShot(0, self._toggle_overlay)
                elif self._check_hotkey_match(self._freeze_keys):
                    QTimer.singleShot(0, self._toggle_freeze)

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

    def _check_hotkey_match(self, target_keys: set) -> bool:
        """Check if currently pressed keys match the target hotkey."""
        for target in target_keys:
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
        try:
            if self._overlay is None:
                try:
                    self._overlay = OverlayWindow(self._config)
                except Exception as e:
                    print(f"[Overlay] Failed to create overlay: {e}", flush=True)
                    self._tray.showMessage(
                        "Screen Translator",
                        "Overlay startup failed once. Retrying with current settings.",
                        QSystemTrayIcon.MessageIcon.Warning,
                        3500,
                    )
                    try:
                        self._overlay = OverlayWindow(self._config)
                    except Exception as e2:
                        print(f"[Overlay] Retry create overlay failed: {e2}", flush=True)
                        self._tray.showMessage(
                            "Screen Translator",
                            "Overlay failed to start. Check logs in %APPDATA%\\ScreenTranslator\\logs.",
                            QSystemTrayIcon.MessageIcon.Critical,
                            5000,
                        )
                        return

            if self._overlay.isVisible():
                self._overlay.hide()
            else:
                self._overlay.show()
                self._overlay.activateWindow()
        except Exception:
            print("[Overlay] Unexpected toggle failure", flush=True)
            traceback.print_exc()
            self._overlay = None
            if self._tray is not None:
                self._tray.showMessage(
                    "Screen Translator",
                    "Overlay crashed while toggling. Please check logs.",
                    QSystemTrayIcon.MessageIcon.Critical,
                    5000,
                )

    def _toggle_freeze(self):
        """Toggle freeze/manual mode on the active overlay."""
        if self._overlay and self._overlay.isVisible():
            self._overlay._is_frozen = not self._overlay._is_frozen
            self._overlay.update()
            
            if self._overlay._is_frozen:
                self._overlay._is_frozen = False
                self._overlay._on_tick()
                self._overlay._is_frozen = True

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
                try:
                    self._overlay = OverlayWindow(self._config)
                except Exception as e:
                    print(f"[Overlay] Failed to apply new settings: {e}", flush=True)
                    self._tray.showMessage(
                        "Screen Translator",
                        "Failed to apply new settings. Keeping your selected backend.",
                        QSystemTrayIcon.MessageIcon.Warning,
                        3500,
                    )
                    self._overlay = OverlayWindow(self._config)
                self._overlay.setGeometry(geo)
                if was_visible:
                    self._overlay.show()

    def _on_tray_activated(self, reason):
        """Handle tray icon activation."""
        print(f"[Tray] Activated: {reason}", flush=True)
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self._safe_call(self._toggle_overlay, "tray-activate")

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
    mp.freeze_support()
    if not _acquire_single_instance_lock():
        print("[Startup] Another instance is already running.", flush=True)
        return
    # Ensure high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = ScreenTranslatorApp()
    sys.exit(app.run())


if __name__ == "__main__":
    main()


