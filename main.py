# main.py - Entry point with system tray and global hotkey
import os
import sys
import ctypes
import ctypes.wintypes
import traceback
import threading
import faulthandler
import multiprocessing as mp
import time
from datetime import datetime
from pathlib import Path


class _TeeStream:
    def __init__(self, *streams):
        self._streams = [s for s in streams if s is not None]

    def write(self, data):
        for stream in self._streams:
            try:
                stream.write(data)
                stream.flush()
            except Exception:
                pass
        return len(data)

    def flush(self):
        for stream in self._streams:
            try:
                stream.flush()
            except Exception:
                pass

    def isatty(self):
        for stream in self._streams:
            try:
                if stream.isatty():
                    return True
            except Exception:
                continue
        return False


def _setup_runtime_logging():
    """Mirror stdout/stderr to a log file for packaged builds and opt-in dev runs."""
    dev_log_opt = str(os.getenv("ST_LOG_TO_FILE", "1")).strip().lower()
    want_dev_log = dev_log_opt not in {"0", "false", "no", "off"}
    if not getattr(sys, "frozen", False) and not want_dev_log:
        return
    try:
        if getattr(sys, "frozen", False):
            appdata = os.getenv("APPDATA") or os.path.expanduser("~")
            log_dir = Path(appdata) / "ScreenTranslator" / "logs"
        else:
            log_dir = Path(__file__).resolve().parent / ".logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = log_dir / f"app-{timestamp}-p{os.getpid()}.log"
        log_file = open(log_path, "w", encoding="utf-8", buffering=1)
        sys.stdout = _TeeStream(sys.__stdout__, log_file)
        sys.stderr = _TeeStream(sys.__stderr__, log_file)
        print(f"[Startup] Logging to {log_path}", flush=True)
    except Exception:
        pass


_setup_runtime_logging()


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

AUTHOR_CREDIT_MESSAGE = (
    "APP NÀY ĐƯỢC VIẾT BỞI BỐ HUY "
    "https://github.com/Desired23/screen-translato"
)


def _preload_native_ocr_runtime():
    """Preload onnxruntime/RapidOCR before PyQt6 to avoid DLL init conflicts."""
    try:
        import onnxruntime as ort

        print(f"[Startup] Preloaded onnxruntime {ort.__version__}", flush=True)
    except Exception as exc:
        print(f"[Startup] onnxruntime preload failed: {exc}", flush=True)
        return

    try:
        from rapidocr_onnxruntime import RapidOCR

        _ = RapidOCR
        print("[Startup] Preloaded rapidocr_onnxruntime", flush=True)
    except Exception as exc:
        print(f"[Startup] RapidOCR preload failed: {exc}", flush=True)


_preload_native_ocr_runtime()

from pynput import keyboard as pynput_keyboard
from PyQt6.QtCore import Qt, QTimer, qInstallMessageHandler, QObject, pyqtSignal
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
    suffix_raw = str(os.getenv("ST_SINGLE_INSTANCE_SUFFIX", "")).strip()
    safe_suffix = ""
    if suffix_raw:
        safe_suffix = "_" + "".join(ch for ch in suffix_raw if ch.isalnum() or ch in {"_", "-"})
    mutex_names = [
        f"Local\\ScreenTranslatorSingletonMutex{safe_suffix}",
        f"Global\\ScreenTranslatorSingletonMutex{safe_suffix}",
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
    "auto": "Auto Detect",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "zh-CN": "Chinese (Simplified)",
    "zh-TW": "Chinese (Traditional)",
}
# Target languages (translation output)
TARGET_LANGUAGES = {
    "vi": "Vietnamese",
}


class _HotkeyDispatcher(QObject):
    toggle_overlay = pyqtSignal()
    toggle_freeze = pyqtSignal()


class _WindowsGlobalHotkeys:
    WM_HOTKEY = 0x0312
    WM_QUIT = 0x0012
    MOD_ALT = 0x0001
    MOD_CONTROL = 0x0002
    MOD_SHIFT = 0x0004
    MOD_WIN = 0x0008

    _KEY_MAP = {
        "space": 0x20,
        "tab": 0x09,
        "enter": 0x0D,
        "esc": 0x1B,
        "escape": 0x1B,
        "up": 0x26,
        "down": 0x28,
        "left": 0x25,
        "right": 0x27,
        "home": 0x24,
        "end": 0x23,
        "pageup": 0x21,
        "pagedown": 0x22,
        "insert": 0x2D,
        "delete": 0x2E,
    }

    def __init__(self):
        self._thread = None
        self._thread_id = 0
        self._callbacks: dict[int, callable] = {}
        self._active_ids: list[int] = []
        self._start_event = threading.Event()
        self._start_ok = False
        self._start_error = "not started"

    @classmethod
    def _parse_hotkey(cls, hotkey_str: str) -> tuple[int, int]:
        parts = [p.strip().lower() for p in str(hotkey_str or "").split("+") if p.strip()]
        if not parts:
            raise ValueError("empty hotkey")

        modifiers = 0
        key_token = None
        for token in parts:
            if token in {"ctrl", "control"}:
                modifiers |= cls.MOD_CONTROL
            elif token == "shift":
                modifiers |= cls.MOD_SHIFT
            elif token == "alt":
                modifiers |= cls.MOD_ALT
            elif token in {"win", "meta"}:
                modifiers |= cls.MOD_WIN
            else:
                key_token = token

        if not key_token:
            raise ValueError("missing key in hotkey")

        if len(key_token) == 1:
            ch = key_token.upper()
            if "A" <= ch <= "Z" or "0" <= ch <= "9":
                return modifiers, ord(ch)
            raise ValueError(f"unsupported key: {key_token}")

        if key_token.startswith("f") and key_token[1:].isdigit():
            fnum = int(key_token[1:])
            if 1 <= fnum <= 24:
                return modifiers, 0x70 + (fnum - 1)

        vk = cls._KEY_MAP.get(key_token)
        if vk is None:
            raise ValueError(f"unsupported key: {key_token}")
        return modifiers, vk

    def start(self, hotkey_defs: dict[int, tuple[str, callable]]) -> tuple[bool, str]:
        self.stop()
        self._callbacks = {hid: cb for hid, (_s, cb) in hotkey_defs.items()}
        self._active_ids = []
        self._start_event.clear()
        self._start_ok = False
        self._start_error = "init"

        user32 = ctypes.windll.user32

        def _worker():
            self._thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
            msg = ctypes.wintypes.MSG()
            # Ensure the message queue exists for this thread.
            user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 0)
            try:
                for hotkey_id, (hotkey_str, _cb) in hotkey_defs.items():
                    modifiers, vk = self._parse_hotkey(hotkey_str)
                    if not user32.RegisterHotKey(None, int(hotkey_id), int(modifiers), int(vk)):
                        err = ctypes.get_last_error()
                        raise RuntimeError(
                            f"RegisterHotKey failed for '{hotkey_str}' (id={hotkey_id}, winerr={err})"
                        )
                    self._active_ids.append(int(hotkey_id))
                self._start_ok = True
                self._start_error = ""
            except Exception as exc:
                self._start_ok = False
                self._start_error = str(exc)
            finally:
                self._start_event.set()

            if not self._start_ok:
                for hid in self._active_ids:
                    try:
                        user32.UnregisterHotKey(None, int(hid))
                    except Exception:
                        pass
                self._active_ids.clear()
                return

            while True:
                rv = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if rv <= 0:
                    break
                if msg.message == self.WM_HOTKEY:
                    hid = int(msg.wParam)
                    cb = self._callbacks.get(hid)
                    if cb is not None:
                        try:
                            cb()
                        except Exception:
                            traceback.print_exc()

            for hid in self._active_ids:
                try:
                    user32.UnregisterHotKey(None, int(hid))
                except Exception:
                    pass
            self._active_ids.clear()

        self._thread = threading.Thread(target=_worker, daemon=True, name="global-hotkey-win32")
        self._thread.start()
        self._start_event.wait(timeout=3.0)
        if not self._start_event.is_set():
            return False, "timeout while starting hotkey thread"
        return self._start_ok, (self._start_error or "ok")

    def stop(self):
        if self._thread is None:
            return
        try:
            if self._thread_id:
                ctypes.windll.user32.PostThreadMessageW(int(self._thread_id), self.WM_QUIT, 0, 0)
        except Exception:
            pass
        self._thread.join(timeout=1.5)
        self._thread = None
        self._thread_id = 0
        self._callbacks = {}
        self._active_ids = []


class _PynputGlobalHotkeys:
    def __init__(self):
        self._listener = None

    @staticmethod
    def _to_spec(hotkey_str: str) -> str:
        parts = [p.strip().lower() for p in str(hotkey_str or "").split("+") if p.strip()]
        converted = []
        for token in parts:
            if token in {"ctrl", "control"}:
                converted.append("<ctrl>")
            elif token == "shift":
                converted.append("<shift>")
            elif token == "alt":
                converted.append("<alt>")
            elif token in {"win", "meta"}:
                converted.append("<cmd>")
            else:
                converted.append(token)
        return "+".join(converted)

    def start(self, hotkey_defs: dict[int, tuple[str, callable]]) -> tuple[bool, str]:
        self.stop()
        bindings: dict[str, callable] = {}
        for _hid, (hotkey_str, cb) in hotkey_defs.items():
            spec = self._to_spec(hotkey_str)
            bindings[spec] = cb
        try:
            self._listener = pynput_keyboard.GlobalHotKeys(bindings)
            self._listener.start()
            return True, "ok"
        except Exception as exc:
            self._listener = None
            return False, str(exc)

    def stop(self):
        if self._listener is None:
            return
        try:
            self._listener.stop()
        except Exception:
            pass
        self._listener = None

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

        source_lang = self._source_lang_combo.currentData()
        model_dir = self._runtime_path(self._config.get("nllb_model_dir", ".models/nllb-ct2-int8"))
        tok_path = self._runtime_path(self._config.get("nllb_tokenizer_path", ".models/nllb-ct2-int8"))
        errors: list[str] = []

        if source_lang in {"auto", "", None}:
            errors.append(
                "- NLLB requires explicit source language (not Auto Detect)."
            )

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
        current_source = self._config.get("source_language", "auto")
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
        self._translation_backend_combo.addItem("Auto (NLLB -> Google)", "auto")
        self._translation_backend_combo.addItem("NLLB (Offline)", "nllb")
        self._translation_backend_combo.addItem("Google (Online)", "google")
        current_backend = self._config.get("translation_backend", "auto")
        if current_backend == "argos":
            current_backend = "auto"
        idx = self._translation_backend_combo.findData(current_backend)
        if idx >= 0:
            self._translation_backend_combo.setCurrentIndex(idx)
        ll.addWidget(self._translation_backend_combo)

        self._translation_fallback_cb = QCheckBox("Allow fallback to Google if primary backend fails")
        self._translation_fallback_cb.setChecked(
            bool(self._config.get("translation_fallback_to_google", True))
        )
        ll.addWidget(self._translation_fallback_cb)

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

        guide_group = QGroupBox("Huong Dan Su Dung Chi Tiet")
        gl = QVBoxLayout(guide_group)
        guide_label = QLabel(
            "1) Bat/Tat overlay:\n"
            "   - Nhan hotkey trong Runtime (mac dinh: ctrl+shift+t).\n"
            "   - Hoac click Tray > Toggle Overlay.\n\n"
            "2) Doi phim nong:\n"
            "   - Vao Settings > Hotkey, click o hotkey roi bam to hop moi.\n"
            "   - Bam Save de ap dung ngay.\n\n"
            "3) Chon backend dich:\n"
            "   - NLLB (Offline): dich local, on dinh khi da co model.\n"
            "   - Google (Online): can internet, thuong cho ket qua muot.\n\n"
            "4) Diagnostics:\n"
            "   - Tray > Diagnostics de xem OCR backend, MT backend, hotkey backend,\n"
            "     va trang thai dang ky hotkey.\n\n"
            "5) Khac phuc su co nhanh:\n"
            "   - Neu hotkey khong an duoc: mo Diagnostics, kiem tra hotkey_backend.\n"
            "   - Neu OCR yeu: thu bat WinRT OCR hoac dieu chinh confidence.\n"
            "   - Neu dich cham: dung NLLB local, giam capture interval, bat cache.\n\n"
            "6) Chu y:\n"
            "   - Ctrl+Shift+M dung de Freeze/Resume frame hien tai."
        )
        guide_label.setWordWrap(True)
        guide_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        guide_label.setStyleSheet("color: #d8d8e8; font-size: 12px; line-height: 1.35;")
        gl.addWidget(guide_label)
        layout.addWidget(guide_group)
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


class StartupStatusOverlay(QWidget):
    """Lightweight status window shown while heavy overlay startup is in progress."""

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        panel = QWidget(self)
        panel.setStyleSheet(
            """
            background: rgba(12, 16, 28, 230);
            color: #f5f8ff;
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 12px;
            """
        )
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(16, 12, 16, 12)
        panel_layout.setSpacing(4)

        self._title = QLabel("Screen Translator")
        self._title.setStyleSheet("font-weight: 700; font-size: 13px;")

        self._status = QLabel("Dang khoi tao...")
        self._status.setStyleSheet("font-size: 12px; color: #d6def0;")

        panel_layout.addWidget(self._title)
        panel_layout.addWidget(self._status)
        root.addWidget(panel)

        self.setFixedSize(320, 84)

    def show_status(self, message: str):
        self._status.setText(message)
        self._center_on_screen()
        self.show()
        self.raise_()

    def _center_on_screen(self):
        screen = QApplication.primaryScreen()
        if not screen:
            return
        geo = screen.availableGeometry()
        x = geo.x() + (geo.width() - self.width()) // 2
        y = geo.y() + (geo.height() - self.height()) // 2
        self.move(x, y)


class ScreenTranslatorApp:
    """Main application controller."""

    def __init__(self):
        self._app = QApplication(sys.argv)
        self._app.setQuitOnLastWindowClosed(False)
        self._qt_message_handler = None
        # Qt tray balloon windows are unstable on some frozen Windows runtimes.
        # Hard-disable in release runtime to avoid QtTrayIconMessageWindow crashes.
        self._tray_messages_enabled = not getattr(sys, "frozen", False)

        self._config = load_config()
        self._overlay: OverlayWindow | None = None
        self._overlay_initializing = False
        self._startup_status = StartupStatusOverlay()
        self._tray: QSystemTrayIcon | None = None
        self._hotkey_backend = "none"
        self._hotkey_status = "not registered"
        self._hotkey_native = _WindowsGlobalHotkeys()
        self._hotkey_fallback = _PynputGlobalHotkeys()
        self._hotkey_dispatcher = _HotkeyDispatcher()
        self._hotkey_dispatcher.toggle_overlay.connect(
            lambda: self._safe_call(self._toggle_overlay, "hotkey-toggle")
        )
        self._hotkey_dispatcher.toggle_freeze.connect(
            lambda: self._safe_call(self._toggle_freeze, "hotkey-freeze")
        )

        self._setup_qt_logging()
        self._setup_tray()
        self._notify_starting()
        self._register_hotkey()
        self._notify_ready()
        self._setup_autotest_hooks()

    def _tray_show_message(self, title: str, message: str, icon, timeout_ms: int = 4000):
        if self._tray is None:
            return
        if not self._tray_messages_enabled:
            print(f"[Tray] Message skipped (disabled): {title} | {message}", flush=True)
            return
        try:
            self._tray.showMessage(title, message, icon, int(timeout_ms))
        except Exception as exc:
            print(f"[Tray] showMessage failed: {exc}", flush=True)

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
                self._tray_show_message(
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

        diagnostics_action = QAction("Diagnostics", self._app)
        diagnostics_action.triggered.connect(
            lambda _checked=False: self._safe_call(
                self._show_diagnostics, "tray-diagnostics"
            )
        )

        quit_action = QAction("Quit", self._app)
        quit_action.triggered.connect(
            lambda _checked=False: self._safe_call(self._quit, "tray-quit")
        )

        menu.addAction(toggle_action)
        menu.addSeparator()
        menu.addAction(settings_action)
        menu.addAction(diagnostics_action)
        menu.addSeparator()
        menu.addAction(quit_action)

        self._tray.setContextMenu(menu)
        self._tray.setToolTip("Screen Translator - Starting...")
        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()
        print(
            f"[Tray] Icon initialized (messages_enabled={self._tray_messages_enabled})",
            flush=True,
        )

        # Author credit message
        self._tray_show_message(
            "Screen Translator",
            AUTHOR_CREDIT_MESSAGE,
            QSystemTrayIcon.MessageIcon.Information,
            5000,
        )

    def _notify_starting(self):
        if self._tray is None:
            return
        self._tray.setToolTip("Screen Translator - Dang khoi dong...")
        self._tray_show_message(
            "Screen Translator",
            "App dang khoi dong, vui long cho...",
            QSystemTrayIcon.MessageIcon.Information,
            4500,
        )

    def _notify_ready(self):
        if self._tray is None:
            return
        hotkey = self._config.get("hotkey", "ctrl+shift+t")
        self._tray.setToolTip(
            f"Screen Translator - Ready ({hotkey}) [{self._hotkey_backend}]"
        )
        self._tray_show_message(
            "Screen Translator",
            f"Khoi dong thanh cong. Nhan {hotkey} de bat/tat overlay.",
            QSystemTrayIcon.MessageIcon.Information,
            5000,
        )

    def _register_hotkey(self):
        """Register global hotkeys with native Win32 first, then fallback to pynput."""
        self._hotkey_native.stop()
        self._hotkey_fallback.stop()

        main_hotkey = self._config.get("hotkey", "ctrl+shift+t")
        freeze_hotkey = "ctrl+shift+m"
        hotkey_defs = {
            1: (main_hotkey, self._hotkey_dispatcher.toggle_overlay.emit),
            2: (freeze_hotkey, self._hotkey_dispatcher.toggle_freeze.emit),
        }

        ok_native, native_reason = self._hotkey_native.start(hotkey_defs)
        if ok_native:
            self._hotkey_backend = "win32"
            self._hotkey_status = "registered"
            print(f"[Hotkey] Registered via win32: {main_hotkey}", flush=True)
            return

        ok_fallback, fallback_reason = self._hotkey_fallback.start(hotkey_defs)
        if ok_fallback:
            self._hotkey_backend = "pynput"
            self._hotkey_status = f"win32 failed: {native_reason}"
            print(
                f"[Hotkey] Win32 failed, using pynput. reason={native_reason}",
                flush=True,
            )
            return

        self._hotkey_backend = "none"
        self._hotkey_status = (
            f"win32 failed: {native_reason}; pynput failed: {fallback_reason}"
        )
        print(f"[Hotkey] Failed: {self._hotkey_status}", flush=True)
        if self._tray is not None:
            self._tray_show_message(
                "Screen Translator",
                "Khong dang ky duoc hotkey. Mo Diagnostics de xem chi tiet.",
                QSystemTrayIcon.MessageIcon.Warning,
                6000,
            )

    def _toggle_overlay(self):
        """Toggle the overlay window visibility."""
        try:
            print("[HotkeyCase] toggle_overlay:begin", flush=True)
            if self._overlay_initializing:
                print("[Overlay] Initialization already in progress", flush=True)
                self._startup_status.show_status("Dang khoi tao overlay... vui long doi")
                self._app.processEvents()
                return

            if self._overlay is None:
                self._overlay_initializing = True
                started = time.perf_counter()
                self._startup_status.show_status("Dang nap OCR va translator...")
                self._app.processEvents()
                try:
                    try:
                        self._overlay = OverlayWindow(self._config)
                    except Exception as e:
                        print(f"[Overlay] Failed to create overlay: {e}", flush=True)
                        self._startup_status.show_status("Khoi tao loi, dang thu lai...")
                        self._app.processEvents()
                        self._tray_show_message(
                            "Screen Translator",
                            "Overlay startup failed once. Retrying with current settings.",
                            QSystemTrayIcon.MessageIcon.Warning,
                            3500,
                        )
                        try:
                            self._overlay = OverlayWindow(self._config)
                        except Exception as e2:
                            print(f"[Overlay] Retry create overlay failed: {e2}", flush=True)
                            self._tray_show_message(
                                "Screen Translator",
                                "Overlay failed to start. Check logs in %APPDATA%\\ScreenTranslator\\logs.",
                                QSystemTrayIcon.MessageIcon.Critical,
                                5000,
                            )
                            return
                    self._startup_status.show_status("Khoi tao xong, dang hien thi...")
                    self._app.processEvents()
                    self._overlay.show()
                    self._overlay.activateWindow()
                    elapsed_ms = int((time.perf_counter() - started) * 1000)
                    print(f"[HotkeyCase] toggle_overlay:show (cold_start_ms={elapsed_ms})", flush=True)
                finally:
                    self._overlay_initializing = False
                    self._startup_status.hide()
                return

            if self._overlay.isVisible():
                self._overlay.hide()
                print("[HotkeyCase] toggle_overlay:hide", flush=True)
            else:
                self._overlay.show()
                self._overlay.activateWindow()
                print("[HotkeyCase] toggle_overlay:show", flush=True)
        except Exception:
            print("[Overlay] Unexpected toggle failure", flush=True)
            traceback.print_exc()
            self._overlay = None
            if self._tray is not None:
                self._tray_show_message(
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
                print("[HotkeyCase] freeze:enabled", flush=True)
            else:
                print("[HotkeyCase] freeze:disabled", flush=True)
        else:
            print("[HotkeyCase] freeze:ignored_overlay_not_visible", flush=True)

    def _show_settings(self):
        """Show settings dialog."""
        dialog = SettingsDialog(self._config)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            old_hotkey = self._config.get("hotkey")
            self._config = dialog.get_config()

            # Always re-register to recover from stale/hijacked registrations.
            self._register_hotkey()
            if self._config.get("hotkey") != old_hotkey:
                self._setup_tray()
            else:
                self._tray.setToolTip(
                    f"Screen Translator - Ready ({self._config.get('hotkey', 'ctrl+shift+t')}) [{self._hotkey_backend}]"
                )

            # Recreate overlay with new settings if it exists
            if self._overlay is not None:
                old_overlay = self._overlay
                was_visible = old_overlay.isVisible()
                geo = old_overlay.geometry()
                try:
                    candidate_overlay = OverlayWindow(self._config)
                except Exception as e:
                    print(f"[Overlay] Failed to apply new settings: {e}", flush=True)
                    self._tray_show_message(
                        "Screen Translator",
                        "Failed to apply new settings. Keeping previous overlay.",
                        QSystemTrayIcon.MessageIcon.Warning,
                        3500,
                    )
                    self._overlay = old_overlay
                    return

                self._overlay = candidate_overlay
                self._overlay.setGeometry(geo)
                if was_visible:
                    self._overlay.show()
                    self._overlay.activateWindow()
                old_overlay.close()

    def _show_diagnostics(self):
        lines = [
            "Screen Translator Diagnostics",
            "",
            f"hotkey: {self._config.get('hotkey', 'ctrl+shift+t')}",
            f"hotkey_backend: {self._hotkey_backend}",
            f"hotkey_status: {self._hotkey_status}",
            f"source_language: {self._config.get('source_language', 'auto')}",
            f"target_language: {self._config.get('target_language', 'vi')}",
            f"translation_backend_config: {self._config.get('translation_backend', 'auto')}",
            f"translation_fallback_to_google: {bool(self._config.get('translation_fallback_to_google', True))}",
        ]

        if self._overlay is None:
            lines.append("")
            lines.append("overlay: not created yet")
        else:
            snap = self._overlay.diagnostics_snapshot()
            lines.append("")
            lines.append("overlay:")
            for key in (
                "overlay_visible",
                "safe_boot_native",
                "ocr_backend",
                "mt_backend",
                "detected_blocks",
                "translated_blocks",
            ):
                lines.append(f"  {key}: {snap.get(key, 'unknown')}")

        QMessageBox.information(
            None,
            "Diagnostics",
            "\n".join(lines),
        )

    def _on_tray_activated(self, reason):
        """Handle tray icon activation."""
        print(f"[Tray] Activated: {reason}", flush=True)
        # On Windows, a double click can emit both Trigger and DoubleClick.
        # Handle Trigger only to avoid toggling overlay twice.
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self._safe_call(self._toggle_overlay, "tray-activate")

    def _quit(self):
        """Quit the application."""
        if self._overlay:
            self._overlay.close()
        self._hotkey_native.stop()
        self._hotkey_fallback.stop()
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
