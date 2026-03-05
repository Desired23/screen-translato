# config.py - Application configuration
import json
import os

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

DEFAULT_CONFIG = {
    "hotkey": "ctrl+shift+t",
    "capture_interval_ms": 1500,
    "target_language": "vi",
    "overlay_opacity": 0.85,
    "font_family": "Segoe UI",
    "font_size_min": 10,
    "font_size_max": 36,
    "background_color": "#1a1a2e",
    "text_color": "#ffffff",
    "border_color": "#e94560",
    "title_bar_color": "#16213e",
}


def load_config() -> dict:
    """Load config from file, falling back to defaults."""
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                config.update(saved)
        except (json.JSONDecodeError, IOError):
            pass
    return config


def save_config(config: dict):
    """Save config to file."""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except IOError:
        pass
