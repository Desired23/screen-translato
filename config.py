# config.py - Application configuration
import json
import os
import sys
from pathlib import Path


APP_NAME = "ScreenTranslator"


def _get_config_file() -> Path:
    """Resolve settings.json location.

    - Development mode: keep settings next to source files for convenience.
    - Frozen/release mode: store settings under %APPDATA%\\ScreenTranslator.
    """
    if getattr(sys, "frozen", False):
        appdata = os.getenv("APPDATA") or str(Path.home())
        config_dir = Path(appdata) / APP_NAME
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "settings.json"
    return Path(__file__).resolve().parent / "settings.json"


CONFIG_FILE = _get_config_file()

DEFAULT_CONFIG = {
    "hotkey": "ctrl+shift+t",
    "capture_interval_ms": 1500,
    "drop_frames_when_busy": True,
    # if frame looks like UI/game subtitles, keep line-level blocks (avoid bubble merge)
    "auto_ui_line_mode_enabled": True,
    "auto_ui_line_mode_min_blocks": 12,
    "auto_ui_line_mode_short_ratio": 0.55,
    "auto_ui_line_mode_tech_ratio": 0.28,
    "auto_ui_line_mode_joined_ratio": 0.22,
    # UI/game panel-aware merge tuning
    "ui_panel_merge_enabled": True,
    "ui_panel_merge_max_lines_per_group": 2,
    "ui_panel_merge_min_overlap_ratio": 0.45,
    "ui_panel_merge_center_distance_factor": 0.40,
    "ui_panel_merge_max_width_expand_ratio": 1.45,
    # dark UI OCR retry (yellow/bright text on dark background)
    "ocr_dark_ui_retry_enabled": True,
    "ocr_dark_ui_retry_confidence_thresh": 0.84,
    "ocr_dark_ui_retry_joined_ratio_thresh": 0.26,
    "ocr_dark_ui_retry_score_margin": 0.02,
    # auto document mode: keep line-level blocks for long article/web text
    "auto_document_line_mode_enabled": True,
    "auto_document_line_mode_min_blocks": 4,
    "auto_document_line_mode_avg_words": 4.8,
    "auto_document_line_mode_long_ratio": 0.28,
    "auto_document_line_mode_sentence_ratio": 0.14,
    "auto_document_line_mode_bullet_ratio": 0.08,
    # when document mode is detected, merge lines into paragraph blocks before translation
    "auto_document_translate_by_paragraph": True,
    "document_paragraph_max_lines_per_group": 8,
    "document_paragraph_min_overlap_ratio": 0.30,
    "document_paragraph_center_distance_factor": 0.85,
    "document_paragraph_max_width_expand_ratio": 3.00,
    "ocr_smart_max_boxes": 80,
    "source_language": "auto",
    "target_language": "vi",
    "overlay_opacity": 0.85,
    "font_family": "Segoe UI",
    "font_size_min": 10,
    "font_size_max": 36,
    "background_color": "#ffffff",
    "text_color": "#000000",
    "border_color": "#ffffff",
    "title_bar_color": "#1a1a2e",
    # OCR backend strategy
    "primary_backend": "rapidocr",
    "fallback_backend": "none",
    "confidence_thresh": 0.75,
    "winrt_enabled": False,
    # If WinRT falls back to another language pack (e.g. ja -> en),
    # keep it as a last-resort OCR path instead of disabling all OCR.
    "allow_winrt_mismatch_fallback": True,
    "easyocr_enabled": False,
    # Optional bundled PaddleOCR models for CJK-heavy fallback OCR in release zip.
    # When present, app will load models from this local bundle instead of
    # relying on user-level PaddleX caches under %USERPROFILE%.
    "paddleocr_enabled": True,
    "paddle_model_root": ".models/paddleocr",
    "paddle_text_detection_model_dir": "",
    "paddle_text_recognition_model_dir": "",
    "paddle_textline_orientation_model_dir": "",
    "paddle_device": "cpu",
    "paddle_enable_mkldnn": False,
    "paddle_cpu_threads": 2,
    # limit expensive Paddle rescue to a few suspicious lines instead of whole frames
    "paddle_region_rescue_max_regions": 2,
    "paddle_region_rescue_max_lines_per_region": 2,
    # optional slower second-pass OCR enhancements for noisy manga pages
    "rapid_quality_retry": False,
    # static-frame dedup tuning (higher thresholds = less re-scan noise)
    "dirty_mad_threshold": 4.5,
    "dirty_changed_ratio_threshold": 0.02,
    # visual-novel / dating-sim style dialog layout heuristics
    "auto_dialog_ui_mode_enabled": True,
    "auto_dialog_ui_mode_min_blocks": 4,
    "auto_dialog_ui_mode_max_blocks": 10,
    "auto_dialog_ui_bottom_ratio": 0.72,
    "auto_dialog_ui_choice_ratio": 0.86,
    "dialog_ui_hud_max_y_ratio": 0.68,
    "dialog_ui_hud_max_width_ratio": 0.42,
    "dialog_ui_hud_max_height_ratio": 0.16,
    "dialog_ui_hud_max_chars": 18,
    "dialog_ui_speaker_min_y_ratio": 0.62,
    "dialog_ui_speaker_max_width_ratio": 0.22,
    "dialog_ui_speaker_max_chars": 10,
    "dialog_ui_max_lines_per_group": 3,
    "dialog_ui_min_overlap_ratio": 0.52,
    "dialog_ui_center_distance_factor": 0.40,
    "dialog_ui_max_width_expand_ratio": 1.28,
    "log_game_layout_debug": True,
    # merge nearby OCR lines into paragraph blocks before translation
    "paragraph_merge_enabled": True,
    # conservative bubble-aware paragraph merge (to avoid cross-bubble merging)
    "paragraph_max_lines_per_group": 6,
    "paragraph_min_overlap_ratio": 0.25,
    "paragraph_center_distance_factor": 0.65,
    "paragraph_max_width_expand_ratio": 2.20,
    # translation backend: auto (prefer local NLLB, then Google), nllb, google
    "translation_backend": "auto",
    # if nllb fails/unavailable, fallback to Google automatically
    "translation_fallback_to_google": True,
    # local NLLB (CTranslate2) backend options
    "nllb_model_dir": ".models/nllb-ct2-int8",
    "nllb_tokenizer_path": ".models/nllb-ct2-int8",
    "nllb_device": "cpu",
    "nllb_compute_type": "int8",
    "nllb_beam_size": 2,
    "nllb_max_decoding_length": 192,
    # clip excessively long OCR paragraphs to keep local NLLB responsive
    "nllb_max_source_tokens": 384,
    # refine short dialogue blocks using left/right neighbors without merging blocks
    "translation_context_refine_enabled": True,
    "translation_context_refine_max_chars": 48,
    "translation_context_refine_max_words": 10,
    "translation_context_refine_max_per_batch": 1,
    # manga translation quality options
    "manga_translation_mode": True,
    "manga_source_fixes_enabled": True,
    "manga_target_post_edit_enabled": True,
    # disable manga phrase-rewrite rules on long document-like text
    "translation_auto_document_guard": True,
    # preserve key game terms + small post-edits for game skill text
    "translation_game_term_guard_enabled": True,
    "translation_game_post_edit_enabled": True,
    # auto-route source language per OCR script for local backend (nllb)
    "translation_auto_source_routing_enabled": True,
    # language detector: script heuristics first, then langid for ambiguous Latin OCR.
    "language_detector_backend": "langid",
    "language_detector_min_confidence": 0.55,
    # reuse translations for near-identical OCR lines to reduce jitter cost
    "translation_semantic_cache_enabled": True,
    "custom_glossary_enabled": True,
    # packaged safety mode: avoid native DLL crashes during first overlay boot
    "safe_boot_native": True,
    # key: source phrase (EN), value: preferred translation (VI)
    "custom_glossary": {},
}


def load_config() -> dict:
    """Load config from file, falling back to defaults."""
    config = DEFAULT_CONFIG.copy()
    if CONFIG_FILE.exists():
        try:
            # utf-8-sig also accepts plain utf-8 and tolerates BOM-written files.
            with CONFIG_FILE.open("r", encoding="utf-8-sig") as f:
                saved = json.load(f)
                config.update(saved)
        except (json.JSONDecodeError, IOError):
            pass
    return config


def save_config(config: dict):
    """Save config to file."""
    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with CONFIG_FILE.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except IOError:
        pass
