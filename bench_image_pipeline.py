r"""
bench_image_pipeline.py - Benchmark OCR + translation pipeline on a single image.

Usage:
    python bench_image_pipeline.py --image C:\path\to\image.png
"""
from __future__ import annotations

import argparse
import re
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image

from config import load_config
from ocr_engine import OCREngine, merge_same_line_blocks, merge_paragraph_blocks
from translator import TranslationEngine


def _is_joined_token(text: str) -> bool:
    token = text.strip()
    if " " in token:
        return False
    letters = [c for c in token if c.isalpha()]
    if len(letters) < 6:
        return False
    upper_ratio = sum(1 for c in letters if c.isupper()) / max(len(letters), 1)
    return upper_ratio >= 0.8


def _looks_ui_like_text(text: str) -> bool:
    s = text.strip()
    if not s:
        return False
    if re.search(r"[0-9]", s) and re.search(r"[A-Za-z]", s):
        return True
    if re.search(r"[a-z][A-Z]", s):
        return True
    if any(ch in s for ch in ("-", "/", "_", ":", "%")):
        return True
    return False


def _looks_paragraph_like_frame(blocks: list) -> bool:
    if not blocks:
        return False

    punct_count = 0
    long_count = 0
    total_words = 0

    for b in blocks:
        t = getattr(b, "text", "").strip()
        if not t:
            continue
        words = len(t.split())
        total_words += words
        if words >= 6:
            long_count += 1
        if words >= 4 and re.search(r"[.,;:!?)]$", t):
            punct_count += 1

    n = max(len(blocks), 1)
    avg_words = total_words / n
    long_ratio = long_count / n
    punct_ratio = punct_count / n

    return (
        avg_words >= 4.6
        or long_ratio >= 0.30
        or (long_ratio >= 0.20 and punct_ratio >= 0.12)
    )


def _is_bullet_like(text: str) -> bool:
    s = text.strip()
    if not s:
        return False
    if s[0] in {"•", "-", "*", "–"}:
        return True
    return bool(re.match(r"^\d+[\.)]\s+", s))


def _looks_like_document_frame(blocks: list, cfg: dict) -> bool:
    if not blocks:
        return False
    if len(blocks) < int(cfg.get("auto_document_line_mode_min_blocks", 4)):
        return False

    total_words = 0
    long_count = 0
    sentence_like = 0
    bullet_like = 0
    tech_like = 0
    joined_like = 0
    alpha_lines = 0

    for b in blocks:
        t = getattr(b, "text", "").strip()
        if not t:
            continue
        words = len(t.split())
        total_words += words
        if words >= 6:
            long_count += 1
        if words >= 4 and re.search(r"[.,;:!?)]$", t):
            sentence_like += 1
        if _is_bullet_like(t):
            bullet_like += 1
        if _looks_ui_like_text(t):
            tech_like += 1
        if _is_joined_token(t):
            joined_like += 1
        if re.search(r"[A-Za-z]", t):
            alpha_lines += 1

    n = max(len(blocks), 1)
    if alpha_lines / n < 0.5:
        return False

    avg_words = total_words / n
    long_ratio = long_count / n
    sentence_ratio = sentence_like / n
    bullet_ratio = bullet_like / n
    tech_ratio = tech_like / n
    joined_ratio = joined_like / n

    if tech_ratio >= 0.40 and joined_ratio >= 0.20:
        return False

    return (
        avg_words >= float(cfg.get("auto_document_line_mode_avg_words", 4.8))
        or long_ratio >= float(cfg.get("auto_document_line_mode_long_ratio", 0.28))
        or sentence_ratio >= float(cfg.get("auto_document_line_mode_sentence_ratio", 0.14))
        or bullet_ratio >= float(cfg.get("auto_document_line_mode_bullet_ratio", 0.08))
    )


def _looks_like_ui_game_frame(blocks: list, cfg: dict) -> bool:
    if not blocks:
        return False
    if len(blocks) < int(cfg.get("auto_ui_line_mode_min_blocks", 12)):
        return False
    if _looks_paragraph_like_frame(blocks):
        return False

    short_count = 0
    tech_count = 0
    for b in blocks:
        t = getattr(b, "text", "").strip()
        if not t:
            continue
        if len(t.split()) <= 4:
            short_count += 1
        if _looks_ui_like_text(t):
            tech_count += 1

    n = max(len(blocks), 1)
    short_ratio = short_count / n
    tech_ratio = tech_count / n
    joined_ratio = sum(1 for b in blocks if _is_joined_token(getattr(b, "text", ""))) / n

    return (
        short_ratio >= float(cfg.get("auto_ui_line_mode_short_ratio", 0.55))
        and (
            tech_ratio >= float(cfg.get("auto_ui_line_mode_tech_ratio", 0.28))
            or (
                tech_ratio >= 0.12
                and joined_ratio >= float(cfg.get("auto_ui_line_mode_joined_ratio", 0.22))
            )
        )
    )


def _load_image(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def _run_once(
    image: np.ndarray,
    ocr: OCREngine,
    translator: TranslationEngine,
    clear_cache_before: bool,
    paragraph_merge_enabled: bool,
    paragraph_max_lines: int,
    paragraph_min_overlap: float,
    paragraph_center_factor: float,
    paragraph_width_expand: float,
    auto_ui_line_mode_enabled: bool,
    auto_document_line_mode_enabled: bool,
    config: dict,
) -> dict:
    if clear_cache_before:
        translator.clear_cache()

    t0 = time.perf_counter()

    t_ocr0 = time.perf_counter()
    blocks = ocr.detect(image)
    t_ocr = (time.perf_counter() - t_ocr0) * 1000

    t_merge0 = time.perf_counter()
    line_blocks = merge_same_line_blocks(blocks)
    merged = line_blocks
    mode = "line"
    if paragraph_merge_enabled:
        if auto_document_line_mode_enabled and _looks_like_document_frame(line_blocks, config):
            if bool(config.get("auto_document_translate_by_paragraph", True)):
                doc_merged = merge_paragraph_blocks(
                    line_blocks,
                    max_lines_per_group=int(config.get("document_paragraph_max_lines_per_group", 8)),
                    min_overlap_ratio=float(config.get("document_paragraph_min_overlap_ratio", 0.30)),
                    center_distance_factor=float(config.get("document_paragraph_center_distance_factor", 0.85)),
                    max_width_expand_ratio=float(config.get("document_paragraph_max_width_expand_ratio", 3.00)),
                )
                if doc_merged:
                    collapse_ratio = len(doc_merged) / max(len(line_blocks), 1)
                    if len(line_blocks) >= 10 and collapse_ratio < 0.06:
                        merged = line_blocks
                        mode = "document_line_fallback"
                    else:
                        merged = doc_merged
                        mode = "document_paragraph"
                else:
                    merged = line_blocks
                    mode = "document_line"
            else:
                merged = line_blocks
                mode = "document_line"
        elif auto_ui_line_mode_enabled and _looks_like_ui_game_frame(line_blocks, config):
            if bool(config.get("ui_panel_merge_enabled", True)):
                ui_merged = merge_paragraph_blocks(
                    line_blocks,
                    max_lines_per_group=int(config.get("ui_panel_merge_max_lines_per_group", 3)),
                    min_overlap_ratio=float(config.get("ui_panel_merge_min_overlap_ratio", 0.45)),
                    center_distance_factor=float(config.get("ui_panel_merge_center_distance_factor", 0.40)),
                    max_width_expand_ratio=float(config.get("ui_panel_merge_max_width_expand_ratio", 1.45)),
                )
                if ui_merged and len(ui_merged) > max(2, len(line_blocks) // 3):
                    merged = ui_merged
                    mode = "ui_panel_merge"
                else:
                    merged = line_blocks
                    mode = "ui_line"
            else:
                merged = line_blocks
                mode = "ui_line"
        else:
            merged_candidate = merge_paragraph_blocks(
                line_blocks,
                max_lines_per_group=paragraph_max_lines,
                min_overlap_ratio=paragraph_min_overlap,
                center_distance_factor=paragraph_center_factor,
                max_width_expand_ratio=paragraph_width_expand,
            )
            collapse_threshold = max(2, len(line_blocks) // 4)
            if len(line_blocks) >= 8 and len(merged_candidate) <= collapse_threshold:
                merged = line_blocks
                mode = "line_fallback"
            else:
                merged = merged_candidate
                mode = "paragraph"
    t_merge = (time.perf_counter() - t_merge0) * 1000

    texts = [b.text for b in merged]

    t_tr0 = time.perf_counter()
    translated = translator.translate_batch(texts)
    t_tr = (time.perf_counter() - t_tr0) * 1000

    t_total = (time.perf_counter() - t0) * 1000
    return {
        "ocr_ms": t_ocr,
        "merge_ms": t_merge,
        "translate_ms": t_tr,
        "total_ms": t_total,
        "raw_blocks": len(blocks),
        "merged_blocks": len(merged),
        "translated_count": len(translated),
        "unique_texts": len(set(texts)),
        "merge_mode": mode,
    }


def _fmt(ms: float) -> str:
    return f"{ms:.0f}ms"


def main():
    parser = argparse.ArgumentParser(description="Benchmark OCR + translation on one image.")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--runs", type=int, default=3, help="Number of warm runs (default: 3)")
    parser.add_argument("--source-lang", default=None, help="OCR source language override")
    parser.add_argument("--target-lang", default=None, help="Translation target language override")
    parser.add_argument(
        "--translation-backend",
        default=None,
        choices=["auto", "google", "argos", "nllb"],
        help="Translation backend override",
    )
    parser.add_argument(
        "--no-google-fallback",
        action="store_true",
        help="Disable runtime fallback to Google when backend is argos",
    )
    parser.add_argument(
        "--argos-pivot-language",
        default=None,
        help="Pivot language for Argos multi-hop routing (default from config)",
    )
    parser.add_argument(
        "--nllb-model-dir",
        default=None,
        help="Path to local CTranslate2 NLLB model directory",
    )
    parser.add_argument(
        "--nllb-tokenizer-path",
        default=None,
        help="Path to tokenizer files for NLLB (default: model dir)",
    )
    parser.add_argument(
        "--nllb-device",
        default=None,
        help="NLLB device (cpu/cuda)",
    )
    parser.add_argument(
        "--nllb-compute-type",
        default=None,
        help="NLLB compute type (int8, int8_float16, float32, ...)",
    )
    parser.add_argument(
        "--nllb-beam-size",
        type=int,
        default=None,
        help="NLLB beam size",
    )
    parser.add_argument(
        "--nllb-max-decoding-length",
        type=int,
        default=None,
        help="NLLB max decoding length",
    )
    parser.add_argument(
        "--disable-context-refine",
        action="store_true",
        help="Disable local-context refinement for short bubbles",
    )
    parser.add_argument(
        "--context-max-chars",
        type=int,
        default=None,
        help="Override context refine max chars",
    )
    parser.add_argument(
        "--context-max-words",
        type=int,
        default=None,
        help="Override context refine max words",
    )
    parser.add_argument(
        "--context-max-per-batch",
        type=int,
        default=None,
        help="Override max number of blocks refined with context per frame",
    )
    parser.add_argument("--primary-backend", default=None, help="OCR primary backend override")
    parser.add_argument("--fallback-backend", default=None, help="OCR fallback backend override")
    parser.add_argument("--winrt-enabled", action="store_true", help="Enable WinRT backend selection")
    parser.add_argument("--easyocr-enabled", action="store_true", help="Enable EasyOCR backend selection")
    parser.add_argument("--disable-winrt", action="store_true", help="Force disable WinRT backend")
    parser.add_argument("--disable-easyocr", action="store_true", help="Force disable EasyOCR backend")
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip OCR warm-up step",
    )
    args = parser.parse_args()

    img_path = Path(args.image).expanduser()
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    config = load_config()
    source_lang = args.source_lang or config.get("source_language", "en")
    target_lang = args.target_lang or config.get("target_language", "vi")
    translation_backend = args.translation_backend or config.get("translation_backend", "auto")
    translation_fallback_to_google = bool(
        config.get("translation_fallback_to_google", True)
    ) and not bool(args.no_google_fallback)
    argos_pivot_language = args.argos_pivot_language or config.get("argos_pivot_language", "en")
    nllb_model_dir = args.nllb_model_dir or config.get("nllb_model_dir", ".models/nllb-ct2-int8")
    nllb_tokenizer_path = args.nllb_tokenizer_path or config.get(
        "nllb_tokenizer_path",
        ".models/nllb-ct2-int8",
    )
    nllb_device = args.nllb_device or config.get("nllb_device", "cpu")
    nllb_compute_type = args.nllb_compute_type or config.get("nllb_compute_type", "int8")
    nllb_beam_size = int(
        args.nllb_beam_size if args.nllb_beam_size is not None else config.get("nllb_beam_size", 2)
    )
    nllb_max_decoding_length = int(
        args.nllb_max_decoding_length
        if args.nllb_max_decoding_length is not None
        else config.get("nllb_max_decoding_length", 192)
    )
    context_refine_enabled = bool(config.get("translation_context_refine_enabled", True)) and (
        not bool(args.disable_context_refine)
    )
    context_refine_max_chars = int(
        args.context_max_chars
        if args.context_max_chars is not None
        else config.get("translation_context_refine_max_chars", 48)
    )
    context_refine_max_words = int(
        args.context_max_words
        if args.context_max_words is not None
        else config.get("translation_context_refine_max_words", 10)
    )
    context_refine_max_per_batch = int(
        args.context_max_per_batch
        if args.context_max_per_batch is not None
        else config.get("translation_context_refine_max_per_batch", 1)
    )

    if args.primary_backend:
        config["primary_backend"] = args.primary_backend
    if args.fallback_backend:
        config["fallback_backend"] = args.fallback_backend
    if args.winrt_enabled:
        config["winrt_enabled"] = True
    if args.easyocr_enabled:
        config["easyocr_enabled"] = True
    if args.disable_winrt:
        config["winrt_enabled"] = False
    if args.disable_easyocr:
        config["easyocr_enabled"] = False

    image = _load_image(img_path)
    print("=" * 70)
    print(f"Image        : {img_path}")
    print(f"Resolution   : {image.shape[1]}x{image.shape[0]}")
    print(f"Source/Target: {source_lang} -> {target_lang}")
    print(
        f"Backends     : primary={config.get('primary_backend')} "
        f"fallback={config.get('fallback_backend')} "
        f"winrt={config.get('winrt_enabled')} easyocr={config.get('easyocr_enabled')}"
    )
    print(
        "Translation  : "
        f"backend={translation_backend} "
        f"fallback_to_google={translation_fallback_to_google} "
        f"argos_pivot={argos_pivot_language}"
    )
    print(
        "NLLB         : "
        f"model_dir={nllb_model_dir} "
        f"tokenizer={nllb_tokenizer_path} "
        f"device={nllb_device} "
        f"compute={nllb_compute_type} "
        f"beam={nllb_beam_size}"
    )
    print(
        "ContextRefine: "
        f"enabled={context_refine_enabled} "
        f"max_chars={context_refine_max_chars} "
        f"max_words={context_refine_max_words} "
        f"max_per_batch={context_refine_max_per_batch}"
    )
    print(
        "Translation+ : "
        f"auto_source_route={bool(config.get('translation_auto_source_routing_enabled', True))} "
        f"semantic_cache={bool(config.get('translation_semantic_cache_enabled', True))}"
    )
    print(f"Paragraph merge: {bool(config.get('paragraph_merge_enabled', True))}")
    print(
        "Paragraph cfg : "
        f"max_lines={int(config.get('paragraph_max_lines_per_group', 4))} "
        f"min_overlap={float(config.get('paragraph_min_overlap_ratio', 0.35))} "
        f"center_factor={float(config.get('paragraph_center_distance_factor', 0.55))} "
        f"width_expand={float(config.get('paragraph_max_width_expand_ratio', 1.8))}"
    )
    print(
        "UI/Game auto line-level: "
        f"{bool(config.get('auto_ui_line_mode_enabled', True))}"
    )
    print(
        "Document auto line-level: "
        f"{bool(config.get('auto_document_line_mode_enabled', True))}"
    )
    print("=" * 70)

    ocr = OCREngine(source_language=source_lang, config=config)
    translator = TranslationEngine(
        target_language=target_lang,
        source_language=source_lang,
        translation_backend=translation_backend,
        fallback_to_google=translation_fallback_to_google,
        argos_pivot_language=argos_pivot_language,
        nllb_model_dir=nllb_model_dir,
        nllb_tokenizer_path=nllb_tokenizer_path,
        nllb_device=nllb_device,
        nllb_compute_type=nllb_compute_type,
        nllb_beam_size=nllb_beam_size,
        nllb_max_decoding_length=nllb_max_decoding_length,
        context_refine_enabled=context_refine_enabled,
        context_refine_max_chars=context_refine_max_chars,
        context_refine_max_words=context_refine_max_words,
        context_refine_max_per_batch=context_refine_max_per_batch,
        manga_mode=bool(config.get("manga_translation_mode", True)),
        manga_source_fixes_enabled=bool(config.get("manga_source_fixes_enabled", True)),
        manga_target_post_edit_enabled=bool(config.get("manga_target_post_edit_enabled", True)),
        auto_document_guard=bool(config.get("translation_auto_document_guard", True)),
        game_term_guard_enabled=bool(config.get("translation_game_term_guard_enabled", True)),
        game_post_edit_enabled=bool(config.get("translation_game_post_edit_enabled", True)),
        auto_source_routing_enabled=bool(
            config.get("translation_auto_source_routing_enabled", True)
        ),
        semantic_cache_enabled=bool(config.get("translation_semantic_cache_enabled", True)),
        custom_glossary_enabled=bool(config.get("custom_glossary_enabled", True)),
        custom_glossary=config.get("custom_glossary", {}),
    )

    if not args.skip_warmup:
        t0 = time.perf_counter()
        ocr.warm_up()
        print(f"OCR warm-up  : {_fmt((time.perf_counter() - t0) * 1000)}")

    print("\n[Cold run] (translation cache cleared)")
    paragraph_merge_enabled = bool(config.get("paragraph_merge_enabled", True))
    paragraph_max_lines = int(config.get("paragraph_max_lines_per_group", 4))
    paragraph_min_overlap = float(config.get("paragraph_min_overlap_ratio", 0.35))
    paragraph_center_factor = float(config.get("paragraph_center_distance_factor", 0.55))
    paragraph_width_expand = float(config.get("paragraph_max_width_expand_ratio", 1.8))
    auto_ui_line_mode_enabled = bool(config.get("auto_ui_line_mode_enabled", True))
    auto_document_line_mode_enabled = bool(config.get("auto_document_line_mode_enabled", True))
    cold = _run_once(
        image,
        ocr,
        translator,
        clear_cache_before=True,
        paragraph_merge_enabled=paragraph_merge_enabled,
        paragraph_max_lines=paragraph_max_lines,
        paragraph_min_overlap=paragraph_min_overlap,
        paragraph_center_factor=paragraph_center_factor,
        paragraph_width_expand=paragraph_width_expand,
        auto_ui_line_mode_enabled=auto_ui_line_mode_enabled,
        auto_document_line_mode_enabled=auto_document_line_mode_enabled,
        config=config,
    )
    print(
        f"ocr={_fmt(cold['ocr_ms'])}  merge={_fmt(cold['merge_ms'])}  "
        f"translate={_fmt(cold['translate_ms'])}  total={_fmt(cold['total_ms'])}  "
        f"blocks={cold['raw_blocks']}->{cold['merged_blocks']}  unique={cold['unique_texts']}  "
        f"mode={cold['merge_mode']}"
    )

    runs = max(1, int(args.runs))
    warm_results = []
    print(f"\n[Warm runs x{runs}]")
    for i in range(runs):
        r = _run_once(
            image,
            ocr,
            translator,
            clear_cache_before=False,
            paragraph_merge_enabled=paragraph_merge_enabled,
            paragraph_max_lines=paragraph_max_lines,
            paragraph_min_overlap=paragraph_min_overlap,
            paragraph_center_factor=paragraph_center_factor,
            paragraph_width_expand=paragraph_width_expand,
            auto_ui_line_mode_enabled=auto_ui_line_mode_enabled,
            auto_document_line_mode_enabled=auto_document_line_mode_enabled,
            config=config,
        )
        warm_results.append(r)
        print(
            f"run{i+1}: ocr={_fmt(r['ocr_ms'])}  merge={_fmt(r['merge_ms'])}  "
            f"translate={_fmt(r['translate_ms'])}  total={_fmt(r['total_ms'])}  "
            f"mode={r['merge_mode']}"
        )

    print("\n[Summary]")
    avg_total = statistics.mean(r["total_ms"] for r in warm_results)
    avg_ocr = statistics.mean(r["ocr_ms"] for r in warm_results)
    avg_merge = statistics.mean(r["merge_ms"] for r in warm_results)
    avg_translate = statistics.mean(r["translate_ms"] for r in warm_results)
    print(f"warm_avg_total     : {_fmt(avg_total)}")
    print(f"warm_avg_ocr       : {_fmt(avg_ocr)}")
    print(f"warm_avg_merge     : {_fmt(avg_merge)}")
    print(f"warm_avg_translate : {_fmt(avg_translate)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
