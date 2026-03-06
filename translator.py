# translator.py - Translation engine with caching
from deep_translator import GoogleTranslator
from functools import lru_cache
import threading


class TranslationEngine:
    """Google Translate wrapper with caching and batch support."""

    def __init__(self, target_language: str = "vi", source_language: str = "auto"):
        self._target_lang = target_language
        self._source_lang = "auto"  # always use auto-detect regardless of source_language param
        self._translator = GoogleTranslator(source="auto", target=target_language)
        self._cache: dict[str, str] = {}
        self._lock = threading.Lock()

    @property
    def target_language(self) -> str:
        return self._target_lang

    @target_language.setter
    def target_language(self, lang: str):
        if lang != self._target_lang:
            self._target_lang = lang
            self._translator = GoogleTranslator(source="auto", target=lang)
            self._cache.clear()

    def translate(self, text: str) -> str:
        """Translate a single text string with caching."""
        if not text or not text.strip():
            return text

        text = text.strip()

        with self._lock:
            if text in self._cache:
                return self._cache[text]

        try:
            result = self._translator.translate(text)
            if result:
                with self._lock:
                    self._cache[text] = result
                return result
        except Exception:
            pass

        return text

    def translate_batch(self, texts: list[str]) -> list[str]:
        """
        Translate multiple texts, using cache for already-seen texts.
        Batches uncached texts into a single API call when possible.
        """
        results = []
        uncached_indices = []
        uncached_texts = []

        # Check cache first
        with self._lock:
            for i, text in enumerate(texts):
                stripped = text.strip()
                if not stripped:
                    results.append(text)
                elif stripped in self._cache:
                    results.append(self._cache[stripped])
                else:
                    results.append(None)  # placeholder
                    uncached_indices.append(i)
                    uncached_texts.append(stripped)

        # Translate uncached texts
        if uncached_texts:
            try:
                # Join with separator for batch translation
                separator = " ||| "
                combined = separator.join(uncached_texts)

                if len(combined) <= 4500:
                    # Single batch call
                    translated_combined = self._translator.translate(combined)
                    if translated_combined:
                        translated_parts = translated_combined.split("|||")
                        translated_parts = [p.strip() for p in translated_parts]

                        # If splitting produced wrong count, fall back to individual
                        if len(translated_parts) == len(uncached_texts):
                            with self._lock:
                                for idx, orig, trans in zip(uncached_indices, uncached_texts, translated_parts):
                                    self._cache[orig] = trans
                                    results[idx] = trans
                        else:
                            self._translate_individually(uncached_texts, uncached_indices, results)
                    else:
                        self._translate_individually(uncached_texts, uncached_indices, results)
                else:
                    # Too long for single call, translate individually
                    self._translate_individually(uncached_texts, uncached_indices, results)
            except Exception:
                self._translate_individually(uncached_texts, uncached_indices, results)

        # Fill any remaining None placeholders
        for i in range(len(results)):
            if results[i] is None:
                results[i] = texts[i]

        return results

    def _translate_individually(self, texts: list[str], indices: list[int], results: list):
        """Fallback: translate texts one by one."""
        for idx, text in zip(indices, texts):
            try:
                translated = self._translator.translate(text)
                if translated:
                    with self._lock:
                        self._cache[text] = translated
                    results[idx] = translated
                else:
                    results[idx] = text
            except Exception:
                results[idx] = text

    def clear_cache(self):
        """Clear the translation cache."""
        with self._lock:
            self._cache.clear()
