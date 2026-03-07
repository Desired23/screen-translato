"""
install_argos_model.py - Install Argos Translate language model(s).

Examples:
    python install_argos_model.py --from en --to vi
    python install_argos_model.py --from-langs en,ko,ja,zh --to vi --pivot en
"""

from __future__ import annotations

import argparse

try:
    import argostranslate.package
except Exception as exc:
    print("[Argos] argostranslate is not installed.")
    print("Install first: pip install argostranslate>=1.9")
    raise SystemExit(1) from exc


LANG_ALIASES = {
    "zh-cn": "zh",
    "zh-tw": "zh",
    "zh-hans": "zh",
    "zh-hant": "zh",
    "pt-br": "pt",
}


def normalize_lang(code: str) -> str:
    code = (code or "").strip().lower()
    return LANG_ALIASES.get(code, code)


def parse_lang_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    out: list[str] = []
    for item in raw.split(","):
        c = normalize_lang(item)
        if c and c not in out:
            out.append(c)
    return out


def find_pkg(available, src: str, tgt: str):
    return next((pkg for pkg in available if pkg.from_code == src and pkg.to_code == tgt), None)


def ensure_pair(
    available,
    src: str,
    tgt: str,
    installed_pairs: set[tuple[str, str]],
) -> bool:
    pair = (src, tgt)
    if src == tgt:
        return True
    if pair in installed_pairs:
        return True
    pkg = find_pkg(available, src, tgt)
    if pkg is None:
        return False
    print(f"[Argos] Downloading {src}->{tgt} ...")
    package_path = pkg.download()
    print(f"[Argos] Installing {package_path} ...")
    argostranslate.package.install_from_path(package_path)
    installed_pairs.add(pair)
    print(f"[Argos] Installed {src}->{tgt}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Install Argos translation model(s)")
    parser.add_argument("--from", dest="from_lang", default=None, help="Single source language code")
    parser.add_argument(
        "--from-langs",
        dest="from_langs",
        default="en,ko,ja,zh",
        help="Comma-separated source languages (default: en,ko,ja,zh)",
    )
    parser.add_argument("--to", dest="to_lang", default="vi", help="Target language code (default: vi)")
    parser.add_argument(
        "--pivot",
        dest="pivot_lang",
        default="en",
        help="Pivot language for fallback route (default: en)",
    )
    parser.add_argument(
        "--direct-only",
        action="store_true",
        help="Install only direct src->tgt models; do not install pivot route",
    )
    args = parser.parse_args()

    target = normalize_lang(args.to_lang)
    pivot = normalize_lang(args.pivot_lang)
    sources = parse_lang_list(args.from_langs)
    if args.from_lang:
        sources = [normalize_lang(args.from_lang)]

    if not target or not sources:
        print("[Argos] Invalid language input.")
        return 1

    print("[Argos] Updating package index...")
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()
    available_pairs = {f"{pkg.from_code}->{pkg.to_code}" for pkg in available}

    installed_pairs: set[tuple[str, str]] = set()
    missing: list[str] = []

    for src in sources:
        if src == target:
            continue

        if ensure_pair(available, src, target, installed_pairs):
            continue

        if args.direct_only:
            missing.append(f"{src}->{target}")
            continue

        ok_route = True
        if not ensure_pair(available, src, pivot, installed_pairs):
            ok_route = False
            missing.append(f"{src}->{pivot}")
        if ok_route and not ensure_pair(available, pivot, target, installed_pairs):
            ok_route = False
            missing.append(f"{pivot}->{target}")
        if ok_route:
            print(f"[Argos] Route ready for {src}->{target} via {pivot}")

    if missing:
        print("[Argos] Missing model pairs:")
        for pair in sorted(set(missing)):
            print(f"  - {pair}")
        print(f"[Argos] Available pairs in index: {len(available_pairs)}")
        return 1

    print("[Argos] Completed model installation plan successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
