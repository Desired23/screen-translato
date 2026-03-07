"""Lightweight autorestart runner for local development (nodemon-like).

Usage:
  python dev_watch.py
  python dev_watch.py -- python main.py
  python dev_watch.py -- python bench_image_pipeline.py --image C:\\path\\img.png
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


WATCH_EXTS = {".py", ".json"}
IGNORE_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
}
POLL_INTERVAL_SEC = 0.6


def _iter_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in WATCH_EXTS:
            continue
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        yield path


def _snapshot_mtimes(root: Path) -> dict[Path, int]:
    mtimes: dict[Path, int] = {}
    for path in _iter_files(root):
        try:
            mtimes[path] = path.stat().st_mtime_ns
        except OSError:
            continue
    return mtimes


def _diff_snapshot(old: dict[Path, int], new: dict[Path, int]) -> list[Path]:
    changed: list[Path] = []
    all_paths = set(old) | set(new)
    for path in all_paths:
        if old.get(path) != new.get(path):
            changed.append(path)
    changed.sort()
    return changed


def _spawn(command: list[str], cwd: Path) -> subprocess.Popen:
    print(f"[watch] starting: {' '.join(command)}", flush=True)
    return subprocess.Popen(command, cwd=str(cwd))


def _stop(proc: subprocess.Popen | None):
    if proc is None or proc.poll() is not None:
        return
    print("[watch] stopping process...", flush=True)
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)


def _resolve_command(argv: list[str]) -> list[str]:
    if "--" in argv:
        idx = argv.index("--")
        cmd = argv[idx + 1 :]
        if not cmd:
            raise SystemExit("Missing command after `--`.")
        return cmd
    return [sys.executable, "main.py"]


def main():
    root = Path(__file__).resolve().parent
    cmd = _resolve_command(sys.argv[1:])

    last = _snapshot_mtimes(root)
    proc = _spawn(cmd, cwd=root)

    try:
        while True:
            time.sleep(POLL_INTERVAL_SEC)
            current = _snapshot_mtimes(root)
            changed = _diff_snapshot(last, current)
            last = current

            if not changed:
                continue

            preview = ", ".join(p.name for p in changed[:4])
            if len(changed) > 4:
                preview += ", ..."
            print(f"[watch] change detected ({len(changed)} files): {preview}", flush=True)
            _stop(proc)
            proc = _spawn(cmd, cwd=root)
    except KeyboardInterrupt:
        print("\n[watch] stopped by user.", flush=True)
    finally:
        _stop(proc)


if __name__ == "__main__":
    main()
