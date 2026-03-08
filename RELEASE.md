# Screen Translator Release Guide (Windows)

## 1) Prepare environment

- Python 3.10+ installed
- Inno Setup 6 installed (optional, only if you need `.exe` installer)
- Project dependencies installed:

```powershell
python -m pip install -r requirements.txt
```

## 2) Build app bundle

Run from repo root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\release\build.ps1 -Version 0.1.0 -Clean
```

If build machine already has PyInstaller and network is restricted:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\release\build.ps1 -Version 0.1.0 -Clean -SkipToolInstall
```

Output:
- App bundle: `release/ScreenTranslator-0.1.0/`
- Installer (if `iscc.exe` is installed): `release/ScreenTranslator-setup-0.1.0.exe`

## 3) Smoke test checklist

- Launch `ScreenTranslator.exe`
- Open tray menu, open Settings, save once
- Verify config file is created at `%APPDATA%\ScreenTranslator\settings.json`
- Test hotkey capture on a sample subtitle/game text
- Verify primary OCR (rapidocr) works
- Verify fallback OCR behavior when primary fails
- Verify translation output appears in overlay

## 4) Release notes template

- Version: `v0.1.0`
- Date: `YYYY-MM-DD`
- Added:
  - Windows app bundle build via PyInstaller
  - Windows installer via Inno Setup
  - AppData-based config path in packaged builds
- Known limitations:
  - Optional heavy backends (EasyOCR/NLLB/Argos) are excluded from default installer

## 5) Optional heavy backend strategy

Keep default installer light. Offer separate scripts/docs for users who need offline/large models:
- `install_argos_model.py`
- `install_nllb_ct2.py`
