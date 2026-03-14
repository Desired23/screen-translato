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
powershell -ExecutionPolicy Bypass -File .\scripts\release\build.ps1 -Version 0.1.0 -Flavor full -Clean
```

If build machine already has PyInstaller and network is restricted:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\release\build.ps1 -Version 0.1.0 -Flavor full -Clean -SkipToolInstall
```

For a smaller online-first build:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\release\build.ps1 -Version 0.1.0 -Flavor lite -Clean -SkipToolInstall
```

Output:
- Full app bundle: `release/ScreenTranslator-0.1.0-full/`
- Full installer: `release/ScreenTranslator-setup-0.1.0-full.exe`
- Lite app bundle: `release/ScreenTranslator-0.1.0-lite/`
- Lite installer: `release/ScreenTranslator-setup-0.1.0-lite.exe`

## 3) Smoke test checklist

- Launch `ScreenTranslator.exe`
- Open tray menu, open Settings, save once
- Verify config file is created at `%APPDATA%\ScreenTranslator\settings.json`
- Test hotkey capture on a sample subtitle/game text
- Verify primary OCR (rapidocr) works
- Verify fallback OCR behavior when primary fails
- Verify translation output appears in overlay
- For full flavor, disable internet and verify offline NLLB translation still works

## 4) Release notes template

- Version: `v0.1.0`
- Date: `YYYY-MM-DD`
- Added:
  - Windows app bundle build via PyInstaller
  - Windows installer via Inno Setup
  - AppData-based config path in packaged builds
  - Two packaging flavors (`full` with offline NLLB + `lite` online-first)
