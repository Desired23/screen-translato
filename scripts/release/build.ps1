param(
    [string]$Python = "python",
    [string]$Version = "0.1.0",
    [switch]$Clean,
    [switch]$SkipToolInstall,
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptRoot "..\..")).Path
Set-Location $repoRoot

Write-Host "==> Repo root: $repoRoot"

if ($Clean) {
    Write-Host "==> Cleaning build artifacts"
    Remove-Item -Recurse -Force "$repoRoot\build" -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force "$repoRoot\dist" -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force "$repoRoot\release" -ErrorAction SilentlyContinue
}

if (-not $SkipToolInstall) {
    Write-Host "==> Ensuring PyInstaller is available"
    & $Python -c "import PyInstaller" *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "==> Installing packaging tools"
        & $Python -m pip install --disable-pip-version-check --upgrade pip pyinstaller
    }
}

Write-Host "==> Building executable with PyInstaller"
& $Python -m PyInstaller --noconfirm "$repoRoot\screen_translator.spec"

$releaseRoot = Join-Path $repoRoot "release"
$bundleDir = Join-Path $releaseRoot "ScreenTranslator-$Version"
New-Item -ItemType Directory -Path $bundleDir -Force | Out-Null

Write-Host "==> Copying dist bundle to $bundleDir"
Copy-Item -Recurse -Force "$repoRoot\dist\ScreenTranslator\*" $bundleDir

if (-not $SkipInstaller) {
    $iscc = Get-Command "iscc.exe" -ErrorAction SilentlyContinue
    $isccPath = if ($null -ne $iscc) { $iscc.Source } else { $null }
    if (-not $isccPath) {
        $candidateIsccPaths = @(
            "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
            "C:\Program Files\Inno Setup 6\ISCC.exe"
        )
        foreach ($candidate in $candidateIsccPaths) {
            if (Test-Path $candidate) {
                $isccPath = $candidate
                break
            }
        }
    }

    if ($isccPath) {
        Write-Host "==> Building installer with Inno Setup"
        & $isccPath "/DMyAppVersion=$Version" "$repoRoot\scripts\release\installer.iss"
    }
    else {
        Write-Warning "Inno Setup compiler (iscc.exe) not found. Installer step skipped."
    }
}

Write-Host "==> Done"
