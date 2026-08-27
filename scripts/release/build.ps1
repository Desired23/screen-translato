param(
    [string]$Python = "python",
    [string]$Version = "0.1.0",
    [ValidateSet("full", "lite")]
    [string]$Flavor = "full",
    [switch]$Clean,
    [switch]$SkipToolInstall,
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptRoot "..\..")).Path
Set-Location $repoRoot

Write-Host "==> Repo root: $repoRoot"
Write-Host "==> Flavor: $Flavor"

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
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install packaging tools."
        }
    }
}

Write-Host "==> Building executable with PyInstaller"
$specPath = if ($Flavor -eq "full") {
    Join-Path $repoRoot "screen_translator_full.spec"
} else {
    Join-Path $repoRoot "screen_translator.spec"
}
& $Python -m PyInstaller --noconfirm $specPath
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed."
}

if ($Flavor -eq "full") {
    $modelSrc = Join-Path $repoRoot ".models"
    $modelDstDist = Join-Path $repoRoot "dist\ScreenTranslator\.models"
    if (Test-Path $modelSrc) {
        Write-Host "==> Copying offline models into dist bundle"
        Remove-Item -Recurse -Force $modelDstDist -ErrorAction SilentlyContinue
        New-Item -ItemType Directory -Path $modelDstDist -Force | Out-Null
        Copy-Item -Recurse -Force "$modelSrc\*" $modelDstDist
    } else {
        Write-Warning "Flavor 'full' selected but .models folder is missing. Offline NLLB translation will not be bundled."
    }
}

$releaseRoot = Join-Path $repoRoot "release"
$flavorSuffix = if ($Flavor -eq "full") { "full" } else { "lite" }
$bundleDir = Join-Path $releaseRoot "ScreenTranslator-$Version-$flavorSuffix"
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
        & $isccPath "/DMyAppVersion=$Version" "/DMyAppFlavor=$flavorSuffix" "$repoRoot\scripts\release\installer.iss"
        if ($LASTEXITCODE -ne 0) {
            throw "Inno Setup build failed."
        }
    }
    else {
        Write-Warning "Inno Setup compiler (iscc.exe) not found. Installer step skipped."
    }
}

Write-Host "==> Done"
