param(
    [string]$ExePath = "release\ScreenTranslator-0.1.0-full\ScreenTranslator.exe",
    [switch]$Wait
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ExePath)) {
    throw "EXE_NOT_FOUND: $ExePath"
}

$exeFull = (Resolve-Path $ExePath).Path
$exeDir = Split-Path -Parent $exeFull
$internalDir = Join-Path $exeDir "_internal"
$torchDir = Join-Path $internalDir "torch\lib"
$ct2Dir = Join-Path $internalDir "ctranslate2"

$parts = @()
if ($env:PATH) {
    $parts = $env:PATH.Split(';') | Where-Object { $_ -and $_.Trim().Length -gt 0 }
}

$filtered = New-Object System.Collections.Generic.List[string]
foreach ($p in $parts) {
    $normalized = $p.ToLowerInvariant()
    if ($normalized -like "*\program files\screen translator\*") { continue }
    if ($normalized -like "*\torch\lib*") { continue }
    $filtered.Add($p)
}

if (Test-Path $ct2Dir) { $filtered.Insert(0, $ct2Dir) }
if (Test-Path $torchDir) { $filtered.Insert(0, $torchDir) }

$env:PATH = [string]::Join(';', $filtered)
$env:USE_TORCH = "0"
$env:TRANSFORMERS_NO_TORCH = "1"
$env:TRANSFORMERS_NO_TF = "1"
$env:TRANSFORMERS_NO_FLAX = "1"

Write-Host "Launching:" $exeFull
Write-Host "PATH prefixed with:" $torchDir "," $ct2Dir

if ($Wait) {
    Start-Process -FilePath $exeFull -Wait
} else {
    Start-Process -FilePath $exeFull | Out-Null
}
