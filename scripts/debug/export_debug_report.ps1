param(
    [string]$ExePath = "release\ScreenTranslator-0.1.0-full\ScreenTranslator.exe",
    [int]$WaitSeconds = 10,
    [switch]$PressHotkey,
    [switch]$CollectOnly
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptRoot "..\..")).Path
Set-Location $repoRoot

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$debugDir = Join-Path $repoRoot "release\debug"
New-Item -ItemType Directory -Path $debugDir -Force | Out-Null

$logDir = Join-Path $env:APPDATA "ScreenTranslator\logs"
$latestLogBefore = $null
if (Test-Path $logDir) {
    $latestLogBefore = Get-ChildItem $logDir -File | Sort-Object LastWriteTime -Descending | Select-Object -First 1
}

$status = "COLLECT_ONLY"
$exitCode = ""
$pidValue = ""

if (-not $CollectOnly) {
    if (-not (Test-Path $ExePath)) {
        throw "EXE_NOT_FOUND: $ExePath"
    }

    $proc = Start-Process -FilePath $ExePath -PassThru
    $pidValue = "$($proc.Id)"
    Start-Sleep -Seconds $WaitSeconds

    if ($PressHotkey) {
        python press_hotkey.py
        Start-Sleep -Seconds 4
    }

    $status = "RUNNING"
    if ($proc.HasExited) {
        $status = "EXITED"
        $exitCode = "$($proc.ExitCode)"
    } else {
        Stop-Process -Id $proc.Id -Force
        $status = "RUNNING_THEN_STOPPED"
    }

    Start-Sleep -Seconds 1
}

$latestLogAfter = $null
if (Test-Path $logDir) {
    $latestLogAfter = Get-ChildItem $logDir -File | Sort-Object LastWriteTime -Descending | Select-Object -First 1
}

$logPath = if ($latestLogAfter) { $latestLogAfter.FullName } else { "" }
$logText = if ($latestLogAfter) { Get-Content $latestLogAfter.FullName -Raw } else { "No app log found." }

$txtOut = Join-Path $debugDir "debug-$timestamp.txt"
$mdOut = Join-Path $debugDir "debug-$timestamp.md"

@(
    "timestamp=$timestamp"
    "exe=$ExePath"
    "status=$status"
    "exit_code=$exitCode"
    "pid=$pidValue"
    "wait_seconds=$WaitSeconds"
    "hotkey_sent=$($PressHotkey.IsPresent)"
    "log_path=$logPath"
    "latest_log_before=$($latestLogBefore.FullName)"
    "latest_log_after=$($latestLogAfter.FullName)"
    ""
    "===== APP LOG ====="
    $logText
) | Set-Content -Path $txtOut -Encoding UTF8

$mdLines = @(
    "# ScreenTranslator Debug Report"
    ""
    "- Timestamp: $timestamp"
    "- EXE: $ExePath"
    "- Status: $status"
    "- ExitCode: $exitCode"
    "- PID: $pidValue"
    "- WaitSeconds: $WaitSeconds"
    "- HotkeySent: $($PressHotkey.IsPresent)"
    "- LogPath: $logPath"
    ""
    "## App Log"
    ""
    "APP_LOG_BEGIN"
    $logText
    "APP_LOG_END"
)

$mdLines | Set-Content -Path $mdOut -Encoding UTF8

Write-Host "TXT: $txtOut"
Write-Host "MD : $mdOut"
