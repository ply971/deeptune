<#
.SYNOPSIS
    One-command project check: pip check, full pytest suite, video smoke runs.

.DESCRIPTION
    Run from anywhere; paths resolve against the repository root.
    Uses .venv\Scripts\python.exe when present, otherwise `python`.
    Writes a JUnit report to .artifacts\check\junit.xml and prints a one-line
    summary at the end. Exits non-zero if any step fails.

.PARAMETER Quick
    Skip the two video smoke runs (they train tiny models end to end).

.EXAMPLE
    .\scripts\check.ps1
    .\scripts\check.ps1 -Quick
#>
param([switch]$Quick)

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { $python = 'python' }

$reportDir = Join-Path $root '.artifacts\check'
New-Item -ItemType Directory -Force $reportDir | Out-Null
$junit = Join-Path $reportDir 'junit.xml'

$results = [ordered]@{}
$failed = $false
$started = Get-Date

$device = & $python -c "import torch; print(torch.__version__ + ' | ' + (('cuda ' + torch.version.cuda + ' | ' + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'cpu'))"
if ($LASTEXITCODE -ne 0) { $device = 'unknown (torch import failed)' }
Write-Host "== device: torch $device"

Write-Host '== pip check'
& $python -m pip check
if ($LASTEXITCODE -eq 0) { $results['pip check'] = 'ok' } else { $results['pip check'] = 'FAIL'; $failed = $true }

Write-Host '== pytest'
& $python -m pytest tests -q --durations=10 -p no:cacheprovider "--junitxml=$junit" | Tee-Object -Variable pytestLines
$pytestCode = $LASTEXITCODE
$summary = ($pytestLines | Where-Object { $_ -match '\d+ (passed|failed|error)' } | Select-Object -Last 1)
if (-not $summary) { $summary = "no summary (exit $pytestCode)" }
$summary = $summary.Trim(' ', '=')
if ($pytestCode -eq 0) { $results['pytest'] = $summary } else { $results['pytest'] = "FAIL: $summary"; $failed = $true }

if ($Quick) {
    $results['video smoke'] = 'skipped (-Quick)'
} else {
    $smokes = @(
        @{ Name = 'smoke resnet18 --manifest'; Args = @('scripts\smoke_video.py', '--model', 'resnet18', '--manifest') },
        @{ Name = 'smoke r3d_18'; Args = @('scripts\smoke_video.py', '--model', 'r3d_18') },
        @{ Name = 'smoke resnet18 --parquet_bytes'; Args = @('scripts\smoke_video.py', '--model', 'resnet18', '--parquet_bytes') }
    )
    foreach ($smoke in $smokes) {
        Write-Host "== $($smoke.Name)"
        & $python @($smoke.Args)
        if ($LASTEXITCODE -eq 0) { $results[$smoke.Name] = 'ok' } else { $results[$smoke.Name] = "FAIL (exit $LASTEXITCODE)"; $failed = $true }
    }
}

$elapsed = [int]((Get-Date) - $started).TotalSeconds
$parts = ($results.GetEnumerator() | ForEach-Object { "$($_.Key): $($_.Value)" }) -join '; '
$status = if ($failed) { 'FAIL' } else { 'PASS' }
Write-Host ''
Write-Host "CHECK $status | $parts | device: $device | ${elapsed}s"
if ($failed) { exit 1 }
exit 0
