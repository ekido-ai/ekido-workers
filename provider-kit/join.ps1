# Joins the network and starts taking work. Stop any time with Ctrl+C.
# Run with:  powershell -ExecutionPolicy Bypass -File .\join.ps1 --coordinator <address> --name "My GPU" --price 0.002
$ErrorActionPreference = "Stop"
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Run setup.ps1 first." }
$env:PYTHONPATH = "$PSScriptRoot\core;$PSScriptRoot\agent;$PSScriptRoot\runtime"
Set-Location $PSScriptRoot
& $py -m ekido_agent.join @args
exit $LASTEXITCODE
