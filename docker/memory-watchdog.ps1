# Stops this project's own programs if the laptop runs short of memory, and writes a log. The limit comes from a
# laptop freeze on 5 Nov. It only matches programs inside this repo's .cache folder, so nothing else can be hit.
#
#   powershell -ExecutionPolicy Bypass -File infra\memory-watchdog.ps1 [-LimitMb 2000]
#
# It exits after stopping them, so start it again before the next run. It doesn't cover Docker.
param([int]$LimitMb = 2000)

$cache = Join-Path (Split-Path $PSScriptRoot -Parent) ".cache\"
$log = Join-Path $env:TEMP "gpuchain_memory.log"
$low = [int]::MaxValue
$below = 0
"Watching. Log: $log"
while ($true) {
    $mb = [int](Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue
    if ($mb -lt $low) { $low = $mb }
    "$(Get-Date -Format HH:mm:ss) available=$mb lowest=$low" | Out-File -Append -Encoding ascii $log
    if ($mb -lt $LimitMb) { $below++ } else { $below = 0 }
    if ($below -ge 2) {
        "$(Get-Date -Format HH:mm:ss) UNDER $LimitMb MB: stopping this project's programs" | Out-File -Append -Encoding ascii $log
        Get-Process | Where-Object { $_.Path -and $_.Path.StartsWith($cache, [System.StringComparison]::OrdinalIgnoreCase) } | Stop-Process -Force -ErrorAction SilentlyContinue
        exit 1
    }
    Start-Sleep -Seconds 3
}
