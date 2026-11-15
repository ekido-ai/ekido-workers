# Ekido provider kit, Windows: a private Python environment in .venv, PyTorch for your GPU
# (or the CPU), and the model and dataset jobs use. Nothing is installed system-wide.
# Run with:  powershell -ExecutionPolicy Bypass -File .\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$made = $false
if (Get-Command py -ErrorAction SilentlyContinue) {
  # `py -3.12` writes to stderr when that version isn't installed. Under "Stop", Windows PowerShell 5.1 turns that
  # into a terminating error, so try each version with errors allowed and judge by the exit code alone.
  $ErrorActionPreference = "Continue"
  foreach ($v in "3.12", "3.13", "3.11") {
    & py "-$v" -m venv .venv 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { $made = $true; break }
  }
  $ErrorActionPreference = "Stop"
}
if (-not $made) {
  if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw "Python 3.11 or newer is needed. Install it from python.org, then run this again." }
  & python -m venv .venv
  if ($LASTEXITCODE -ne 0) { throw "Couldn't create the .venv folder with python. Check that Python 3.11 or newer is installed." }
}
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $py -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 11) else 1)"
if ($LASTEXITCODE -ne 0) { throw "Python 3.11 or newer is needed." }
& $py -m pip install --quiet --upgrade pip

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  $driver = [int]((& nvidia-smi --query-gpu=driver_version --format=csv,noheader | Select-Object -First 1).Split(".")[0])
  $index = if ($driver -ge 580) { "cu130" } else { "cu128" }
  Write-Host "NVIDIA GPU found (driver $driver): installing PyTorch for $index"
  & $py -m pip install torch --index-url "https://download.pytorch.org/whl/$index"
} else {
  Write-Host "No NVIDIA GPU found: installing the CPU build"
  & $py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
}
if ($LASTEXITCODE -ne 0) { throw "Installing PyTorch failed. Check your internet connection and that your Python version has PyTorch wheels (3.11 to 3.13 are safest)." }
& $py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Installing the kit's requirements failed. See the pip message above." }

Write-Host "Downloading the model and dataset once, so jobs start fast..."
& $py -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('distilbert-base-uncased'); AutoTokenizer.from_pretrained('distilbert-base-uncased')"
& $py -c "from datasets import load_dataset; load_dataset('fancyzhx/ag_news')"
& $py -c "import torch; print('Ready on', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'the CPU')"
Write-Host ""
Write-Host "Setup done. Next, from this folder, run the join command the website shows you."
