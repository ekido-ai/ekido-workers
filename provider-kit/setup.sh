#!/usr/bin/env bash
# GPUChain provider kit, macOS and Linux: a private Python environment in .venv, PyTorch for
# your GPU (or the CPU), and the model and dataset jobs use. Nothing is installed system-wide.
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for candidate in python3.12 python3.13 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then PY="$candidate"; break; fi
done
if [ -z "$PY" ]; then echo "Python 3.11 or newer is needed. Install it from python.org, then run this again."; exit 1; fi
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 11) else 1)'; then echo "$PY is older than 3.11."; exit 1; fi

"$PY" -m venv .venv
. .venv/bin/activate
python -m pip install --quiet --upgrade pip

if [ "$(uname -s)" = "Darwin" ]; then
  echo "macOS: installing PyTorch (jobs run on the CPU here)"
  pip install torch
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  index=cu130
  if [ "${driver:-0}" -lt 580 ]; then index=cu128; fi
  echo "NVIDIA GPU found (driver $driver): installing PyTorch for $index"
  pip install torch --index-url "https://download.pytorch.org/whl/$index"
else
  echo "No NVIDIA GPU found: installing the CPU build"
  pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
pip install -r requirements.txt

echo "Downloading the model and dataset once, so jobs start fast..."
python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('distilbert-base-uncased'); AutoTokenizer.from_pretrained('distilbert-base-uncased')"
python -c "from datasets import load_dataset; load_dataset('fancyzhx/ag_news')"
python -c "import torch; print('Ready on', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'the CPU')"
echo
echo "Setup done. Next, from this folder, run the join command the website shows you:"
echo "  ./join.sh --coordinator <address> --name \"My GPU\" --price 0.002"
