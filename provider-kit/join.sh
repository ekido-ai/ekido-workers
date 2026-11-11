#!/usr/bin/env bash
# Joins the network and starts taking work. Stop any time with Ctrl+C.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then echo "Run ./setup.sh first."; exit 1; fi
export PYTHONPATH="$PWD/core:$PWD/agent:$PWD/runtime"
exec .venv/bin/python -m gpuchain_agent.join "$@"
