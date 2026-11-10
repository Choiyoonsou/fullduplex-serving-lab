#!/usr/bin/env bash
# Experiment console (job queue, GPU scheduler, run browser): http://localhost:${PORT:-8800}
# Remote server: ssh -L 8800:localhost:8800 <host>   (keep BIND=127.0.0.1: anyone reaching the port can start GPU jobs)
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-$PWD/hf_cache}
exec .venv/bin/python console/server.py --port "${PORT:-8800}" --host "${BIND:-127.0.0.1}"
