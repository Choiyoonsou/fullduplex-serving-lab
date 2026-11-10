#!/usr/bin/env bash
# Environment setup: Python venv + torch (CUDA) + deps, then download model and data.
#
#   ./setup.sh                 # venv + model + dev/test DailyTalk dialogs (splits/)
#   ./setup.sh --no-download   # environment only
#
# Env vars:
#   TORCH_INDEX  PyTorch wheel index (default cu126; e.g. https://download.pytorch.org/whl/cu128)
#   HF_HOME      where model weights are cached (default ./hf_cache, ~16GB)
#   HF_TOKEN     optional; avoids anonymous Hugging Face rate limits
set -euo pipefail
cd "$(dirname "$0")"

TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}
export HF_HOME=${HF_HOME:-$PWD/hf_cache}
PY=.venv/bin/python

command -v ffmpeg >/dev/null || echo "[warn] ffmpeg not found: only needed for scripts/05_export_demo.py (apt install ffmpeg)"

if command -v uv >/dev/null; then
  [ -x "$PY" ] || uv venv --python 3.12 .venv
  uv pip install --python "$PY" "torch==2.9.1" "torchaudio==2.9.1" --index-url "$TORCH_INDEX"
  uv pip install --python "$PY" -r requirements.txt
else
  echo "[info] uv not found, using python3 -m venv (needs Python >= 3.10 and python3-venv)"
  [ -x "$PY" ] || python3 -m venv .venv
  "$PY" -m pip install -U pip
  "$PY" -m pip install "torch==2.9.1" "torchaudio==2.9.1" --index-url "$TORCH_INDEX"
  "$PY" -m pip install -r requirements.txt
fi

"$PY" - <<'EOF'
import torch
n = torch.cuda.device_count()
print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPUs: {n}")
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i} {p.name} {p.total_memory / 2**30:.1f}GB")
EOF

if [ "${1:-}" != "--no-download" ]; then
  "$PY" scripts/00_download.py   # dev + test splits (200 dialogs)
fi
echo "setup done. Next: PYTHONPATH=. $PY tests/test_policies.py && scripts/run_all.sh"
