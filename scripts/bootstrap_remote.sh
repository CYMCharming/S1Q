#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/home/cym/S1Q}"
BASE="${2:-/home/cym/miniconda3/envs/sherry-ptq/bin/python}"
mkdir -p "$ROOT/work" "$ROOT/results" "$ROOT/artifacts"
uv venv --system-site-packages --python "$BASE" "$ROOT/.venv"
"$ROOT/.venv/bin/python" - <<'PY'
import importlib.metadata as m
for name in ('torch','transformers','peft','datasets','huggingface-hub','safetensors','pytest','einops'):
    try: print(name,m.version(name))
    except m.PackageNotFoundError: print(name,'MISSING')
PY
echo "Isolated environment prepared; install only missing packages in this venv."
