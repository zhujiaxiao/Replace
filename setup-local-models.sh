#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3.12}"
if [ ! -d .venv-models ]; then "$PYTHON" -m venv .venv-models; fi
.venv-models/bin/python -m pip install --upgrade pip
.venv-models/bin/python -m pip install -r requirement.txt
if [ ! -d local-models/Video-Depth-Anything ]; then
  mkdir -p local-models
  git clone --depth 1 https://github.com/DepthAnything/Video-Depth-Anything.git local-models/Video-Depth-Anything
fi
mkdir -p local-models/Video-Depth-Anything/checkpoints
checkpoint=local-models/Video-Depth-Anything/checkpoints/video_depth_anything_vits.pth
if [ ! -f "$checkpoint" ]; then
  curl -fL --retry 3 -o "$checkpoint" https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth
fi
.venv-models/bin/python -c 'import demucs, torch, transformers; print("Local model runtime: OK")'
