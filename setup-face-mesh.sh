#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
FACE_PYTHON="${FACE_PYTHON:-python3.12}"
"$FACE_PYTHON" -m venv .venv-face
.venv-face/bin/python -m pip install -r requirement.txt
