#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
else
  PYTHON=""
  for candidate in /opt/miniconda3/bin/python3 /opt/homebrew/bin/python3 /usr/bin/python3; do
    if [[ -x "$candidate" ]] && "$candidate" -c 'from PIL import Image, ImageDraw, ImageFont' >/dev/null 2>&1; then
      PYTHON="$candidate"
      break
    fi
  done
fi

if [[ -z "$PYTHON" ]]; then
  echo "未找到带 Pillow 的 Python；可用 PYTHON_BIN=/path/to/python 指定。" >&2
  exit 1
fi

cd "$PROJECT_ROOT"
exec "$PYTHON" 5-inference/render_lateral_pseudolabel_review.py "$@"
