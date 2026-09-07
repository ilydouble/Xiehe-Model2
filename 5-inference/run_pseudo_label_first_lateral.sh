#!/usr/bin/env bash
# 为第一批侧面数据生成只供人工复核的23类LabelMe候选标注。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -x "/opt/miniconda3/bin/python3" ]]; then
    DEFAULT_PYTHON="/opt/miniconda3/bin/python3"
else
    DEFAULT_PYTHON="python3"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/pseudo_label_first_lateral.py" "$@"
