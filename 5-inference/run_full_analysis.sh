#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ARGS=()
while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--max-images" ]]; then
        shift 2
    else
        ARGS+=("$1")
        shift
    fi
done

"$SCRIPT_DIR/run_batch_predict.sh" "${ARGS[@]}"
