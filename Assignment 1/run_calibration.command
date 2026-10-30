#!/bin/zsh
set -eu

# Use the selected Python environment; install requirements.txt into it first.
TASK_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mmuq-matplotlib"
"$PYTHON_EXECUTABLE" -u "$TASK_DIR/ab_calibrate_hbv.py" "$@"
