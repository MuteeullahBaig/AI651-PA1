#!/usr/bin/env bash
# Stage-1 exploration: one or two seeds per configuration, early stopping on the 18 validation blocks.
# Usage: bash sweep.sh "<seeds>" <tag>:"<key=value ...>" ...
PY=/c/Users/Matee/miniconda3/envs/gpu_env/python.exe
SEEDS="$1"; shift
for spec in "$@"; do
  tag="${spec%%:*}"; sets="${spec#*:}"
  PYTHONIOENCODING=utf-8 $PY experiment.py --tag "$tag" --seeds $SEEDS --max-epochs 8 --patience 2 --set $sets 2>&1 \
    | grep -E "^\[|Traceback|Error" | grep -v Warning
done
echo "SWEEP DONE"
