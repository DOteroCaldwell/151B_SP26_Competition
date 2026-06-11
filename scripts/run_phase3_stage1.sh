#!/usr/bin/env bash
# Stage 1: temperature sweep [0.3, 0.5, 0.6, 0.7, 0.9] × 200q, n_samples=1
# Skips any temperature whose summary file already exists (safe to restart after a crash).
#
# Run inside tmux:
#   tmux new-session -s p3_stage1
#   bash scripts/run_phase3_stage1.sh

set -euo pipefail
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=".venv/lib/python3.13/site-packages/torchvision.libs:${LD_LIBRARY_PATH:-}"
mkdir -p logs

TEMPS=(0.3 0.5 0.6 0.7 0.9)

echo "[stage1] Starting temperature sweep at $(date)"

for TEMP in "${TEMPS[@]}"; do
    PNAME="phase3_temp${TEMP//.}"
    if [[ -f "results/${PNAME}_summary.json" ]]; then
        echo "[stage1] $PNAME already done — skipping"
        continue
    fi
    echo "[stage1] Starting $PNAME (temp=$TEMP) at $(date)"
    .venv/bin/python testing_template.py \
        --phase "$PNAME" \
        --n 200 \
        --temp "$TEMP" \
        --samples 1 \
        --max-tokens 32768 \
        --compare phase2_full \
        2>&1 | tee -a logs/phase3_stage1.log
    echo "[stage1] $PNAME complete at $(date)"
done

echo "[stage1] All temperature experiments done at $(date)"
