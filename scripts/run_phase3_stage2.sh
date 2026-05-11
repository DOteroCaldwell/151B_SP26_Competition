#!/usr/bin/env bash
# Stage 2: majority voting sweep [n=3, 5, 7] × 200q at best temperature.
# Skips any n whose summary file already exists.
#
# Usage:
#   bash run_phase3_stage2.sh <best_temperature>
#
# Run inside tmux:
#   tmux new-session -s p3_stage2
#   bash scripts/run_phase3_stage2.sh 0.7

set -euo pipefail
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=".venv/lib/python3.13/site-packages/torchvision.libs:${LD_LIBRARY_PATH:-}"
mkdir -p logs

BEST_TEMP="${1:?Usage: bash run_phase3_stage2.sh <best_temperature>}"
N_VALUES=(3 5 7)

echo "[stage2] Starting majority voting sweep (temp=$BEST_TEMP) at $(date)"

for N in "${N_VALUES[@]}"; do
    PNAME="phase3_mv${N}"
    if [[ -f "results/${PNAME}_summary.json" ]]; then
        echo "[stage2] $PNAME already done — skipping"
        continue
    fi
    echo "[stage2] Starting $PNAME (n=$N, temp=$BEST_TEMP) at $(date)"
    .venv/bin/python testing_template.py \
        --phase "$PNAME" \
        --n 200 \
        --temp "$BEST_TEMP" \
        --samples "$N" \
        --max-tokens 32768 \
        --checkpoint-every 50 \
        --compare "phase3_temp${BEST_TEMP//.}" \
        2>&1 | tee -a logs/phase3_stage2.log
    echo "[stage2] $PNAME complete at $(date)"
done

echo "[stage2] All majority voting experiments done at $(date)"
