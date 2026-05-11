#!/usr/bin/env bash
# Stage 4: full 1126-question run with winning (temperature, n_samples) config.
#
# Usage:
#   bash run_phase3_final.sh <best_temperature> <best_n_samples> [--resume]
#
# Run inside tmux:
#   tmux new-session -s p3_final
#   bash scripts/run_phase3_final.sh 0.7 3

set -euo pipefail
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=".venv/lib/python3.13/site-packages/torchvision.libs:${LD_LIBRARY_PATH:-}"
mkdir -p logs

BEST_TEMP="${1:?Usage: bash run_phase3_final.sh <best_temperature> <best_n_samples>}"
BEST_N="${2:?Usage: bash run_phase3_final.sh <best_temperature> <best_n_samples>}"

ARGS=(
    --phase phase3_final
    --full
    --temp "$BEST_TEMP"
    --samples "$BEST_N"
    --max-tokens 32768
    --checkpoint-every 200
    --compare phase2_full
)

if [[ "${3-}" == "--resume" ]]; then
    ARGS+=(--resume)
    echo "[phase3_final] Resume mode."
fi

echo "[phase3_final] Full run: temp=$BEST_TEMP, n_samples=$BEST_N at $(date)"
echo "[phase3_final] Log: logs/phase3_final.log"

.venv/bin/python testing_template.py "${ARGS[@]}" 2>&1 | tee -a logs/phase3_final.log

echo "[phase3_final] Done at $(date)"
