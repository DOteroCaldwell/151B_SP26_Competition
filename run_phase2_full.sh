#!/usr/bin/env bash
# run_phase2_full.sh — Phase 2 prompts on all 1,126 public questions.
#
# Designed to run inside a tmux session so SSH disconnects don't kill the job.
# See docs/dsmlp-long-jobs.md for full launch instructions.
#
# Usage:
#   bash run_phase2_full.sh            # fresh run
#   bash run_phase2_full.sh --resume   # pick up after a crash

set -euo pipefail
cd "$(dirname "$0")"

export LD_LIBRARY_PATH=".venv/lib/python3.13/site-packages/torchvision.libs:${LD_LIBRARY_PATH:-}"

mkdir -p logs

ARGS=(
    --phase phase2_full
    --full
    --compare baseline
    --checkpoint-every 200
)

if [[ "${1-}" == "--resume" ]]; then
    ARGS+=(--resume)
    echo "[run_phase2_full] Resume mode: skipping already-processed questions."
fi

echo "[run_phase2_full] Started at $(date)"
echo "[run_phase2_full] Log: logs/phase2_full_run.log"
echo "[run_phase2_full] Args: ${ARGS[*]}"

.venv/bin/python testing_template.py "${ARGS[@]}" 2>&1 | tee -a logs/phase2_full_run.log

echo "[run_phase2_full] Finished at $(date)"
