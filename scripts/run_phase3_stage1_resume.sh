#!/usr/bin/env bash
# Resume phase 3 stage 1 for temp07 (from Q100 checkpoint) and run temp09 fresh.
# Skips any temperature whose summary file already exists.
#
# Run inside tmux:
#   tmux new-session -s p3_resume
#   bash scripts/run_phase3_stage1_resume.sh

set -euo pipefail
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=".venv/lib/python3.13/site-packages/torchvision.libs:${LD_LIBRARY_PATH:-}"
mkdir -p logs results

LOG=logs/phase3_stage1_resume.log
log() { echo "[stage1_resume $(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== Stage 1 resume: temp07 + temp09 ==="
log "Node: $(hostname)"

TEMPS=(0.7 0.9)

for TEMP in "${TEMPS[@]}"; do
    PNAME="phase3_temp${TEMP//.}"
    if [[ -f "results/${PNAME}_summary.json" ]]; then
        log "$PNAME already complete — skipping"
        continue
    fi
    log "Starting $PNAME (temp=$TEMP, n=200, max_tokens=32768)"
    .venv/bin/python testing_template.py \
        --phase "$PNAME" \
        --n 200 \
        --temp "$TEMP" \
        --samples 1 \
        --max-tokens 32768 \
        --checkpoint-every 50 \
        --resume \
        --compare phase2_full \
        2>&1 | tee -a "$LOG"
    log "$PNAME done"
done

log "=== Stage 1 resume complete ==="
