#!/usr/bin/env bash
# Phase 3 end-to-end pipeline:
#   Stage 1: temperature sweep [0.3, 0.5, 0.6, 0.7, 0.9] × 200q
#   Stage 2: majority voting sweep [n=3, 5, 7] × 200q at best temperature
#   Final:   full 1126q run at best (temperature, n_samples)
#
# Crash-resilient: each run uses --checkpoint-every and --resume.
# Skip logic: if a summary file already exists, the run is skipped.
#
# Run inside tmux to survive SSH disconnects:
#   tmux new-session -s p3_pipeline
#   bash scripts/run_phase3_pipeline.sh
#
# Resume after a crash: just re-run the same command — all completed runs skip,
# interrupted runs resume from their last checkpoint.
#
# To skip the final 1126q run (e.g. to inspect stage 2 first):
#   RUN_FINAL=0 bash scripts/run_phase3_pipeline.sh

set -euo pipefail
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=".venv/lib/python3.13/site-packages/torchvision.libs:${LD_LIBRARY_PATH:-}"
mkdir -p logs results

RUN_FINAL="${RUN_FINAL:-1}"
LOG=logs/phase3_pipeline.log

log() { echo "[pipeline $(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== Phase 3 pipeline starting ==="
log "Node: $(hostname)"
log "RUN_FINAL=$RUN_FINAL"

# ── Stage 1: temperature sweep ─────────────────────────────────────────────────
log "--- Stage 1: temperature sweep [0.3 0.5 0.6 0.7 0.9] ---"

TEMPS=(0.3 0.5 0.6 0.7 0.9)

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

log "--- Stage 1 complete ---"

# ── Select best temperature ────────────────────────────────────────────────────
log "Selecting best temperature from stage 1 results..."

BEST_TEMP=$(.venv/bin/python - <<'PYEOF'
import json, glob, re, sys

results = []
for path in sorted(glob.glob("results/phase3_temp*_summary.json")):
    m = re.search(r"phase3_temp(\d+)_summary", path)
    if not m:
        continue
    raw = m.group(1)               # e.g. "03", "07"
    temp = int(raw) / 10           # "03" -> 0.3, "07" -> 0.7
    data = json.loads(open(path).read())
    acc = data.get("overall_accuracy", 0)
    results.append((temp, acc))
    print(f"  temp={temp:.1f}: {acc:.2f}%", file=sys.stderr)

if not results:
    print("ERROR: no stage1 summary files found", file=sys.stderr)
    sys.exit(1)

best_temp, best_acc = max(results, key=lambda x: x[1])
print(f"Best: temp={best_temp:.1f} ({best_acc:.2f}%)", file=sys.stderr)

# Save for reference
import os
os.makedirs("results", exist_ok=True)
with open("results/phase3_best_config.json", "w") as f:
    json.dump({"best_temp": best_temp, "stage1_results": [
        {"temp": t, "overall_accuracy": a} for t, a in sorted(results)
    ]}, f, indent=2)

print(best_temp)
PYEOF
)

BEST_TEMP=$(echo "$BEST_TEMP" | tr -d '[:space:]')
log "Best temperature: $BEST_TEMP"

# ── Stage 2: majority voting sweep ────────────────────────────────────────────
BEST_TEMP_PNAME="phase3_temp${BEST_TEMP//.}"
log "--- Stage 2: majority voting [3 5 7] at temp=$BEST_TEMP (compare: $BEST_TEMP_PNAME) ---"

N_VALUES=(3 5 7)

for N in "${N_VALUES[@]}"; do
    PNAME="phase3_mv${N}"
    if [[ -f "results/${PNAME}_summary.json" ]]; then
        log "$PNAME already complete — skipping"
        continue
    fi
    log "Starting $PNAME (n_samples=$N, temp=$BEST_TEMP, n=200, max_tokens=32768)"
    .venv/bin/python testing_template.py \
        --phase "$PNAME" \
        --n 200 \
        --temp "$BEST_TEMP" \
        --samples "$N" \
        --max-tokens 32768 \
        --checkpoint-every 25 \
        --resume \
        --compare "$BEST_TEMP_PNAME" \
        2>&1 | tee -a "$LOG"
    log "$PNAME done"
done

log "--- Stage 2 complete ---"

# ── Select best n_samples ──────────────────────────────────────────────────────
log "Selecting best n_samples from stage 2 results..."

BEST_N=$(.venv/bin/python - <<'PYEOF'
import json, glob, re, sys

results = []
for path in sorted(glob.glob("results/phase3_mv*_summary.json")):
    m = re.search(r"phase3_mv(\d+)_summary", path)
    if not m:
        continue
    n = int(m.group(1))
    data = json.loads(open(path).read())
    acc = data.get("overall_accuracy", 0)
    results.append((n, acc))
    print(f"  n_samples={n}: {acc:.2f}%", file=sys.stderr)

if not results:
    print("ERROR: no stage2 summary files found", file=sys.stderr)
    sys.exit(1)

# Also factor in n=1 from stage1 for comparison
try:
    cfg = json.loads(open("results/phase3_best_config.json").read())
    best_temp = cfg["best_temp"]
    pname = f"results/phase3_temp{str(best_temp).replace('.','')}_summary.json"
    n1_data = json.loads(open(pname).read())
    n1_acc = n1_data.get("overall_accuracy", 0)
    results.append((1, n1_acc))
    print(f"  n_samples=1 (baseline): {n1_acc:.2f}%", file=sys.stderr)
except Exception as e:
    print(f"  (could not load n=1 baseline: {e})", file=sys.stderr)

best_n, best_acc = max(results, key=lambda x: x[1])
print(f"Best: n_samples={best_n} ({best_acc:.2f}%)", file=sys.stderr)

# Update phase3_best_config.json
existing = {}
try:
    existing = json.loads(open("results/phase3_best_config.json").read())
except Exception:
    pass
existing["best_n_samples"] = best_n
existing["stage2_results"] = [
    {"n_samples": n, "overall_accuracy": a} for n, a in sorted(results)
]
with open("results/phase3_best_config.json", "w") as f:
    json.dump(existing, f, indent=2)

print(best_n)
PYEOF
)

BEST_N=$(echo "$BEST_N" | tr -d '[:space:]')
log "Best n_samples: $BEST_N"

# ── Final run ─────────────────────────────────────────────────────────────────
if [[ "$RUN_FINAL" == "1" ]]; then
    if [[ -f "results/phase3_final_summary.json" ]]; then
        log "phase3_final already complete — skipping"
    else
        log "--- Final run: 1126q, temp=$BEST_TEMP, n_samples=$BEST_N ---"
        .venv/bin/python testing_template.py \
            --phase phase3_final \
            --full \
            --temp "$BEST_TEMP" \
            --samples "$BEST_N" \
            --max-tokens 32768 \
            --checkpoint-every 100 \
            --resume \
            --compare phase2_full \
            2>&1 | tee -a "$LOG"
        log "Final run complete"
    fi
else
    log "Skipping final run (RUN_FINAL=0)"
fi

log "=== Phase 3 pipeline complete ==="
log "Best config: temp=$BEST_TEMP, n_samples=$BEST_N"
log "See results/phase3_best_config.json for stage 1 and 2 breakdowns"
