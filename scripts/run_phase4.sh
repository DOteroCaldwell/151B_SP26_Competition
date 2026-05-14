#!/usr/bin/env bash
# Phase 4: precision prompt sweep (A, B, C, D, E, ABC) × 200q, temp=0.7,
# n_samples=1, max_tokens=32768. Compares each against phase3_temp07.
#
# Skips any candidate whose summary file already exists (safe to restart
# after a crash or pod expiration). Each candidate is one full harness run.
#
# Run inside tmux:
#   tmux new-session -d -s p4_sweep "bash scripts/run_phase4.sh"
#
# See docs/process.md "Phase 4" for the rationale and the exact prompt text
# tested by each candidate.

set -euo pipefail
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=".venv/lib/python3.13/site-packages/torchvision.libs:${LD_LIBRARY_PATH:-}"
mkdir -p logs

# ── Candidate prompt deltas ───────────────────────────────────────────────────
# Each is appended to SYSTEM_PROMPT_MATH via --prompt-math-append.

PROMPT_A='Express the final answer in exact form (fractions like 1/3, radicals like \sqrt{2}, symbolic constants like \pi, e) whenever possible. Only convert to a decimal if the problem explicitly asks for a decimal or numerical approximation.'

PROMPT_B='When a decimal answer is required, report at least 10 significant figures. Do not round intermediate calculations — only round the final answer, and only if the problem specifies a precision.'

PROMPT_C='Read the problem for precision cues (round to N decimal places, to the nearest integer, exact value, accurate to N decimal places). Match the requested precision exactly. If no precision is specified, prefer exact symbolic form, or otherwise 10 significant figures.'

PROMPT_D='Do not write more decimal digits than you can derive from your work. If you cannot verify a digit with explicit arithmetic, do not include it. Prefer fewer correct digits over more guessed digits.'

PROMPT_E='Before placing your final answer in \boxed{}, verify it by either (1) substituting it back into the original equation/condition, or (2) re-computing the final arithmetic step from scratch. State the verification result briefly. If the check fails, redo the calculation.'

PROMPT_ABC='Express the final answer in exact form (fractions, radicals, \pi, e) whenever possible. Only convert to a decimal when the problem explicitly asks for a numerical value or when an exact form is not available. If a decimal is required, report at least 10 significant figures, do not round intermediate calculations, and match any precision specified in the problem (round to N places, nearest integer, etc.).'

# ── Candidate table: (phase_name, prompt_var) ────────────────────────────────
CANDIDATES=(
    "phase4_a:PROMPT_A"
    "phase4_b:PROMPT_B"
    "phase4_c:PROMPT_C"
    "phase4_d:PROMPT_D"
    "phase4_e:PROMPT_E"
    "phase4_abc:PROMPT_ABC"
)

echo "[phase4] Starting precision prompt sweep at $(date)"
echo "[phase4] Candidates: ${CANDIDATES[*]}"
echo "[phase4] Base config: 200q, temp=0.7, n_samples=1, max_tokens=32768, compare=phase3_temp07"

for ENTRY in "${CANDIDATES[@]}"; do
    PNAME="${ENTRY%%:*}"
    VAR="${ENTRY##*:}"
    PROMPT="${!VAR}"

    if [[ -f "results/${PNAME}_summary.json" ]]; then
        echo "[phase4] $PNAME already complete — skipping"
        continue
    fi

    echo "[phase4] === Starting $PNAME at $(date) ==="
    echo "[phase4] Prompt append: ${PROMPT}"

    .venv/bin/python testing_template.py \
        --phase "$PNAME" \
        --n 200 \
        --temp 0.7 \
        --samples 1 \
        --max-tokens 32768 \
        --checkpoint-every 100 \
        --compare phase3_temp07 \
        --prompt-math-append "$PROMPT" \
        2>&1 | tee -a logs/phase4.log

    echo "[phase4] === $PNAME complete at $(date) ==="
done

echo "[phase4] All candidates complete at $(date)"
echo "[phase4] Summary files: results/phase4_*_summary.json"
echo "[phase4] Next: identify winner and run on full 1126q (see docs/process.md §4.2)"
