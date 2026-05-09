# Process Documentation

This document tracks all work performed on the CSE 151B Math Reasoning Competition, organized by phase and aligned with milestone report deliverables.

**Objective:** Improve Qwen3-4B-Thinking-2507's mathematical reasoning accuracy through model-intrinsic methods (prompt engineering, supervised fine-tuning, reinforcement learning).

---

## Phase 1: Establish Baseline (In Progress)

**Goal:** Run inference on all 1,126 public questions, measure accuracy by question type, log error patterns for targeted improvements.

**Deliverables for milestone report:**
- Baseline accuracy: overall, MCQ, free-form (single-part vs. multi-part)
- Error analysis: categorization of failure modes
- Reproducibility details: model, quantization, hyperparameters, hardware

### 1.1 Baseline Inference Script

**Created:** `run_baseline.py`

**Configuration:**
- **Model:** Qwen3-4B-Thinking-2507 (INT8 BitsAndBytes quantization via vLLM)
- **Inference method:** vLLM with batched generation
- **vLLM parameters:**
  - `gpu_memory_utilization=0.50`
  - `max_model_len=16384`
  - `max_num_seqs=4`, `max_num_batched_tokens=8192`
- **Sampling parameters:**
  - `temperature=0.6` (deterministic but not greedy)
  - `top_p=0.95`, `top_k=20` (diversity-focused)
  - `max_tokens=2048` (sufficient for reasoning traces)
- **Hardware:** NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation (1x GPU, 24GB VRAM with MIG)

**Scoring:**
- **MCQ:** Extract last letter from `\boxed{}` via regex; match against gold letter (case-insensitive)
- **Free-form:** Use `judger.auto_judge()` (symbolic/numeric equivalence, tolerance 1e-8)
- **Multi-part:** All sub-answers must be correct for question to count

**Outputs:**
- `results/baseline_results.jsonl` — complete results per question
- `results/baseline_errors.jsonl` — detailed error logs with categorization
- `results/baseline_summary.json` — quantitative summary by question type

**Error Categories:**
- `MCQ_NO_VALID_LETTER` — response has no valid letter or extracted letter ∉ {A-J}
- `MCQ_WRONG_LETTER` — extracted letter exists but doesn't match gold
- `FREE_FORM_EMPTY` — response is empty
- `FREE_FORM_NO_BOX` — response has no `\boxed{}` wrapper
- `FREE_FORM_EXTRACTION_FAILED` — `judger.extract_ans()` could not parse response
- `FREE_FORM_WRONG_ANSWER` — answer extracted but doesn't match gold

### 1.2 Running the baseline

**Command:**
```bash
python run_baseline.py
```

**Expected runtime:** ~6–8 hours (1,126 questions × ~20 sec per batch of 8)

**Expected results (target for comparison):**
- Overall accuracy: [TBD — will measure]
- MCQ accuracy: [TBD]
- Free-form accuracy: [TBD]
- Error breakdown: [TBD]

**Status:** Ready to run. Script fully implements Phase 1 from objectives.md.

---

## Phase 2: Low-effort prompt fixes (Planned)

**Goal:** Address formatting issues identified in Phase 1 errors (multi-part answers, MCQ confusion).

**Plan:**
- Strengthen system prompt to explicitly state multi-part answer format
- Add one-shot example for MCQ if extraction reliability is low
- Adjust `MAX_TOKENS` if responses are truncated

**Timeline:** After Phase 1 results available (week of [date TBD])

---

## Phase 3: Sampling strategy (Planned)

**Goal:** Majority voting and temperature tuning to boost accuracy without retraining.

**Plan:**
- Implement N-sample majority voting (N=8)
- Temperature sweep on held-out slice (e.g., 200 questions)
- Best-of-N for free-form only

**Timeline:** Week [TBD]

---

## Phase 4: Prompt engineering (Planned)

**Goal:** Few-shot exemplars and chain-of-thought steering.

**Plan:**
- Add 2–3 worked examples in user turn
- Explicit instruction: "write 'Therefore, the answer is X' before `\boxed{X}`"
- Domain-specific prompts for high-error categories (calculus, complex analysis, etc.)

**Timeline:** Week [TBD]

---

## Phase 5: Faster inference (Planned)

**Goal:** Reduce wall-clock time for majority voting experiments.

**Plan:**
- Switch from Transformers to vLLM (INT8 quantization)
- Tune batch size and GPU memory utilization
- Measure throughput improvement

**Timeline:** Week [TBD]

---

## Phase 6: Supervised fine-tuning (Planned)

**Goal:** Train on public questions + external datasets to improve answer format and correctness.

**Plan:**
- SFT on public data (1,126 questions) with 80/20 train/val split
- Augment with MATH, GSM8K, AIME datasets
- LoRA/QLoRA to keep memory manageable
- Merge adapters before final inference

**Timeline:** Week [TBD]

---

## Phase 7: Reinforcement learning (Planned)

**Goal:** Use `judger.py` as reward signal to refine generated solutions.

**Plan:**
- GRPO or DPO with binary reward (correct=+1, wrong=0)
- Generate solutions, filter with `judger.py`, use correct ones as preferred
- Alternative: outcome-based reward model

**Timeline:** Week [TBD]

---

## Phase 8: Submission (Planned)

**Goal:** Generate final CSV submission with model responses on private test set.

**Plan:**
- Write submission script (JSONL → CSV format)
- Run final inference on `private.jsonl` (no ground truth)
- Submit to Kaggle

**Timeline:** Before competition deadline

---

## Deliverables Summary

| Deliverable | Status | Location |
|---|---|---|
| CLAUDE.md (codebase guide) | ✅ Done | `/CLAUDE.md` |
| objectives.md (plan of attack) | ✅ Done | `docs/objectives.md` |
| kaggle-overview.md (competition details) | ✅ Done | `docs/kaggle-overview.md` |
| Baseline accuracy (all splits) | 🟡 In progress | `results/baseline_summary.json` |
| Error analysis + categorization | 🟡 In progress | `results/baseline_errors.jsonl` |
| Reproducibility documentation | 🟡 In progress | `docs/process.md` (this file) |
| Improved prompt engineering | ⬜ Planned | TBD |
| Fine-tuned model checkpoint | ⬜ Planned | TBD |
| Final leaderboard score + rank | ⬜ Planned | Will be in milestone report abstract |

---

## Key Metrics (For Milestone Report)

### Phase 1 Results (Expected after baseline runs)
- **Overall accuracy:** [measured]
- **MCQ accuracy:** [measured]
- **Free-form accuracy:** [measured]
- **Error rate breakdown:** [by category]

### Improvements (To be measured after each phase)
- **Accuracy gain vs baseline:** (Phase X result - Phase 1 baseline) / Phase 1 baseline
- **Computational cost:** wall-clock time, VRAM usage

---

## Notes

- All code is reproducible: script specifies model ID, quantization method, sampling params, batch size, GPU type
- Results are version-controlled in git: `results/baseline_*.jsonl` and `results/baseline_summary.json`
- Error logs enable targeted debugging: each failure includes question excerpt, full response, extracted answer, and error category
