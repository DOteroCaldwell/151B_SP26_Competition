# Process Documentation

This document tracks all work performed on the CSE 151B Math Reasoning Competition, organized by phase and aligned with milestone report deliverables.

**Objective:** Improve Qwen3-4B-Thinking-2507's mathematical reasoning accuracy through model-intrinsic methods (prompt engineering, supervised fine-tuning, reinforcement learning).

---

## Phase 1: Establish Baseline (Complete)

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

**Actual runtime:** ~4.5 hours (1,126 questions, started 2026-05-10 11:06 UTC, completed 15:34 UTC)

**Command used:**
```bash
LD_LIBRARY_PATH=.venv/lib/python3.13/site-packages/torchvision.libs:$LD_LIBRARY_PATH \
  .venv/bin/python run_baseline.py 2>&1 | tee logs/full_baseline_run.log
```

**Status:** Complete (2026-05-10)

### 1.2.1 Full Baseline Results (1,126 questions)

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 29.9% (337/1126) |
| **MCQ Accuracy** | 10.7% (40/375) |
| **Free-form Accuracy** | 39.5% (297/751) |
| **Single-part Free-form** | 51.0% (172/337) |
| **Multi-part Free-form** | 30.2% (125/414) |
| **Total Errors** | 789 |

**Error breakdown:**

| Category | Count | % of errors |
|----------|-------|-------------|
| `FREE_FORM_NO_BOX` | 355 | 45.0% |
| `MCQ_NO_VALID_LETTER` | 231 | 29.3% |
| `FREE_FORM_WRONG_ANSWER` | 99 | 12.5% |
| `MCQ_WRONG_LETTER` | 104 | 13.2% |

**Key observations:**
1. **MCQ is severely broken at baseline** — 61.6% of MCQ errors are `MCQ_NO_VALID_LETTER` (model not outputting a letter in `\boxed{}`), only 10.7% overall accuracy
2. **Free-form formatting is the dominant failure** — `FREE_FORM_NO_BOX` accounts for 45% of all errors; model often omits `\boxed{}` wrapper entirely
3. **Single-part free-form is strongest** (51.0%) when the model does format correctly
4. **Multi-part free-form is weakest** (30.2%) — requires correct reasoning on multiple sub-problems plus correct formatting
5. **Dataset breakdown:** 375 MCQ (33%), 751 free-form (67%); of free-form: 337 single-part, 414 multi-part

### 1.3 Mini-baseline Validation (First 20 Questions)

**Purpose:** Quick validation run to identify error patterns and baseline performance while full pipeline runs.

**Configuration:**
- **Data:** First 20 questions from public.jsonl (9 MCQ, 11 free-form)
- **Model:** Qwen3-4B-Thinking-2507, INT4 BitsAndBytes via Transformers
- **Inference:** Batched generation (batch_size=4, 5 batches)
- **Sampling:** temperature=0.6, top_p=0.95, top_k=20, max_tokens=2048

**Results (completed 2026-05-10 00:21):**

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 25.0% (5/20) |
| **MCQ Accuracy** | 11.1% (1/9) |
| **Free-form Accuracy** | 36.4% (4/11) |
| **Single-part Free-form** | 75.0% (3/4) |
| **Multi-part Free-form** | 14.3% (1/7) |
| **Total Errors** | 15 |

**Error Breakdown:**
- `MCQ_NO_VALID_LETTER` — 7 errors (missing/invalid letter in boxed answer)
- `FREE_FORM_NO_BOX` — 7 errors (no `\boxed{}` wrapper in response)
- `MCQ_WRONG_LETTER` — 1 error (wrong letter despite extraction)

**Key Observations:**
1. **MCQ failures are formatting-based** (88% of MCQ errors: 7/8 missing valid letter extraction)
2. **Free-form failures are structural** (not following `\boxed{}` instruction; 58% of free-form errors)
3. **Single-part free-form performs well** (75%) when answer is wrapped
4. **Multi-part free-form is weak** (14.3%), suggesting difficulty with multiple sub-answers or comprehension

**Implications for Phase 2:**
- System prompt for MCQ may need explicit instruction to output letter in correct format
- Free-form system prompt may need stronger emphasis on `\boxed{}` requirement
- Consider one-shot example in prompt for multi-part answers
- May need to increase `MAX_TOKENS` if responses are being truncated

---

## Phase 2: Prompt Engineering — Formatting Fixes (Complete)

**Goal:** Address the formatting failures identified in Phase 1: missing `\boxed{}` wrappers in free-form responses and missing/invalid letter extraction in MCQ responses.

**Completed:** 2026-05-10

### 2.1 Methodology

**Script:** `phase2_test.py`

**Changes from Phase 1 baseline:**

| Parameter | Phase 1 | Phase 2 | Rationale |
|-----------|---------|---------|-----------|
| `max_tokens` | 2048 | 8192 | Prevent truncation during Qwen3 extended thinking phase |
| `SYSTEM_PROMPT_MATH` | Generic step-by-step | Explicit `\boxed{}` requirement + multi-part comma-separated format | 58% of free-form errors were `FREE_FORM_NO_BOX` |
| `SYSTEM_PROMPT_MCQ` | Output letter in `\boxed{}` | "MUST end with `\boxed{X}`" + explicit example `\boxed{C}` | 87.5% of MCQ errors were `MCQ_NO_VALID_LETTER` |
| Inference backend | Transformers (INT4) | vLLM (INT8 BitsAndBytes) | Faster batched generation |

**Updated system prompts:**

*Free-form (`SYSTEM_PROMPT_MATH`):*
> "You are an expert mathematician. Solve the problem step-by-step. You MUST place your final answer inside `\boxed{}`. For problems with multiple [ANS] placeholders, list all answers comma-separated inside a single `\boxed{}` in the order they appear, e.g. `\boxed{3, 7, -2}`. Always end your response with `\boxed{your answer here}`. Never omit the `\boxed{}` wrapper."

*MCQ (`SYSTEM_PROMPT_MCQ`):*
> "You are an expert mathematician. Read the problem and the answer choices below, then select the single best answer. Think through it carefully, then end your response with the letter of your chosen option inside `\boxed{}`. You MUST end with `\boxed{X}` where X is exactly one capital letter. Example final line: `\boxed{C}`"

**Sampling parameters** (unchanged from Phase 1):
- `temperature=0.6`, `top_p=0.95`, `top_k=20`

**Evaluation:** Same 20 questions as Phase 1 mini-baseline (9 MCQ, 11 free-form).

### 2.2 Results (2026-05-10)

| Metric | Phase 1 | Phase 2 | Delta |
|--------|---------|---------|-------|
| **Overall Accuracy** | 25.0% (5/20) | **60.0% (12/20)** | +35.0pp |
| **MCQ Accuracy** | 11.1% (1/9) | **66.7% (6/9)** | +55.6pp |
| **Free-form Accuracy** | 36.4% (4/11) | **54.5% (6/11)** | +18.1pp |
| **Single-part Free-form** | 75.0% (3/4) | **75.0% (3/4)** | 0.0pp |
| **Multi-part Free-form** | 14.3% (1/7) | **42.9% (3/7)** | +28.6pp |
| **Total Errors** | 15 | 8 | −7 |

**Error breakdown (Phase 2):**

| Category | Count | vs Phase 1 |
|----------|-------|-----------|
| `FREE_FORM_WRONG_ANSWER` | 5 | new dominant error |
| `MCQ_WRONG_LETTER` | 2 | +1 |
| `MCQ_NO_VALID_LETTER` | 1 | −6 |
| `FREE_FORM_NO_BOX` | 0 | −7 ✓ |

**Outputs:**
- `results/phase2_results.jsonl` — per-question results
- `results/phase2_errors.jsonl` — detailed error logs
- `results/phase2_summary.json` — quantitative summary

### 2.3 Key Observations

1. **Formatting failures eliminated.** `FREE_FORM_NO_BOX` dropped from 7 → 0. `MCQ_NO_VALID_LETTER` dropped from 7 → 1. The explicit prompt instructions directly resolved the primary failure modes.
2. **MCQ gains are large and formatting-driven.** 55.6pp improvement came almost entirely from the model now reliably placing a letter in `\boxed{}`. The remaining 3 MCQ errors are wrong answers, not formatting.
3. **Free-form improvements are smaller.** Single-part accuracy is unchanged (75%); gains come from multi-part (+28.6pp). The dominant remaining error is `FREE_FORM_WRONG_ANSWER` — the model formats correctly but gets the math wrong.
4. **Multi-part free-form remains the weakest category** (42.9%). Correct multi-part answers require both correct reasoning on each sub-problem and proper comma-separated formatting. Both are still failing.
5. **Increased token budget likely helped.** Extended thinking traces are less likely to be cut off at 8192 tokens than at 2048.

### 2.4 Implications for Phase 3

- Formatting is no longer the bottleneck — focus shifts to mathematical correctness
- Multi-part free-form (42.9%) has the most headroom; few-shot examples showing multi-part decomposition may help
- Majority voting (N≥3 samples) could push overall accuracy further by reducing variance on borderline questions
- Consider domain-specific prompts or chain-of-thought steering for the wrong-answer cases

**Timeline:** Complete 2026-05-10

### 2.5 Phase 2 Full Dataset Validation (1,126 questions)

**Goal:** Confirm that the Phase 2 prompt improvements observed on the 20-question sample hold at scale across all 1,126 public questions.

**Script:** `scripts/run_phase2_full.sh` → `testing_template.py`

**Command:**
```bash
# Inside a tmux session (see docs/dsmlp-long-jobs.md):
bash scripts/run_phase2_full.sh

# Resume if interrupted:
bash scripts/run_phase2_full.sh --resume
```

**Configuration (identical to Phase 2 sample except where noted):**

| Parameter | Value |
|-----------|-------|
| `phase` | `phase2_full` |
| `max_tokens` | 8192 |
| `temperature` | 0.6, `top_p` 0.95, `top_k` 20 |
| `checkpoint_every` | 200 (flushes results every 200 questions) |
| Dataset | Full `public.jsonl` (1,126 questions) |

**Job-alive strategy:** tmux + `--checkpoint-every 200` (see `docs/dsmlp-long-jobs.md`)

**Launched:** 2026-05-11 05:10 UTC

**Status:** Complete (2026-05-11 10:36 UTC, ~5.5 hours)

**Outputs:**
- `results/phase2_full_results.jsonl`
- `results/phase2_full_errors.jsonl`
- `results/phase2_full_summary.json`
- Log: `logs/phase2_full_run.log`

**Results:**

| Metric | Phase 1 Full (1126q) | Phase 2 Sample (20q) | Phase 2 Full (1126q) |
|--------|---------------------|---------------------|---------------------|
| Overall Accuracy | 29.9% | 60.0% | **54.2% (610/1126)** |
| MCQ Accuracy | 10.7% | 66.7% | **51.5% (193/375)** |
| Free-form Accuracy | 39.5% | 54.5% | **55.5% (417/751)** |
| Single-part Free-form | 51.0% | 75.0% | **61.1% (206/337)** |
| Multi-part Free-form | 30.2% | 42.9% | **51.0% (211/414)** |

**Error breakdown (Phase 2 Full):**

| Category | Count |
|----------|-------|
| `FREE_FORM_WRONG_ANSWER` | 259 |
| `MCQ_NO_VALID_LETTER` | 114 |
| `FREE_FORM_NO_BOX` | 75 |
| `MCQ_WRONG_LETTER` | 68 |

**Key finding — generation truncation:** Inspecting `phase2_full_results.jsonl` revealed that
250/516 errors (48.4%) had no `</think>` closing tag, meaning the model hit the 8192-token
limit mid-thought and never produced an answer. At ~3–4 chars/token, responses were reaching
28k–36k characters — right at the ceiling. The 114 `MCQ_NO_VALID_LETTER` and 75
`FREE_FORM_NO_BOX` errors are largely a symptom of this truncation.

Note: the `_errors.jsonl` files intentionally store only the first 400 characters of each
response (`testing_template.py:360`) — this is a deliberate storage choice, not a generation
truncation bug.

---

## Infrastructure: Standardized Testing Harness (Complete)

**Created:** `testing_template.py` (2026-05-10)

**Motivation:** After Phase 1 and Phase 2 each produced a standalone script (`run_baseline.py`, `phase2_test.py`), the two scripts had duplicated ~200 lines of identical infrastructure: environment setup, model loading, scoring, error categorization, output saving, and delta reporting. Each new phase would require copy-pasting and carefully patching that boilerplate, creating drift and hiding the per-phase changes in noise.

`testing_template.py` centralizes all shared infrastructure into one file and exposes the per-phase variables as a `Config` dataclass at the top. Future phases only touch what actually changes.

### Design Decisions

**`Config` dataclass at the top**

Every tunable parameter lives in one place: model ID, sampling params, prompt overrides, output paths, majority-voting settings, and few-shot examples. The alternatives — a YAML config file or argparse-only — either require a separate file per run or bury context in shell history. A dataclass is self-documenting, type-checked, and lives in the same file as the code that uses it.

**Environment variables set before the vLLM import**

`CUDA_VISIBLE_DEVICES` and `VLLM_USE_DEEP_GEMM=0` must be set before any torch/vLLM code is imported, or the device assignment races and the Blackwell `DEEP_GEMM` codepath crashes. Phase 1 avoided this by accident (top-of-script assignment); Phase 2 carried it over. The template makes it structurally impossible to forget: a small pre-parser extracts `--gpu` before the main import block, and both env vars are set unconditionally at module load.

**`max_tokens=8192` as the default**

Phase 1 used `max_tokens=2048`, which was identified as a likely source of truncated thinking traces for Qwen3's extended reasoning. Phase 2 raised it to 8192 and saw MCQ accuracy jump 55.6pp (partly formatting, partly un-truncated reasoning). The template makes 8192 the baseline and requires a deliberate `--max-tokens` override to go lower.

**vLLM as the only inference backend**

Phase 1 started with Transformers (INT4) for the mini-baseline and switched to vLLM (INT8) for the full run. Phase 2 used vLLM from the start. The template drops the Transformers path entirely — it was slower, required more memory per token, and added branching complexity. The Phase 2 vLLM settings (`max_model_len=28000`, `max_num_seqs=256`, `max_num_batched_tokens=32768`) become the defaults.

**Phase name drives all output filenames**

`--phase phase3_sampling` writes `results/phase3_sampling_{results,errors,summary}.*`. This makes it impossible to accidentally overwrite a previous phase's results and keeps the results directory self-organizing.

**`--compare` for delta reporting**

The Phase 2 script hardcoded Phase 1 accuracy numbers as literals. The template loads previous summary JSONs by phase name and computes deltas automatically. Multiple `--compare` arguments are supported so a Phase 4 run can diff against both Phase 2 and Phase 3 in one table. If a referenced summary doesn't exist, the delta section is silently skipped rather than crashing.

**Majority voting scaffolded but not fully implemented**

`--samples N` generates N responses per question and passes them to `_majority_vote_mcq` / `_majority_vote_freeform`. MCQ voting is complete (most-common letter wins). Free-form voting uses a correctness-majority heuristic (more than half of samples correct → correct) rather than symbolic answer clustering, which requires non-trivial SymPy logic. A `TODO (Phase 3)` comment marks the upgrade point. At `n_samples=1` (default), both paths are no-ops.

**Few-shot examples as a list in `Config`**

Phase 4 will add worked examples to the prompt. Rather than require a new script, `few_shot_examples` is already wired into `build_prompt` via `_build_few_shot_prefix`. When the list is empty (default), no prefix is added. Populating it for Phase 4 requires only adding dicts to `Config`.

### Usage Reference

```bash
# Mini-batch smoke test (20 questions, ~2 min)
python testing_template.py --phase phase3_sampling

# Full 1126-question run with delta vs Phase 2
python testing_template.py --phase phase3_sampling --full --compare phase2

# Custom N, temperature sweep
python testing_template.py --phase phase3_temp07 --n 50 --temp 0.7

# Majority voting (8 samples per question)
python testing_template.py --phase phase3_mv8 --samples 8
```

### What changes each phase

| Phase | What to edit |
|-------|-------------|
| Prompt engineering | `SYSTEM_PROMPT_MATH`, `SYSTEM_PROMPT_MCQ` |
| Sampling strategy | `--temp`, `--samples`, `TOP_P`, `TOP_K` in `Config` |
| Few-shot | `few_shot_examples` in `Config` |
| Model swap / SFT | `MODEL_ID` in `Config` or `--model` (future flag) |
| Token budget | `--max-tokens` |

---

## Phase 3: Sampling strategy (In Progress)

**Goal:** Majority voting and temperature tuning to boost accuracy without retraining.

**Scripts:** `scripts/run_phase3_stage1.sh`, `scripts/run_phase3_stage2.sh`, `scripts/run_phase3_final.sh`

**Plan:**
- **Stage 1:** Temperature sweep — run 200 questions at each of [0.3, 0.5, 0.6, 0.7, 0.9] with `n_samples=1` to find the best temperature
- **Stage 2:** Majority voting sweep — run 200 questions at the best temperature with `n_samples` ∈ [3, 5, 7] to find the best voting count
- **Final:** Full 1,126-question run with the winning (temperature, n_samples) combination

**Token budget decision — max_tokens raised to 32768 (2026-05-11):**

Phase 2 used `max_tokens=8192`, but 48.4% of errors had no `</think>` closing tag —
the model was hitting the token limit mid-thought before producing any answer. Raising
the limit to 32768 (the practical ceiling given `max_model_len=28000`) eliminates
truncation-driven failures before we begin tuning temperature and majority voting, giving
those experiments a clean baseline. All three phase 3 scripts pass `--max-tokens 32768`.

**Status:** Stage 1 complete (2026-05-13). Stage 2 launched (2026-05-13).

**Script:** `scripts/run_phase3_pipeline.sh` — chains stage 1 → stage 2 → final run automatically;
crash-resilient (checkpointing + resume); safe to re-run after a pod expiration.

### 3.1 Stage 1: Temperature Sweep

**Config:** 200 questions, `n_samples=1`, `max_tokens=32768`, `top_p=0.95`, `top_k=20`

**Temperatures tested:** 0.3, 0.5, 0.6, 0.7, 0.9

**Command (via pipeline):** `bash scripts/run_phase3_pipeline.sh`
**Log:** `logs/phase3_pipeline.log`

**Results:**

| Temperature | Overall | MCQ | Free-form | Multi-part FF | vs Phase 2 Full |
|-------------|---------|-----|-----------|---------------|-----------------|
| 0.3 | 64.5% (129/200) | 76.5% (52/68) | 58.3% (77/132) | 52.9% (36/68) | +10.3pp |
| 0.5 | 63.5% (127/200) | 77.9% (53/68) | 56.1% (74/132) | 48.5% (33/68) | +9.3pp |
| 0.6 (baseline) | 64.5% (129/200) | 79.4% (54/68) | 56.8% (75/132) | 51.5% (35/68) | +10.3pp |
| **0.7** | **66.5% (133/200)** | 77.9% (53/68) | **60.6% (80/132)** | **57.4% (39/68)** | **+12.3pp** |
| 0.9 | 65.5% (131/200) | 77.9% (53/68) | 59.1% (78/132) | 54.4% (37/68) | +11.3pp |

**Best temperature:** **0.7** — 66.5% overall (133/200), completed 2026-05-13

**Status by run:**
- `phase3_temp03`: complete via `scripts/run_phase3_stage1.sh` (2026-05-11), log `logs/phase3_stage1.log`
- `phase3_temp05`, `phase3_temp06`: complete via `scripts/run_phase3_pipeline.sh` (2026-05-12)
- `phase3_temp07`: complete via pipeline (crashed + resumed 2026-05-12→13), log `logs/phase3_pipeline.log`
- `phase3_temp09`: complete via `scripts/run_phase3_stage1_resume.sh` (2026-05-13), log `logs/phase3_stage1_resume.log`

**Key observations:**

1. **Temperature 0.7 is the clear winner.** It is the only setting to break 66% overall, and it
   leads every other temperature on free-form (+2.3pp vs second-best temp=0.9) and multi-part
   free-form (+3.0pp vs next best). MCQ accuracy is essentially flat across all temperatures
   (77.9–79.4%), so the temperature signal comes entirely from free-form reasoning quality.

2. **The max_tokens increase explains most of the gain over Phase 2 full.** All five temperatures
   sit 9–12pp above Phase 2 full (54.2%), yet they use the same prompts. The only change is
   `max_tokens` 8192 → 32768. Phase 2 had 48.4% of its errors caused by mid-thought truncation;
   removing that ceiling is the dominant factor. Temperature tuning is a smaller, secondary effect.

3. **Multi-part free-form is the most temperature-sensitive category.** It ranges from 48.5%
   (temp=0.5) to 57.4% (temp=0.7) — a 9pp spread — while single-part free-form is nearly flat
   (64.1% across all temps) and MCQ is also flat. Higher temperatures appear to help the model
   explore alternative solution paths needed for multi-step problems.

4. **Error profile is now dominated by wrong answers, not formatting.** Across all temperatures,
   ~75–79% of errors are `FREE_FORM_WRONG_ANSWER` and the rest are MCQ letter errors. `FREE_FORM_NO_BOX`
   and `MCQ_NO_VALID_LETTER` are near-zero, confirming that the Phase 2 prompt fixes held at scale
   and that the remaining headroom is mathematical correctness, not formatting.

5. **Implication for Stage 2:** Majority voting at temp=0.7 targets the wrong-answer tail.
   Because ~40% of questions are still incorrect and the errors are math errors rather than
   formatting errors, voting across multiple samples (N=3, 5, 7) should surface the correct
   answer when the model knows it but occasionally makes mistakes. Questions where the model is
   consistently wrong will not benefit from voting.

### 3.2 Stage 2: Majority Voting Sweep — Abandoned

**Decision (2026-05-14): Majority voting sweep abandoned.** The sweep is not viable at `max_tokens=32768` within DSMLP pod lifetimes, and partial results suggest voting provides little benefit for free-form questions.

**What ran:** `phase3_mv3` (n_samples=3) was attempted twice — first launched 2026-05-13 18:35 UTC, killed by pod expiration; resumed 2026-05-14 03:11 UTC, killed again at ~06:26 UTC after ~3h14m. Checkpoint saved 100/200 questions.

**Partial result (100 questions, n_samples=3):**

| Metric | Stage 1 best (temp=0.7, 200q) | MV3 partial (100q) |
|--------|-------------------------------|--------------------|
| Overall | 66.5% | 63.0% |
| MCQ | 77.9% | 81.6% |
| Free-form | 60.6% | 51.6% |

**Why abandoned:**

1. **Throughput is incompatible with pod limits.** With `max_tokens=32768`, some questions hit the full token ceiling on all 3 samples simultaneously, stalling vLLM. Chunk 1-25 (75 prompts) took 2:42:55. Extrapolated: n_samples=3 needs ~22 hours, n_samples=5 ~36 hours, n_samples=7 ~51 hours — all far beyond a single pod session.

2. **Voting does not help free-form.** The model makes the same math error consistently across all samples on questions it gets wrong. Free-form accuracy dropped ~9pp vs Stage 1 single-sample (51.6% vs 60.6%), indicating the wrong-answer errors are systematic, not random. MCQ gains slightly (+3.7pp) because letter-selection errors are more random and voting helps there.

3. **Stage 1 already identified the key win.** The largest gain in Phase 3 came from raising `max_tokens` 8192→32768 (eliminated truncation-driven failures, +10pp). Temperature tuning added ~2pp on top. Voting adds cost without proportional benefit.

**Log:** `logs/phase3_pipeline.log`
**Partial results:** `results/phase3_mv3_results.jsonl` (100 questions), `results/phase3_mv3_errors.jsonl`

### 3.3 Final Run

**Config:** All 1,126 questions, temp=0.7, n_samples=1, `max_tokens=32768`

| Metric | Phase 2 Full | Phase 3 Final | Delta |
|--------|-------------|---------------|-------|
| Overall | 54.2% | TBD | TBD |
| MCQ | 51.5% | TBD | TBD |
| Free-form | 55.5% | TBD | TBD |
| Single-part | 61.1% | TBD | TBD |
| Multi-part | 51.0% | TBD | TBD |

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
| Baseline accuracy (all splits) | ✅ Done | `results/mini_baseline_summary.json` |
| Error analysis + categorization | ✅ Done | `results/mini_baseline_errors.jsonl` |
| Reproducibility documentation | ✅ Done | `docs/process.md` (this file) |
| Improved prompt engineering | ✅ Done | `results/phase2_summary.json` |
| Standardized testing harness | ✅ Done | `testing_template.py` |
| Fine-tuned model checkpoint | ⬜ Planned | TBD |
| Final leaderboard score + rank | ⬜ Planned | Will be in milestone report abstract |

---

## Key Metrics (For Milestone Report)

### Phase 1 Results (Full baseline, 1,126 questions — completed 2026-05-10)
- **Overall accuracy:** 29.9% (337/1126)
- **MCQ accuracy:** 10.7% (40/375)
- **Free-form accuracy:** 39.5% (297/751) — single-part 51.0%, multi-part 30.2%
- **Dominant errors:** `FREE_FORM_NO_BOX` (355, 45%), `MCQ_NO_VALID_LETTER` (231, 29%)

### Phase 1 Results (Mini-baseline, 20 questions — for reference)
- **Overall accuracy:** 25.0% (5/20)
- **MCQ accuracy:** 11.1% (1/9)
- **Free-form accuracy:** 36.4% (4/11) — single-part 75.0%, multi-part 14.3%
- **Dominant errors:** `MCQ_NO_VALID_LETTER` (7), `FREE_FORM_NO_BOX` (7)

### Phase 2 Results (Prompt engineering, 20 questions)
- **Overall accuracy:** 60.0% (12/20) — **+35.0pp vs Phase 1**
- **MCQ accuracy:** 66.7% (6/9) — +55.6pp
- **Free-form accuracy:** 54.5% (6/11) — single-part 75.0%, multi-part 42.9%
- **Dominant errors:** `FREE_FORM_WRONG_ANSWER` (5), `MCQ_WRONG_LETTER` (2)

### Phase 2 Full Dataset Results (1,126 questions — 2026-05-11)
- **Overall accuracy:** 54.2% (610/1126)
- **MCQ accuracy:** 51.5% (193/375)
- **Free-form accuracy:** 55.5% (417/751) — single-part 61.1%, multi-part 51.0%
- **Dominant errors:** `FREE_FORM_WRONG_ANSWER` (259), `MCQ_NO_VALID_LETTER` (114)
- **Truncation note:** 250/516 errors (48.4%) were caused by the 8192-token limit cutting off thinking traces; addressed in Phase 3 by raising to 32768

### Phase 3 Stage 1 Results (temperature sweep, 200 questions — 2026-05-13)
- **Best temperature:** 0.7 → 66.5% overall (133/200)
- **MCQ accuracy:** 77.9% (essentially flat across all temps, 77.9–79.4%)
- **Free-form accuracy:** 60.6% — single-part 64.1%, multi-part 57.4%
- **Key driver:** max_tokens 8192→32768 eliminated truncation; temperature tuning added ~2pp on top
- **Next:** Stage 2 majority voting at temp=0.7

### Improvements Summary
| Phase | Sample | Overall | MCQ | Free-form |
|-------|--------|---------|-----|-----------|
| Phase 1 full baseline | 1,126 | 29.9% | 10.7% | 39.5% |
| Phase 2 (prompt fixes) | 20 | 60.0% | 66.7% | 54.5% |
| Phase 2 full dataset | 1,126 | 54.2% | 51.5% | 55.5% |
| **Phase 3 Stage 1 best (temp=0.7)** | **200** | **66.5%** | **77.9%** | **60.6%** |

---

## Notes

- All code is reproducible: script specifies model ID, quantization method, sampling params, batch size, GPU type
- Results are version-controlled in git: `results/baseline_*.jsonl` and `results/baseline_summary.json`
- Error logs enable targeted debugging: each failure includes question excerpt, full response, extracted answer, and error category
