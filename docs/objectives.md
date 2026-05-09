# Competition Objectives & Plan of Attack

---

## What We're Doing

**CSE 151B Spring 2026 Math Reasoning Competition.**  
Improve the mathematical reasoning capability of **Qwen/Qwen3-4B-Thinking-2507** — and only that model — using model-intrinsic methods. No external APIs or tool-augmented generation at inference time. The leaderboard ranks teams by unified accuracy on a private test set spanning high-school to graduate-level math.

---

## Hard Constraints

- **Model is fixed:** must use `Qwen/Qwen3-4B-Thinking-2507` for final submission. Training (SFT, RL) is allowed but the final inference model must be this one (possibly fine-tuned).
- **No external tools at inference time:** no code interpreters, calculators, API calls, or retrieval. Pure model output only.
- **Training data:** any publicly available data is allowed for fine-tuning.

---

## Allowed Methods

| Category | Examples |
|----------|----------|
| Prompt engineering | Chain-of-thought, few-shot exemplars, self-consistency / majority voting, progressive-hint prompting |
| Supervised fine-tuning | LoRA, QLoRA, full fine-tuning on any public data |
| Reinforcement learning | GRPO, DPO, outcome-based reward modeling |

---

## Dataset

| Split | Location | Has answers? |
|-------|----------|--------------|
| Public (dev) | `data/public.jsonl` | Yes — use for local accuracy measurement |
| Private (test) | `private.jsonl` (downloaded separately) | No — submit predictions |

**Public set breakdown (1,126 questions):**
- 375 MCQ (up to 10 answer choices, answer is a single capital letter)
- 751 Free-form, of which 414 have multi-part answers (one string per `[ANS]` placeholder in the question)

**Record format:**
```json
// Free-form (single answer)
{"id": 4, "question": "...= [ANS]", "answer": ["-512"]}

// Free-form (multi-part — one answer per [ANS])
{"id": 2, "question": "...(a)...[ANS]...(b)...[ANS]", "answer": ["41", "35", "16"]}

// MCQ
{"id": 1, "question": "...", "options": ["...", "..."], "answer": "C"}
```

Questions use `[ANS]` as a placeholder where each answer goes in free-form questions. The model should produce all answers inside `\boxed{}` (comma-separated for multi-part).

---

## Evaluation

**Metric: Unified accuracy** — total correct / total questions. Every question is weighted equally regardless of source or difficulty.

- **MCQ:** extract the letter from the last `\boxed{}` in the response; compare to gold letter (case-insensitive).
- **Free-form:** `Judger.auto_judge()` in `judger.py` handles symbolic equivalence, numeric tolerance (1e-8 relative), interval notation, ordered/unordered lists, True/False, and open-ended string match. **All sub-answers must be correct for a multi-part question to count.**

**Leaderboard:** shows accuracy on ~30% of the private test set during the competition. Final rankings are on the 100% private test set, revealed after the submission deadline. The private set follows the same distribution as the public set. Submissions are unlimited.

---

## Submission Format

The final submission is a **CSV file** (not JSONL) with two columns: `id` and `response`.

```
id,response
0,"Okay, let's try to solve this... [full reasoning trace] ... \boxed{42}"
1,"This is a complex question... \boxed{580, 660, 80}"
```

- `response` must be the **complete raw model output** including all chain-of-thought / thinking tokens — not just the extracted answer.
- Since responses contain commas, newlines, and quotes, the CSV must be properly quoted (standard double-quote escaping: inner `"` → `""`).
- Every `id` in `private.jsonl` must have a row.
- The competition evaluator extracts the final answer from the response trace using the same logic as `judger.py`.

**The current notebook saves JSONL — a submission script that writes a properly escaped CSV from the results is needed.**

---

## Baseline

Model: **Qwen3-4B-Thinking-2507** loaded in INT4 (BitsAndBytes) via HuggingFace Transformers.  
Results written to `results/starter_results.jsonl`.  
**TODO: run the full 1,126-question eval and record baseline accuracy here.**

---

## Plan of Attack

### Phase 1 — Establish a reliable local benchmark (do this first)

1. Run the full notebook on all 1,126 public questions (not just the first 5). Record accuracy broken down by MCQ vs. free-form and by single-part vs. multi-part.
2. Log every incorrect prediction: extracted answer, gold answer, and question type. Categorize failures — wrong format (no `\boxed{}`), wrong math, MCQ confusion, truncated output.
3. Pin this as the baseline. Every subsequent change must be compared against it on the full public set.

### Phase 2 — Low-effort, high-leverage prompt fixes

4. **Multi-part answer formatting.** The system prompt should explicitly say: "If the question has multiple `[ANS]` placeholders, put all answers comma-separated in a single `\boxed{}`, in order." Verify the model does this reliably; wrong ordering costs full-question accuracy.
5. **MCQ prompt tightening.** Verify responses nearly always contain exactly one letter in `\boxed{}`. If the model hedges or includes extra letters, strengthen the instruction or add a one-shot example.
6. **Token budget.** Check whether any responses are cut off before producing `\boxed{}`. If so, increase `MAX_TOKENS` (currently 2,048). Free-form reasoning traces can run long for graduate-level problems.

### Phase 3 — Sampling and self-consistency

7. **Majority voting.** Generate N ≥ 8 responses per question with non-zero temperature, extract the answer from each, take the most common. Self-consistency is one of the strongest inference-time accuracy gains for reasoning models with zero additional training.
8. **Temperature sweep.** Test temperature ∈ {0.3, 0.6, 0.8} on a held-out slice (e.g., 200 questions) to find the optimum before running the full set.
9. **Progressive-hint prompting.** For questions where majority voting produces a tie or low-confidence result, re-prompt with a hint extracted from the model's own previous partial solutions.

### Phase 4 — Prompt engineering

10. **Few-shot examples.** Add 2–3 worked examples in the user turn covering: a single-answer free-form, a multi-part free-form, and an MCQ. Use problems the model is unlikely to have memorized verbatim.
11. **Chain-of-thought steering.** For free-form, add an explicit instruction: "Before boxing your answer, write 'Therefore, the answer is ...'". This makes extraction more reliable and reduces cases where the final `\boxed{}` is missing or ambiguous.
12. **Domain-targeted prompts.** After Phase 1 error analysis, if a specific domain (e.g., calculus, complex analysis) fails disproportionately, add a domain-specific note to the prompt for those question types.

### Phase 5 — Faster inference infrastructure

13. **Switch to vLLM.** The vLLM path (currently commented out in the notebook) is significantly faster for batch inference and supports INT8 quantization. Faster inference directly enables more majority voting samples within the same wall time.
14. **Batch size tuning.** With vLLM, tune `gpu_memory_utilization`, `max_num_seqs`, and `max_num_batched_tokens` to maximize throughput without OOM.

### Phase 6 — Supervised fine-tuning

15. **SFT on public data.** Use the 1,126 public questions as training signal. Hold out ~20% for validation. The model should learn to reliably format answers in `\boxed{}` and handle multi-part questions correctly.
16. **Augment with external public data.** Pull from MATH, GSM8K, AIME, AMC, or other public math datasets. Focus on question types where the baseline model scores lowest (from Phase 1 error analysis).
17. **Generate synthetic solutions.** Use the model to produce step-by-step solutions for public questions, filter with `judger.py` to keep only correct ones, then fine-tune on those traces. This creates high-quality CoT training data at low cost.
18. **LoRA / QLoRA.** Fine-tune adapters only to keep GPU memory manageable. Use 4-bit QLoRA for training, merge adapters into base weights before final inference.

### Phase 7 — Reinforcement learning

19. **GRPO or DPO on public data.** Use `judger.py` as the reward signal (binary: correct answer = +1, wrong = 0). GRPO can be applied directly to the model's own generations; DPO requires preference pairs (correct response preferred over incorrect). RL often improves accuracy beyond what SFT achieves, especially for harder problems.
20. **Outcome-based reward modeling.** Train a reward model on the public set (correct vs. incorrect) and use it during RL to provide denser reward signal than binary correctness alone.

### Phase 8 — Submission

21. Write a submission script that reads the model's JSONL output and writes a properly quoted CSV (`id,response`) compatible with the Kaggle evaluator.
22. Run inference on `private.jsonl` (no ground truth). Submit the CSV.
23. Compare the Kaggle leaderboard score (~30% of private set) against local public-set accuracy. A large gap indicates distribution shift or overfitting.
24. Final rankings use the full private test set after the deadline — hold back the best-performing checkpoint, not just the most recent one.

---

## Priority Order (TL;DR)

| Priority | Action | Expected effort |
|----------|--------|-----------------|
| 1 | Full public eval + error breakdown | 1–2 hours |
| 2 | Multi-part + MCQ prompt fixes | 30 min |
| 3 | Majority voting (N=8, temperature sweep) | 2–3 hours |
| 4 | Few-shot + CoT steering prompt | 1 hour |
| 5 | Switch to vLLM for faster batching | 1–2 hours |
| 6 | Write CSV submission script | 30 min |
| 7 | SFT with LoRA on public + external data | 1–2 days |
| 8 | RL (GRPO/DPO) with judger as reward | 2–3 days |
