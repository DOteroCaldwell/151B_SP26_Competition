# Dataset Quality & External-Data Strategy

**Status:** Draft, 2026-05-13. Analysis + prioritized recommendations.

This document responds to the team's manual inspection of `data/public.jsonl`,
which surfaced errors in some of the gold answers. The private Kaggle test set
is fixed and likely shares the same noise distribution, since both are
presumably authored by the same course staff. We can't fix the private set —
but we can choose how to train and evaluate so that the noise hurts our model
as little as possible.

---

## 1. Situation & motivation

Some fraction of the 1,126 gold answers in `data/public.jsonl` are wrong,
ambiguous, or stored in a format the judger mishandles. Because the private
test set is also locked, every team competes against the same flawed standard,
so the *leaderboard ranking* is fair. What the leaderboard cannot tell us is
whether a given accuracy improvement comes from genuinely better math
reasoning or from better alignment with the (imperfect) answer key.

That creates a tension:
- Optimizing purely for public-set agreement risks **memorizing flawed labels**,
  inflating our local accuracy without improving real reasoning.
- Optimizing purely for "true" math correctness risks **regressing on the
  leaderboard** if our model becomes less aligned with the noisy gold.

The right strategy threads both: take training signal from clean external
sources, keep using the public set as a proxy for the private set, and watch
for divergence between the two scoreboards.

---

## 2. Quantifying the noise (what we know, what we don't)

We have **no formal measurement** of the label-error rate yet — only
impressionistic signals from scanning `results/phase3_temp07_errors.jsonl`.
The error log surfaces at least four distinct failure modes that look like
candidate label/judger issues rather than model errors:

| ID | Gold | Model output | Likely failure mode |
|----|------|-------------|---------------------|
| 16 | `['atan(4.76)', 'pi']` | `1.364, 3.142` | Symbolic ↔ numeric mismatch — numerically equivalent, judger flagged wrong |
| 5  | `['62.78', '335.93', '604.67']` | `62.78, 335.93, 604.67` | Tight tolerance / rounding mismatch — values match to 2–3 decimals |
| 12 | `['380', '315', '13', '310']` | `380, 315, 14, 310` | One-digit divergence on a multi-part item — model error *or* gold ambiguity |
| 1  | gold `F`, model `B` | — | Clean model error (formatting correct, wrong letter) |

Mapped to fixable categories:

1. **Symbolic ↔ numeric mismatch** — judger weakness, not gold error. Fixable
   in `judger.py` by adding a SymPy `nsimplify` / numeric-fallback pass in
   `is_equal`.
2. **Tight numerical tolerance / rounding inconsistency** — judger weakness or
   format mismatch. Fixable in `judger.py` (tolerance knob) or in
   `SYSTEM_PROMPT_MATH` (instruct the model to round to gold's precision).
3. **Genuinely wrong gold labels** — transcription or answer-key bugs. Not
   fixable; must be routed around via training-data choices.
4. **Ambiguous problem statements** — affect gold *and* model equally;
   sometimes resolvable by prompt clarification, often not.

The team's first concrete action should be measuring the actual proportions
(see §6, action 1). Without that, every downstream recommendation is sized
against a guess.

---

## 3. Implications of a fixed, noisy private test set

The competition rules confirm the public and private sets are drawn from the
same distribution:

> "The public and private test sets follow similar distributions in terms of
> difficulty proportions, question domains, and response formats."
> — `docs/kaggle-overview.md:21`

Two consequences follow:

1. **Public-set accuracy remains a useful relative metric.** Since both sets
   share the same noise, ranking our own experiments against the public set is
   still meaningful — a configuration that scores better on public *probably*
   scores better on private.
2. **Public-set accuracy is a biased absolute metric.** Some fraction of our
   "errors" are actually correct answers being marked wrong, and some fraction
   of our "successes" may be lucky matches on broken gold. Reporting our
   absolute number in the milestone report without acknowledging this overstates
   the model's true ability.

Practical implication: **maintain two scoreboards** — leaderboard-style accuracy
on `public.jsonl`, and "honest" accuracy on a clean external held-out set
(see §6, action 6).

---

## 4. Best practices for tuning on noisy data

These are general principles from the noisy-labels literature, filtered for
what's realistic on a 4B model in a 6-week course budget.

- **Train on cleaner data, evaluate on noisy.** The most defensible position is
  to use `public.jsonl` strictly as a dev/test set, never as a training target,
  and to take training signal from external corpora whose answer keys are
  heavily QA'd. This isolates the noise to evaluation, where it's a measurement
  problem rather than a generalization problem.
- **Confidence-weighted / self-consistency filtering.** If SFT on
  `public.jsonl`-style data *is* needed, drop or down-weight examples where:
  - majority-vote across N samples from the current model disagrees with gold
    (likely label noise, not learnable signal), or
  - the judger flags edge cases of the symbolic↔numeric variety (judger
    weakness, not a math error to learn from).
- **Use a stronger teacher to pseudo-label.** Running a top-tier reasoning
  model (DeepSeek-R1, Qwen3-235B, a GPT-4-class API) on `public.jsonl`
  *offline* and comparing its answers to gold is a cheap way to surface
  candidate-erroneous items for manual review. This is **data curation, not
  inference-time tool use**, so it stays within the rules:
  > "External model calls, API access, and tool-augmented generation
  > (e.g., code interpreters or calculators) are not permitted at inference time."
  > — `docs/kaggle-overview.md:16`
- **Loss-function choices.** Generalized cross-entropy, symmetric
  cross-entropy, and bootstrapped CE are noise-robust SFT alternatives to
  vanilla CE. On a 4B+LoRA setup the marginal benefit is modest and the
  implementation cost is real — recommend deferring unless the audit (§6.1)
  shows >5% label error.
- **RL is structurally more robust to label noise than SFT.** With a binary
  reward, a wrong gold label means *missed reward*, not an *actively wrong
  training target*. SFT under noisy labels actively teaches the model to be
  wrong; RL just under-rewards the right answers. This may argue for
  re-ordering Phase 6 (SFT) and Phase 7 (RL) if the audit reveals heavy
  noise — see §6, action 5.

---

## 5. External datasets to bring in

The competition rules are unambiguous about external training data:

> "**Training data:** any publicly available data is allowed for fine-tuning."
> — `docs/objectives.md:16`

> "Supervised fine-tuning — LoRA, QLoRA, or full fine-tuning on **any
> publicly available training data**" — `docs/kaggle-overview.md:14`

`docs/objectives.md:129` already names MATH, GSM8K, AIME, and AMC as
candidates. The table below operationalizes that plan.

| Dataset | Size | Type | Quality | Why useful here |
|---|---|---|---|---|
| **MATH** (Hendrycks) | ~12.5k | Free-form, competition | High (manually curated) | Multi-step problems with worked solutions; addresses the multi-part free-form weakness (Phase 2 full: 51.0%) |
| **GSM8K** | ~8.5k | Grade-school word problems | High | Arithmetic + problem-setup signal; `utils.py` already imports `GSM8K_ANS_PREFIX`, so format-handling groundwork is partly in place |
| **AIME archive** (1983–) | ~1.1k | Integer-answer competition | Very high | Mirrors the hardest free-form items in the public set |
| **AMC 10/12** | ~5k | MCQ | High | Directly addresses MCQ accuracy (Phase 2 full: 51.5%; still our weakest free-form-vs-MCQ comparison) |
| **NuminaMath / OpenMathInstruct-2** | 200k–860k | Mixed, model-generated solutions | Variable | Bulk SFT volume; must pair with verifier filtering since solutions are not guaranteed correct |

A few notes the team should weigh when integrating any of these:

- **Format-match first.** For SFT, prefer datasets whose final-answer format
  can be reshaped into the `SYSTEM_PROMPT_MATH` convention (boxed answer,
  comma-separated multi-part). Reformatting preserves the Phase 2 prompt fixes
  rather than fighting them.
- **Continual pretraining is more format-tolerant** than SFT. If we go that
  route, raw solution traces from NuminaMath / OpenMathInstruct are usable
  without heavy reformatting.
- **All five sources are explicitly within the competition's "publicly
  available data" allowance.** Nothing here requires special permission.

---

## 6. Recommended paths forward (prioritized)

Ordered roughly by `(value / effort)`. Items 1 and 2 are small and unlock
better decisions on items 3–6.

### Action 1 — Audit pass *(highest priority, ~1–2 days)*

Take a stratified sample of 100–150 entries from
`results/phase3_temp07_errors.jsonl` (the freshest, best-performing error log):

- 50 `MCQ_WRONG_LETTER`
- 50 `FREE_FORM_WRONG_ANSWER` (single-part)
- 25 multi-part `FREE_FORM_WRONG_ANSWER`
- 25 random across categories

For each, human-label as one of:
- (a) true model error
- (b) judger false positive (symbolic/numeric mismatch, tolerance issue)
- (c) gold-label error (transcription, wrong answer key)
- (d) ambiguous problem statement

Write the breakdown and notable examples into a separate file
`docs/label-audit.md`. This single document drives every decision below:
the value of judger hardening (§6.2), the safety of SFT on public data (§6.4),
and whether to reorder SFT vs. RL (§6.5).

**Owner & timeline:** TBD — one team member, ~10 hours total.

### Action 2 — Judger hardening *(small effort, immediate gain)*

If the audit shows judger false positives are ≥5% of errors, patch `judger.py`:

- In `is_equal`, add a numeric-fallback path: if both inputs parse as SymPy
  expressions but symbolic equivalence fails, evaluate both numerically and
  compare with the existing 1e-8 tolerance.
- Expose a `--tolerance` flag in `testing_template.py` so we can sweep
  rounding-mismatch sensitivity without code changes.
- Re-run the Phase 3 best config (`phase3_temp07`); any free accuracy gain
  here flows directly to the leaderboard.

Worth considering even at <5%: the fix is local and small.

### Action 3 — External-data SFT *(largest single investment, Phase 6 in `objectives.md`)*

Start with **MATH + AIME + AMC** (~20k problems combined, all competition-aligned).

- **Reformat** each dataset to match `SYSTEM_PROMPT_MATH`: final answer in
  `\boxed{}`, comma-separated for multi-part. Plan ~1 day per source.
- **Filter for learning signal.** Drop problems Qwen3-4B already solves
  reliably (no gradient signal) and problems it scores <10% across 8 samples
  (likely beyond a 4B model's reach — these waste capacity).
- **LoRA r=16, 1–2 epochs.** Evaluate on `public.jsonl` *and* a held-out 10%
  of MATH on every checkpoint.
- **Do not include `public.jsonl` itself in SFT training** until the audit
  results say it's safe. If audit shows ≥10% label error, public should
  stay test-only for SFT (it's still fine as RL signal — see §6.5).

### Action 4 — Self-distilled labels for borderline public examples *(optional, Phase 6.5)*

For items where an 8-sample majority vote from our current best model
agrees with itself but disagrees with the gold, **do not train on the gold**.
Two options:

- Drop those examples from any SFT mix entirely; or
- Use the model's own majority answer as the SFT target (silver-label
  self-distillation).

This recovers signal from problems with broken gold without injecting that
brokenness into the trained weights. Run this only after Action 3 is working
and after the audit informs the agreement threshold.

### Action 5 — Possibly reorder RL before more SFT *(Phase 7 decision)*

If the audit shows >10% label noise on the items we'd train on, consider
running GRPO/DPO with `judger.auto_judge()` as binary reward *before* doing
more SFT. RL under noisy labels under-rewards correct answers (recoverable);
SFT under noisy labels actively trains wrong outputs (not recoverable without
re-training).

If the audit comes back clean (<5% noise), keep the current ordering: SFT
first, RL second, as in `docs/objectives.md`.

### Action 6 — "Honest" scoreboard alongside the leaderboard scoreboard

Build `data/honest_test.jsonl` — a 200-problem held-out slice of MATH and
AIME, sampled to roughly match the public-set difficulty mix, **never seen
during training or hyperparameter tuning**. Every experiment in
`testing_template.py` should report both:

- accuracy on `data/public.jsonl` (the leaderboard proxy), and
- accuracy on `data/honest_test.jsonl` (the noise-free truth).

A divergence between the two — leaderboard up, honest down — is the
signature of overfitting to label noise. Catching that signal is worth the
~1 day of setup.

---

## 7. Risks & tradeoffs

- **Over-curating away from the public distribution.** External math sets skew
  competition-style; the public set includes physics-flavored multi-part
  problems that AMC/AIME don't cover. Don't drop public-flavored content
  entirely; mix sources.
- **Reformatting cost.** Every external dataset has its own answer format.
  Plan ~1 day of preprocessing per source. Build a small `scripts/reformat_*.py`
  per source and check the output into `data/external/`.
- **Compute budget.** 24 GB VRAM (RTX PRO 6000 Blackwell Max-Q with MIG) is
  tight for inference + LoRA on a 4B model. Phase 6 SFT will probably need
  gradient accumulation or a separate larger pod (see `docs/dsmlp-long-jobs.md`).
- **Schedule.** We're at Phase 3 Stage 2 on 2026-05-13. The audit (Action 1)
  should run *now* because it's small and informs everything else. Action 3
  is the largest single investment and should not start before the audit
  finishes.

---

## 8. Open questions for the team

- **Audit owner.** Who runs Action 1? Suggest one volunteer, ~10 hours over
  two days, output is `docs/label-audit.md`.
- **Instructor signal.** Has anyone asked the staff whether they're aware of
  the gold-answer issues? If a corrected `public_v2.jsonl` is in the
  pipeline, much of this document becomes moot.
- **Piazza disclosure.** Should we file a public issue about the
  symbolic↔numeric judger weakness? Other teams hitting the same false
  positives benefits everyone, but our gain from fixing it is competitive —
  trade-off worth discussing before posting.

---

## Cross-references

- `docs/objectives.md:16, 129` — competition rules on training data; Phase 6 SFT plan
- `docs/kaggle-overview.md:11–21` — distribution claim, allowed methods, inference-time tool ban
- `docs/process.md` §1.2.1, §2.5, §3.1 — Phase 1/2/3 results and error breakdowns
- `judger.py` — `is_equal`, `auto_judge`, type codes (`NV`, `EX`, `EQ`, `INT`, `MCS`, `MCM`, `OL`, `UOL`, `TF`, `OE`)
- `testing_template.py` — `Config.data_path`, `--samples`, `--temp`, `--max-tokens`
- `utils.py` — `GSM8K_ANS_PREFIX` (existing precedent for external-dataset format handling)
- `results/phase3_temp07_errors.jsonl` — concrete examples cited in §2
