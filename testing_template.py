#!/usr/bin/env python3
"""
testing_template.py — Standardized testing harness for all phases.

Usage:
  python testing_template.py --phase <name>           # mini-batch (20 q, fast)
  python testing_template.py --phase <name> --full    # full 1126-question run
  python testing_template.py --phase <name> --n 50    # custom N
  python testing_template.py --phase <name> --compare phase2  # delta report

Knobs to turn each phase:
  - SYSTEM_PROMPT_MATH / SYSTEM_PROMPT_MCQ  (Phase 2, 4)
  - TEMPERATURE, TOP_P, TOP_K               (Phase 3)
  - N_SAMPLES (majority voting)             (Phase 3)
  - MODEL_ID                                (Phase 5, 6)
  - MAX_TOKENS                              (increase if thinking is truncated)
  - FEW_SHOT_EXAMPLES                       (Phase 4)

Common pitfalls this harness avoids:
  - MAX_TOKENS=2048 truncating thinking traces (default 8192)
  - VLLM_USE_DEEP_GEMM crashing on Blackwell (always disabled)
  - Forgetting to set CUDA_VISIBLE_DEVICES before vLLM import
  - Results overwriting previous phase outputs (phase name is baked in)
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── CUDA env must be set before vLLM import ───────────────────────────────────
# Parse GPU_ID early so the env var is set before any torch/vllm code loads.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", default="0")
_gpu_args, _ = _pre.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_args.gpu
os.environ["VLLM_USE_DEEP_GEMM"] = "0"   # required on Blackwell (RTX PRO 6000)

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

sys.path.insert(0, ".")
from judger import Judger


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class Config:
    # ── Identity ──────────────────────────────────────────────────────────────
    phase: str = "phase_template"       # drives all output file names
    compare_to: list[str] = field(default_factory=list)  # phase names to diff against

    # ── Data ──────────────────────────────────────────────────────────────────
    data_path: str = "data/public.jsonl"
    n_questions: int = 20               # mini-batch default; --full overrides to all
    results_dir: Path = Path("results")

    # ── Model ─────────────────────────────────────────────────────────────────
    model_id: str = "Qwen/Qwen3-4B-Thinking-2507"
    gpu_id: str = "0"                   # set by --gpu flag

    # vLLM settings (tuned in Phase 2; adjust carefully)
    gpu_memory_utilization: float = 0.50
    max_model_len: int = 28000
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 32768

    # ── Sampling ──────────────────────────────────────────────────────────────
    max_tokens: int = 8192              # 2048 truncates thinking; keep ≥8192
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    repetition_penalty: float = 1.0

    # ── Majority voting (Phase 3+) ────────────────────────────────────────────
    # Set n_samples > 1 to generate multiple responses per question.
    # Scoring then uses majority vote. See _majority_vote() below.
    n_samples: int = 1

    # ── Few-shot examples (Phase 4+) ─────────────────────────────────────────
    # List of {"question": ..., "solution": ..., "answer": ...} dicts.
    # Prepended to the user turn when non-empty. See _build_few_shot_prefix().
    few_shot_examples: list[dict] = field(default_factory=list)


# ── Prompts ───────────────────────────────────────────────────────────────────
# These are the Phase 2 prompts (best known so far). Edit here to iterate.
# Document what changed and why in process.md after each run.

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "You MUST place your final answer inside \\boxed{}. "
    "For problems with multiple [ANS] placeholders, list all answers comma-separated "
    "inside a single \\boxed{} in the order they appear, e.g. \\boxed{3, 7, -2}. "
    "Always end your response with \\boxed{your answer here}. Never omit the \\boxed{} wrapper."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Think through it carefully, then end your response with the letter of your chosen option "
    "inside \\boxed{}. You MUST end with \\boxed{X} where X is exactly one capital letter. "
    "Example final line: \\boxed{C}"
)


# ── Prompt construction ───────────────────────────────────────────────────────

def _build_few_shot_prefix(examples: list[dict]) -> str:
    """Format few-shot examples as solved worked problems prepended to the question."""
    if not examples:
        return ""
    parts = []
    for ex in examples:
        parts.append(f"Example:\nQ: {ex['question']}\nA: {ex['solution']}\nAnswer: \\boxed{{{ex['answer']}}}")
    return "\n\n".join(parts) + "\n\nNow solve the following:\n"


def build_prompt(question: str, options: Optional[list], cfg: Config) -> tuple[str, str]:
    prefix = _build_few_shot_prefix(cfg.few_shot_examples)
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{prefix}{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, f"{prefix}{question}"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(cfg: Config) -> list[dict]:
    all_data = [json.loads(line) for line in open(cfg.data_path)]
    data = all_data if cfg.n_questions <= 0 else all_data[:cfg.n_questions]
    n_mcq  = sum(bool(d.get("options")) for d in data)
    n_free = len(data) - n_mcq
    print(f"[{cfg.phase}] Loaded {len(data)} questions ({n_mcq} MCQ, {n_free} free-form)")
    return data


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(cfg: Config) -> tuple[LLM, AutoTokenizer, SamplingParams]:
    print(f"[{cfg.phase}] Loading model {cfg.model_id} (vLLM, INT8 bitsandbytes)...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=cfg.model_id,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=False,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_model_len=cfg.max_model_len,
        trust_remote_code=True,
        max_num_seqs=cfg.max_num_seqs,
        max_num_batched_tokens=cfg.max_num_batched_tokens,
    )

    sampling_params = SamplingParams(
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=cfg.repetition_penalty,
        n=cfg.n_samples,
    )
    print(f"  Model loaded. max_tokens={cfg.max_tokens}, temp={cfg.temperature}, "
          f"n_samples={cfg.n_samples}")
    return llm, tokenizer, sampling_params


# ── Generation ────────────────────────────────────────────────────────────────

def generate(data: list[dict], llm: LLM, tokenizer: AutoTokenizer,
             sampling_params: SamplingParams, cfg: Config) -> dict[int, list[str]]:
    """Returns {item_id: [response_1, ..., response_n_samples]}."""
    print(f"[{cfg.phase}] Building prompts and generating responses...")
    prompts = []
    for item in data:
        sys_p, usr_p = build_prompt(item["question"], item.get("options"), cfg)
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "system", "content": sys_p},
             {"role": "user",   "content": usr_p}],
            tokenize=False,
            add_generation_prompt=True,
        ))

    outputs = llm.generate(prompts, sampling_params=sampling_params)
    responses = {
        data[i]["id"]: [o.text.strip() for o in out.outputs]
        for i, out in enumerate(outputs)
    }
    print(f"  Generated responses for {len(responses)} questions "
          f"({cfg.n_samples} sample(s) each)")
    return responses


# ── Majority voting ───────────────────────────────────────────────────────────

def _majority_vote_mcq(candidates: list[str]) -> str:
    """Return the most common letter across candidate responses."""
    counts: dict[str, int] = {}
    for resp in candidates:
        letter = _extract_letter(resp)
        if letter:
            counts[letter] = counts.get(letter, 0) + 1
    return max(counts, key=counts.get) if counts else ""


def _majority_vote_freeform(candidates: list[str], judger: Judger,
                             gold_list: list) -> tuple[bool, Optional[str]]:
    """
    Return (correct, best_response) by majority vote over extracted answers.

    Strategy: pick the response whose extracted answer matches the most other
    responses (symbolic equality). Falls back to the first response if no
    consensus can be formed. For n_samples=1 this is a no-op.
    """
    if len(candidates) == 1:
        resp = candidates[0]
        try:
            correct = judger.auto_judge(pred=resp, gold=gold_list,
                                        options=[[]] * len(gold_list))
        except Exception:
            correct = False
        try:
            extracted = judger.extract_ans(resp)
        except Exception:
            extracted = None
        return correct, extracted

    # Score each candidate independently, then take majority on correctness.
    # TODO (Phase 3): implement symbolic clustering for non-binary aggregation.
    scores = []
    for resp in candidates:
        try:
            ok = judger.auto_judge(pred=resp, gold=gold_list,
                                   options=[[]] * len(gold_list))
        except Exception:
            ok = False
        scores.append(ok)

    correct = sum(scores) > len(scores) / 2  # strict majority
    best_resp = candidates[scores.index(True)] if any(scores) else candidates[0]
    try:
        extracted = judger.extract_ans(best_resp)
    except Exception:
        extracted = None
    return correct, extracted


# ── Scoring ───────────────────────────────────────────────────────────────────

def _extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def _categorize_error_mcq(response: str) -> str:
    letter = _extract_letter(response)
    if not letter or letter not in "ABCDEFGHIJ":
        return "MCQ_NO_VALID_LETTER"
    return "MCQ_WRONG_LETTER"


def _categorize_error_freeform(response: str, extracted) -> str:
    if not response.strip():
        return "FREE_FORM_EMPTY"
    if "\\boxed{" not in response:
        return "FREE_FORM_NO_BOX"
    if extracted is None:
        return "FREE_FORM_EXTRACTION_FAILED"
    return "FREE_FORM_WRONG_ANSWER"


def score(data: list[dict], responses: dict[int, list[str]],
          cfg: Config) -> tuple[list[dict], list[dict]]:
    judger = Judger(strict_extract=False)
    results, errors = [], []

    for item in tqdm(data, desc="Scoring"):
        item_id  = item["id"]
        is_mcq   = bool(item.get("options"))
        gold     = item["answer"]
        candidates = responses[item_id]

        if is_mcq:
            if cfg.n_samples > 1:
                extracted = _majority_vote_mcq(candidates)
            else:
                extracted = _extract_letter(candidates[0])
            correct = extracted == str(gold).strip().upper()
            best_response = candidates[0]   # for logging; majority is in extracted
        else:
            gold_list = gold if isinstance(gold, list) else [gold]
            correct, extracted = _majority_vote_freeform(candidates, judger, gold_list)
            best_response = candidates[0]

        results.append({
            "id":       item_id,
            "question": item["question"],
            "is_mcq":   is_mcq,
            "gold":     gold,
            "response": best_response,
            "correct":  correct,
        })

        if not correct:
            if is_mcq:
                cat = _categorize_error_mcq(best_response)
            else:
                cat = _categorize_error_freeform(best_response, extracted)
            errors.append({
                "id":             item_id,
                "is_mcq":         is_mcq,
                "gold":           gold,
                "extracted":      extracted if not is_mcq else _extract_letter(best_response),
                "question":       item["question"][:200],
                "response":       best_response[:400],
                "error_category": cat,
            })

    return results, errors


# ── Summary computation ───────────────────────────────────────────────────────

def compute_summary(results: list[dict], errors: list[dict], cfg: Config) -> dict:
    def acc(subset):
        return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0

    mcq_r    = [r for r in results if r["is_mcq"]]
    free_r   = [r for r in results if not r["is_mcq"]]
    single_r = [r for r in free_r if not (isinstance(r["gold"], list) and len(r["gold"]) > 1)]
    multi_r  = [r for r in free_r if isinstance(r["gold"], list) and len(r["gold"]) > 1]

    err_breakdown: dict[str, int] = {}
    for e in errors:
        err_breakdown[e["error_category"]] = err_breakdown.get(e["error_category"], 0) + 1

    return {
        "timestamp":                      datetime.now().isoformat(),
        "phase":                          cfg.phase,
        "model_id":                       cfg.model_id,
        "sample_size":                    len(results),
        "max_tokens":                     cfg.max_tokens,
        "temperature":                    cfg.temperature,
        "top_p":                          cfg.top_p,
        "top_k":                          cfg.top_k,
        "n_samples":                      cfg.n_samples,
        "total_correct":                  sum(r["correct"] for r in results),
        "overall_accuracy":               acc(results),
        "mcq_accuracy":                   acc(mcq_r),
        "mcq_count":                      len(mcq_r),
        "free_form_accuracy":             acc(free_r),
        "free_form_count":                len(free_r),
        "single_part_free_form_accuracy": acc(single_r),
        "single_part_free_form_count":    len(single_r),
        "multi_part_free_form_accuracy":  acc(multi_r),
        "multi_part_free_form_count":     len(multi_r),
        "error_count":                    len(errors),
        "error_breakdown":                err_breakdown,
    }


# ── Saving ────────────────────────────────────────────────────────────────────

def save_results(results: list[dict], errors: list[dict],
                 summary: dict, cfg: Config) -> None:
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    prefix = cfg.results_dir / cfg.phase

    with open(f"{prefix}_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(f"{prefix}_errors.jsonl", "w") as f:
        for e in errors:
            f.write(json.dumps(e) + "\n")
    with open(f"{prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {cfg.results_dir}/{cfg.phase}_{{results,errors,summary}}.*")


# ── Printing ──────────────────────────────────────────────────────────────────

_ACCURACY_KEYS = [
    ("overall_accuracy",               "Overall"),
    ("mcq_accuracy",                   "  MCQ"),
    ("free_form_accuracy",             "  Free-form (total)"),
    ("single_part_free_form_accuracy", "    Single-part"),
    ("multi_part_free_form_accuracy",  "    Multi-part"),
]


def print_results(summary: dict, cfg: Config) -> None:
    s = summary
    width = 70
    print("\n" + "=" * width)
    print(f"RESULTS — {cfg.phase.upper()}  "
          f"({s['sample_size']} questions, max_tokens={s['max_tokens']}, "
          f"temp={s['temperature']}, n_samples={s['n_samples']})")
    print("=" * width)
    print(f"{'Overall':<28}: {s['total_correct']:3d} / {s['sample_size']:4d}  "
          f"({s['overall_accuracy']:.1f}%)")
    print(f"{'  MCQ':<28}: "
          f"{int(s['mcq_accuracy']/100 * s['mcq_count']):3d} / {s['mcq_count']:4d}  "
          f"({s['mcq_accuracy']:.1f}%)")
    print(f"{'  Free-form (total)':<28}: "
          f"{int(s['free_form_accuracy']/100 * s['free_form_count']):3d} / {s['free_form_count']:4d}  "
          f"({s['free_form_accuracy']:.1f}%)")
    print(f"{'    Single-part':<28}: "
          f"{int(s['single_part_free_form_accuracy']/100 * s['single_part_free_form_count']):3d} / "
          f"{s['single_part_free_form_count']:4d}  "
          f"({s['single_part_free_form_accuracy']:.1f}%)")
    print(f"{'    Multi-part':<28}: "
          f"{int(s['multi_part_free_form_accuracy']/100 * s['multi_part_free_form_count']):3d} / "
          f"{s['multi_part_free_form_count']:4d}  "
          f"({s['multi_part_free_form_accuracy']:.1f}%)")
    print()
    print(f"Total errors: {s['error_count']}")
    for cat, count in sorted(s["error_breakdown"].items(), key=lambda x: -x[1]):
        print(f"  {cat:<40}: {count}")

    # Delta report vs previous phase(s)
    for compare_phase in cfg.compare_to:
        prev_path = cfg.results_dir / f"{compare_phase}_summary.json"
        if not prev_path.exists():
            print(f"\n[compare] {prev_path} not found — skipping delta vs {compare_phase}")
            continue
        prev = json.load(open(prev_path))
        print(f"\n── vs {compare_phase} ({'full' if prev['sample_size'] > 100 else str(prev['sample_size'])+' q'}) ──")
        for key, label in _ACCURACY_KEYS:
            if key not in prev or key not in s:
                continue
            p_val, c_val = prev[key], s[key]
            delta = c_val - p_val
            sign = "+" if delta >= 0 else ""
            print(f"  {label:<34}: {p_val:.1f}% → {c_val:.1f}%  ({sign}{delta:.1f}pp)")

    print("=" * width)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CSE 151B testing harness")
    parser.add_argument("--phase",   required=True,
                        help="Phase identifier; drives output file names (e.g. phase3_sampling)")
    parser.add_argument("--n",       type=int, default=20,
                        help="Number of questions for mini-batch test (default: 20)")
    parser.add_argument("--full",    action="store_true",
                        help="Run all questions (overrides --n)")
    parser.add_argument("--compare", nargs="*", default=[],
                        help="Phase name(s) to compare against in delta report")
    parser.add_argument("--gpu",     default="0",
                        help="CUDA device index (default: 0; already applied at import)")
    parser.add_argument("--samples", type=int, default=1,
                        help="Responses per question for majority voting (default: 1)")
    parser.add_argument("--temp",    type=float, default=0.6,
                        help="Sampling temperature (default: 0.6)")
    parser.add_argument("--max-tokens", type=int, default=8192, dest="max_tokens",
                        help="Max new tokens per response (default: 8192)")
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    cfg = Config(
        phase      = args.phase,
        compare_to = args.compare or [],
        n_questions= 0 if args.full else args.n,   # 0 = all questions
        gpu_id     = args.gpu,
        n_samples  = args.samples,
        temperature= args.temp,
        max_tokens = args.max_tokens,
    )

    data = load_data(cfg)
    llm, tokenizer, sampling_params = load_model(cfg)
    responses = generate(data, llm, tokenizer, sampling_params, cfg)
    results, errors = score(data, responses, cfg)
    summary = compute_summary(results, errors, cfg)
    save_results(results, errors, summary, cfg)
    print_results(summary, cfg)


if __name__ == "__main__":
    main()
