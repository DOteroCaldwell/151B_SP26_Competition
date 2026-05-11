#!/usr/bin/env python3
"""
Phase 1: Establish baseline accuracy on full public dataset.

Runs inference on all 1,126 public questions, scores them, and logs errors for analysis.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

# Configuration
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID = "0"
DATA_PATH = "data/public.jsonl"
RESULTS_DIR = Path("results")
BASELINE_RESULTS = RESULTS_DIR / "baseline_results.jsonl"
BASELINE_ERRORS = RESULTS_DIR / "baseline_errors.jsonl"
BASELINE_SUMMARY = RESULTS_DIR / "baseline_summary.json"

# Inference hyperparameters
MAX_TOKENS = 2048
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ["VLLM_USE_DEEP_GEMM"] = "0"

# ─── Prompts ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)


def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a question."""
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question


# ─── Load data ────────────────────────────────────────────────────────────────
print("[Phase 1] Loading data...")
data = [json.loads(line) for line in open(DATA_PATH)]
n_mcq = sum(bool(d.get("options")) for d in data)
n_free = sum(not d.get("options") for d in data)
print(f"Loaded {len(data)} questions ({n_mcq} MCQ, {n_free} free-form)")

# ─── Load model ────────────────────────────────────────────────────────────────
print("[Phase 1] Loading model (vLLM, INT8 bitsandbytes)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

llm = LLM(
    model=MODEL_ID,
    quantization="bitsandbytes",
    load_format="bitsandbytes",
    enable_prefix_caching=False,
    gpu_memory_utilization=0.50,
    max_model_len=16384,
    trust_remote_code=True,
    max_num_seqs=4,
    max_num_batched_tokens=8192,
)

sampling_params = SamplingParams(
    max_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    repetition_penalty=1.0,
)
print("Model loaded.")

# ─── Generate responses ────────────────────────────────────────────────────────
print("[Phase 1] Building prompts and generating responses...")
prompts = []
for item in data:
    system, user = build_prompt(item["question"], item.get("options"))
    prompts.append(tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    ))

outputs = llm.generate(prompts, sampling_params=sampling_params)
responses = {data[i]["id"]: out.outputs[0].text.strip() for i, out in enumerate(outputs)}
print(f"Generated {len(responses)} responses")

# ─── Scoring ──────────────────────────────────────────────────────────────────
print("[Phase 1] Scoring responses...")

# Load judger
sys.path.insert(0, ".")
from judger import Judger

judger = Judger(strict_extract=False)


def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def score_mcq(response: str, gold_letter: str) -> bool:
    return extract_letter(response) == gold_letter.strip().upper()


results = []
errors = []

for item in tqdm(data, desc="Scoring"):
    item_id = item["id"]
    response = responses[item_id]
    is_mcq = bool(item.get("options"))
    gold = item["answer"]

    # Score
    if is_mcq:
        correct = score_mcq(response, str(gold))
        extracted = extract_letter(response)
    else:
        gold_list = gold if isinstance(gold, list) else [gold]
        try:
            correct = judger.auto_judge(
                pred=response,
                gold=gold_list,
                options=[[]] * len(gold_list),
            )
        except Exception as e:
            correct = False

        try:
            extracted = judger.extract_ans(response)
        except Exception:
            extracted = None

    result = {
        "id": item_id,
        "is_mcq": is_mcq,
        "gold": gold,
        "response": response,
        "correct": correct,
    }
    results.append(result)

    # Log errors
    if not correct:
        error = {
            "id": item_id,
            "is_mcq": is_mcq,
            "gold": gold,
            "extracted": extracted if not is_mcq else extract_letter(response),
            "question": item["question"][:200],  # First 200 chars
            "response": response[:400],  # First 400 chars
            "error_category": "UNKNOWN",
        }

        # Categorize error
        if is_mcq:
            if not extracted or extracted not in "ABCDEFGHIJ":
                error["error_category"] = "MCQ_NO_VALID_LETTER"
            else:
                error["error_category"] = "MCQ_WRONG_LETTER"
        else:
            if not response.strip():
                error["error_category"] = "FREE_FORM_EMPTY"
            elif "\\boxed{" not in response:
                error["error_category"] = "FREE_FORM_NO_BOX"
            elif extracted is None:
                error["error_category"] = "FREE_FORM_EXTRACTION_FAILED"
            else:
                error["error_category"] = "FREE_FORM_WRONG_ANSWER"

        errors.append(error)

# ─── Compute statistics ───────────────────────────────────────────────────────
mcq_results = [r for r in results if r["is_mcq"]]
free_results = [r for r in results if not r["is_mcq"]]

# Count multi-part free-form
multi_part_free = [r for r in free_results if isinstance(r["gold"], list) and len(r["gold"]) > 1]
single_part_free = [r for r in free_results if not (isinstance(r["gold"], list) and len(r["gold"]) > 1)]

def accuracy(subset):
    if not subset:
        return 0.0
    return sum(1 for r in subset if r["correct"]) / len(subset) * 100

summary = {
    "timestamp": datetime.now().isoformat(),
    "total_questions": len(data),
    "total_correct": sum(1 for r in results if r["correct"]),
    "overall_accuracy": accuracy(results),
    "mcq_accuracy": accuracy(mcq_results),
    "mcq_count": len(mcq_results),
    "free_form_accuracy": accuracy(free_results),
    "free_form_count": len(free_results),
    "single_part_free_form_accuracy": accuracy(single_part_free),
    "single_part_free_form_count": len(single_part_free),
    "multi_part_free_form_accuracy": accuracy(multi_part_free),
    "multi_part_free_form_count": len(multi_part_free),
    "error_count": len(errors),
    "error_breakdown": {},
}

# Error breakdown by category
for error in errors:
    cat = error["error_category"]
    if cat not in summary["error_breakdown"]:
        summary["error_breakdown"][cat] = 0
    summary["error_breakdown"][cat] += 1

# ─── Save results ──────────────────────────────────────────────────────────────
print("[Phase 1] Saving results...")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Save detailed results
with open(BASELINE_RESULTS, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

# Save errors with analysis
with open(BASELINE_ERRORS, "w") as f:
    for e in errors:
        f.write(json.dumps(e) + "\n")

# Save summary
with open(BASELINE_SUMMARY, "w") as f:
    json.dump(summary, f, indent=2)

# ─── Print summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 1: BASELINE EVALUATION RESULTS")
print("=" * 70)
print(f"Overall accuracy        : {summary['total_correct']:4d} / {summary['total_questions']:4d}  ({summary['overall_accuracy']:.2f}%)")
print(f"  MCQ                   : {sum(1 for r in mcq_results if r['correct']):4d} / {len(mcq_results):4d}  ({summary['mcq_accuracy']:.2f}%)")
print(f"  Free-form (total)     : {sum(1 for r in free_results if r['correct']):4d} / {len(free_results):4d}  ({summary['free_form_accuracy']:.2f}%)")
print(f"    - Single-part       : {sum(1 for r in single_part_free if r['correct']):4d} / {len(single_part_free):4d}  ({summary['single_part_free_form_accuracy']:.2f}%)")
print(f"    - Multi-part        : {sum(1 for r in multi_part_free if r['correct']):4d} / {len(multi_part_free):4d}  ({summary['multi_part_free_form_accuracy']:.2f}%)")
print()
print(f"Total errors            : {len(errors)}")
print(f"Error breakdown:")
for cat, count in sorted(summary["error_breakdown"].items(), key=lambda x: -x[1]):
    print(f"  {cat:30s}: {count:4d}")
print("=" * 70)
print(f"\nResults saved to {RESULTS_DIR}/")
print(f"  - Detailed results: {BASELINE_RESULTS}")
print(f"  - Error log: {BASELINE_ERRORS}")
print(f"  - Summary: {BASELINE_SUMMARY}")
