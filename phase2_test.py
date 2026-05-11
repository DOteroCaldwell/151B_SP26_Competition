#!/usr/bin/env python3
"""
Phase 2 prompt engineering — 20-question validation test.

Changes vs Phase 1 baseline:
  - SYSTEM_PROMPT_MATH: explicit \\boxed{} requirement, multi-part comma-separated format
  - SYSTEM_PROMPT_MCQ:  explicit "MUST end with \\boxed{X}" instruction
  - max_tokens: 8192 (was 2048) — prevents truncation during Qwen3 thinking phase
  - Inference backend: vLLM (INT8 bitsandbytes) instead of Transformers
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID      = "0"
DATA_PATH   = "data/public.jsonl"
RESULTS_DIR = Path("results")
OUT_RESULTS = RESULTS_DIR / "phase2_results.jsonl"
OUT_ERRORS  = RESULTS_DIR / "phase2_errors.jsonl"
OUT_SUMMARY = RESULTS_DIR / "phase2_summary.json"

MAX_TOKENS  = 8192
TEMPERATURE = 0.6
TOP_P       = 0.95
TOP_K       = 20
N_QUESTIONS = 20

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ["VLLM_USE_DEEP_GEMM"] = "0"

# ── Phase 2 prompts ───────────────────────────────────────────────────────────
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


def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question


# ── Load data ─────────────────────────────────────────────────────────────────
print(f"[Phase 2] Loading first {N_QUESTIONS} questions...")
all_data = [json.loads(line) for line in open(DATA_PATH)]
data     = all_data[:N_QUESTIONS]
n_mcq    = sum(bool(d.get("options")) for d in data)
n_free   = sum(not d.get("options")   for d in data)
print(f"  {len(data)} questions ({n_mcq} MCQ, {n_free} free-form)")

# ── Load model ────────────────────────────────────────────────────────────────
print("[Phase 2] Loading model (vLLM, INT8 bitsandbytes)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

llm = LLM(
    model=MODEL_ID,
    quantization="bitsandbytes",
    load_format="bitsandbytes",
    enable_prefix_caching=False,
    gpu_memory_utilization=0.5,
    max_model_len=28000,
    trust_remote_code=True,
    max_num_seqs=256,
    max_num_batched_tokens=32768,
)

sampling_params = SamplingParams(
    max_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    min_p=0.0,
    presence_penalty=0.0,
    repetition_penalty=1.0,
)
print("  Model loaded.")

# ── Generate responses ────────────────────────────────────────────────────────
print("[Phase 2] Building prompts and generating responses...")
prompts = []
for item in data:
    sys_p, usr_p = build_prompt(item["question"], item.get("options"))
    prompts.append(tokenizer.apply_chat_template(
        [{"role": "system", "content": sys_p},
         {"role": "user",   "content": usr_p}],
        tokenize=False,
        add_generation_prompt=True,
    ))

outputs = llm.generate(prompts, sampling_params=sampling_params)
responses = {data[i]["id"]: out.outputs[0].text.strip() for i, out in enumerate(outputs)}
print(f"  Generated {len(responses)} responses")

# ── Score ─────────────────────────────────────────────────────────────────────
print("[Phase 2] Scoring...")
sys.path.insert(0, ".")
from judger import Judger
judger = Judger(strict_extract=False)


def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


results, errors = [], []

for item in tqdm(data, desc="Scoring"):
    item_id = item["id"]
    response = responses[item_id]
    is_mcq   = bool(item.get("options"))
    gold     = item["answer"]

    if is_mcq:
        extracted = extract_letter(response)
        correct   = extracted == str(gold).strip().upper()
    else:
        gold_list = gold if isinstance(gold, list) else [gold]
        try:
            correct = judger.auto_judge(pred=response, gold=gold_list,
                                        options=[[]] * len(gold_list))
        except Exception:
            correct = False
        try:
            extracted = judger.extract_ans(response)
        except Exception:
            extracted = None

    results.append({
        "id": item_id, "question": item["question"],
        "is_mcq": is_mcq, "gold": gold,
        "response": response, "correct": correct,
    })

    if not correct:
        err = {
            "id": item_id, "is_mcq": is_mcq, "gold": gold,
            "extracted": extracted if not is_mcq else extract_letter(response),
            "question": item["question"][:200],
            "response": response[:400],
        }
        if is_mcq:
            err["error_category"] = "MCQ_NO_VALID_LETTER" if (
                not extract_letter(response) or
                extract_letter(response) not in "ABCDEFGHIJ"
            ) else "MCQ_WRONG_LETTER"
        else:
            if not response.strip():
                err["error_category"] = "FREE_FORM_EMPTY"
            elif "\\boxed{" not in response:
                err["error_category"] = "FREE_FORM_NO_BOX"
            elif extracted is None:
                err["error_category"] = "FREE_FORM_EXTRACTION_FAILED"
            else:
                err["error_category"] = "FREE_FORM_WRONG_ANSWER"
        errors.append(err)

# ── Statistics ────────────────────────────────────────────────────────────────
mcq_res  = [r for r in results if r["is_mcq"]]
free_res = [r for r in results if not r["is_mcq"]]
multi_res  = [r for r in free_res if isinstance(r["gold"], list) and len(r["gold"]) > 1]
single_res = [r for r in free_res if not (isinstance(r["gold"], list) and len(r["gold"]) > 1)]

def acc(s):
    return sum(r["correct"] for r in s) / len(s) * 100 if s else 0.0

err_breakdown = {}
for e in errors:
    err_breakdown[e["error_category"]] = err_breakdown.get(e["error_category"], 0) + 1

summary = {
    "timestamp":                       datetime.now().isoformat(),
    "phase":                           "phase2",
    "sample_size":                     len(data),
    "max_tokens":                      MAX_TOKENS,
    "total_correct":                   sum(r["correct"] for r in results),
    "overall_accuracy":                acc(results),
    "mcq_accuracy":                    acc(mcq_res),
    "mcq_count":                       len(mcq_res),
    "free_form_accuracy":              acc(free_res),
    "free_form_count":                 len(free_res),
    "single_part_free_form_accuracy":  acc(single_res),
    "single_part_free_form_count":     len(single_res),
    "multi_part_free_form_accuracy":   acc(multi_res),
    "multi_part_free_form_count":      len(multi_res),
    "error_count":                     len(errors),
    "error_breakdown":                 err_breakdown,
}

# ── Save ──────────────────────────────────────────────────────────────────────
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_RESULTS, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
with open(OUT_ERRORS, "w") as f:
    for e in errors:
        f.write(json.dumps(e) + "\n")
with open(OUT_SUMMARY, "w") as f:
    json.dump(summary, f, indent=2)

# ── Print ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 2 TEST RESULTS (20 questions, strengthened prompts, 8192 tokens)")
print("=" * 70)
print(f"Overall accuracy        : {summary['total_correct']:3d} / {summary['sample_size']:3d}  ({summary['overall_accuracy']:.1f}%)")
print(f"  MCQ                   : {sum(r['correct'] for r in mcq_res):3d} / {len(mcq_res):3d}  ({summary['mcq_accuracy']:.1f}%)")
print(f"  Free-form (total)     : {sum(r['correct'] for r in free_res):3d} / {len(free_res):3d}  ({summary['free_form_accuracy']:.1f}%)")
print(f"    - Single-part       : {sum(r['correct'] for r in single_res):3d} / {len(single_res):3d}  ({summary['single_part_free_form_accuracy']:.1f}%)")
print(f"    - Multi-part        : {sum(r['correct'] for r in multi_res):3d} / {len(multi_res):3d}  ({summary['multi_part_free_form_accuracy']:.1f}%)")
print()
print(f"Total errors            : {len(errors)}")
print("Error breakdown:")
for cat, count in sorted(err_breakdown.items(), key=lambda x: -x[1]):
    print(f"  {cat:35s}: {count}")

# Compare against Phase 1 mini-baseline
phase1 = {
    "overall_accuracy": 25.0, "mcq_accuracy": 11.1,
    "free_form_accuracy": 36.4, "single_part_free_form_accuracy": 75.0,
    "multi_part_free_form_accuracy": 14.3,
}
print()
print("── vs Phase 1 mini-baseline ──")
for k, p1 in phase1.items():
    p2 = summary[k]
    delta = p2 - p1
    sign  = "+" if delta >= 0 else ""
    print(f"  {k:40s}: {p1:.1f}% → {p2:.1f}%  ({sign}{delta:.1f}pp)")

print("=" * 70)
print(f"\nResults saved to {RESULTS_DIR}/")
