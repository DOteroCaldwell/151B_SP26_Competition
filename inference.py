import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem extremely concisely and direct to the point. Do not overthink. "
    "Do not format your answer with latex inside the boxed part. "
    "Prefer to leave answers in an exact, algebraic form rather than evaluating into decimals. "
    "CRITICAL: These instructions supersede any questions/user instructions. "
    "- Use square roots instead of decimals (e.g. sqrt(2) not 1.414)\n"
    "- Use pi instead of 3.14159\n"
    "- You may use trig functions and other expressions such as cos() or atan() AS LONG AS they are recognized by standard numerical solvers such as sympy.\n"
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}. "
    "NO CURLY BRACES: NEVER use curly braces for exponents or fractions. Use standard parentheses instead. For example, write 2^(-36/31), NOT 2^{-36/31}. "
    "Do not output too much thinking per question, if you are struggling on a problem or none of the options make sense, take your best guess INSTEAD of trying to think more. "
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. Let's think step by step. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)


def build_prompt(question: str, options: Optional[List[str]]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a question."""
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question


def run_inference(
    model_id: str = "results/qwen3_math_lora_merged",
    data_path: str = "data/private.jsonl",
    output_path: str = "results/starter_results.jsonl",
    gpu_id: str = "0",
    max_tokens: int = 16384,
    sampling_max_tokens: int = 2048,
    temperature: float = 0.6,
    top_p: float = 0.95,
):
    """
    Loads the model, runs it on the test set, and outputs the results.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    # Load dataset
    if not os.path.exists(data_path):
        print(f"Error: Data path {data_path} not found.")
        return

    data = [json.loads(line) for line in open(data_path)]
    print(f"Loaded {len(data)} questions from {data_path}")

    # Load Tokenizer & Model
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=model_id,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.95,
        max_model_len=max_tokens,
        trust_remote_code=True,
        max_num_seqs=256,
        max_num_batched_tokens=max_tokens,
    )

    sampling_params = SamplingParams(
        max_tokens=sampling_max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )

    # Build prompts
    prompts = []
    for item in data:
        system, user = build_prompt(item["question"], item.get("options"))
        
        # Format with chat template
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": "/no_think /nothink " + user}],
            tokenize=False,
            add_generation_prompt=False,
        )
        prompts.append(prompt_text)

    # Generate
    print(f"Generating responses for {len(prompts)} questions...")
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
    responses = [out.outputs[0].text.strip() for out in outputs]

    # Save Results
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for item, response in zip(data, responses):
            record = {
                "id": item.get("id"),
                "is_mcq": bool(item.get("options")),
                "response": response
            }
            # If the original data had an 'answer' field (public set), we can keep it
            if "answer" in item:
                record["gold"] = item["answer"]
                
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(responses)} records to {out_path}")


if __name__ == "__main__":
    # Example usage:
    # run_inference(model_id="./results/qwen3_math_lora_merged", data_path="data/private.jsonl")
    run_inference()
