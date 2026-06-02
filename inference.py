import json
import os
import re
import sys
<<<<<<< Updated upstream
=======
import csv
import yaml
import argparse
>>>>>>> Stashed changes
from pathlib import Path
from typing import Optional, List, Dict, Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

# ── Prompts ───────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

<<<<<<< Updated upstream
    """
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. Let's think step by step. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
    """ EXAMPLES: \n\n
    Q: John found that the average of 15 numbers is 40. If 10 is added to each number then the mean of the
    numbers is? Answer Choices: (a) 50 (b) 45 (c) 65 (d) 78 (e) 64\n
    A: If 10 is added to each number, then the mean of the numbers also increases by 10. So the new mean
    would be 50. The answer is \\boxed(a).\n
    Q: If a / b = 3/4 and 8a + 5b = 22,then find the value of a. Answer Choices: (a) 1/2 (b) 3/2 (c) 5/2 (d) 4/2 (e) 7/2\n
    A: If a / b = 3/4, then b = 4a / 3. So 8a + 5(4a / 3) = 22. This simplifies to 8a + 20a / 3 = 22, which means
    44a / 3 = 22. So a is equal to 3/2. The answer is \\boxed{b}.\n
    Q: A person is traveling at 20 km/hr and reached his destiny in 2.5 hr then find the distance? Answer Choices:
    (a) 53 km (b) 55 km (c) 52 km (d) 60 km (e) 50 km\n
    A: The distance that the person traveled would have been 20 km/hr * 2.5 hrs = 50 km. The answer is \\boxed{e}.\n
    Q: How many keystrokes are needed to type the numbers from 1 to 500? Answer Choices: (a) 1156 (b) 1392
    (c) 1480 (d) 1562 (e) 1788\n
    A: There are 9 one-digit numbers from 1 to 9. There are 90 two-digit numbers from 10 to 99. There are 401
    three-digit numbers from 100 to 500. 9 + 90(2) + 401(3) = 1392. The answer is \\boxed{b}."""
)

=======
def clean_response(value):
    if value is None:
        return ""
    return " ".join(str(value).splitlines())
>>>>>>> Stashed changes

def build_prompt(question: str, options: Optional[List[str]], math_prompt: str, mcq_prompt: str) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a question."""
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return mcq_prompt, f"{question}\n\nOptions:\n{opts_text}"
    return math_prompt, question


def run_inference(
    model_id: str,
    data_path: str,
    output_path: str,
    gpu_id: str,
    max_tokens: int,
    sampling_max_tokens: int,
    temperature: float,
    top_p: float,
    math_prompt: str,
    mcq_prompt: str,
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
        max_num_seqs=1024,
        max_num_batched_tokens=max_tokens,
    )

    sampling_params = SamplingParams(
        max_tokens=sampling_max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.1,
    )

    # Build prompts
    prompts = []
    for item in data:
        system, user = build_prompt(item["question"], item.get("options"), math_prompt, mcq_prompt)
        
        # Format with chat template
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False,
            add_generation_prompt=True,
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
    parser = argparse.ArgumentParser(description="Run inference with a specified config file.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file (default: config.yaml)")
    args = parser.parse_args()

    config = load_config(args.config)

    run_inference(
        model_id=config["MODEL_ID"],
        data_path=config["DATA_PATH"],
        output_path=config["OUTPUT_PATH"],
        gpu_id=config["GPU_ID"],
        max_tokens=config["MAX_TOKENS"],
        sampling_max_tokens=config["SAMPLING_MAX_TOKENS"],
        temperature=config["TEMPERATURE"],
        top_p=config["TOP_P"],
        math_prompt=config["SYSTEM_PROMPT_MATH"],
        mcq_prompt=config["SYSTEM_PROMPT_MCQ"],
    )
