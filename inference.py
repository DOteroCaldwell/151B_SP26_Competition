import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# On regular container, nvcc is not installed
# Sometimes, vllm will try to use flashinfer for sampling for some reason, disable it here
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
import re
import sys
import csv
import yaml
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

# ── Prompts ───────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def clean_response(value):
    if value is None:
        return ""
    return " ".join(str(value).splitlines())

def build_prompt(question: str, options: Optional[List[str]], math_prompt: str, mcq_prompt: str) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a question."""
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return mcq_prompt, f"{question}\n\nOptions:\n{opts_text}"
    return math_prompt, question

def get_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 1
    while True:
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


def load_completed_ids(output_path: Path) -> set[Any]:
    """Return IDs already present in an existing JSONL output file."""
    completed_ids = set()
    if not output_path.exists():
        return completed_ids

    with open(output_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed line {line_number} in {output_path}")
                continue
            if "id" in record:
                completed_ids.add(record["id"])
    return completed_ids


def ensure_trailing_newline(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, "rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) != b"\n":
            f.write(b"\n")


def make_record(item: dict, response: str) -> dict:
    record = {
        "id": item.get("id"),
        "is_mcq": bool(item.get("options")),
        "response": response,
    }
    # If the original data had an 'answer' field (public set), we can keep it
    if "answer" in item:
        record["gold"] = item["answer"]
    return record


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
    batch_size: int = 50,
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
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=True,
        gpu_memory_utilization=0.85,
        max_model_len=max_tokens,
        trust_remote_code=True,
        max_num_seqs=1024,
        max_num_batched_tokens=65536,
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

    # Save incrementally so completed work survives crashes/restarts.
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    completed_ids = load_completed_ids(out_path)
    ensure_trailing_newline(out_path)
    remaining_data = [item for item in data if item.get("id") not in completed_ids]
    batch_size = max(1, batch_size)

    if completed_ids:
        print(f"Found {len(completed_ids)} existing records in {out_path}; skipping them.")
    print(f"Generating responses for {len(remaining_data)} remaining questions...")

    saved_count = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for start in tqdm(range(0, len(remaining_data), batch_size), desc="Inference batches"):
            batch = remaining_data[start:start + batch_size]
            prompts = []
            for item in batch:
                system, user = build_prompt(item["question"], item.get("options"), math_prompt, mcq_prompt)

                # Format with chat template
                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "system", "content": system},
                     {"role": "user",   "content": user}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompts.append(prompt_text)

            outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
            for item, output in zip(batch, outputs):
                response = output.outputs[0].text.strip()
                f.write(json.dumps(make_record(item, response)) + "\n")
                saved_count += 1

            f.flush()
            os.fsync(f.fileno())

    print(f"Saved {saved_count} new records to {out_path}")
    print(f"Total records available in {out_path}: {len(completed_ids) + saved_count}")

    # For cleaning the file
    csv_out_path = get_unique_path(Path("private_test_results.csv"))
    with open(out_path, "r", encoding="utf-8") as infile, open(
        csv_out_path, "w", encoding="utf-8", newline=""
    ) as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["id", "response"])
        writer.writeheader()

        for line in infile:
            if not line.strip():
                continue

            item = json.loads(line)
            writer.writerow(
                {
                    "id": item.get("id", ""),
                    "response": clean_response(item.get("response", "")),
                }
            )
    print(f"Saved cleaned CSV to {csv_out_path}")


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
        batch_size=config.get("BATCH_SIZE", 50),
    )
