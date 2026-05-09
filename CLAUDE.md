# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CSE 151B Spring 2026 Math Reasoning Competition. The goal is to improve a model's accuracy on a mixed math dataset (MCQ + free-form) by modifying prompts, sampling parameters, or fine-tuning. The baseline model is **Qwen3-4B-Thinking-2507** (INT4 quantized) running on DSMLP (DataHub) with a CUDA GPU.

## Environment Setup (first time only)

Run the setup cell in the notebook, then **restart the kernel** and switch to the `Python (cse151b)` kernel:

```bash
# uv installs into .venv/ — run once, then restart kernel
uv venv .venv --seed
.venv/bin/python -m pip install sympy numpy transformers vllm tqdm bitsandbytes antlr4-python3-runtime==4.11.1 ipykernel jupyter
.venv/bin/python -m ipykernel install --user --name cse151b --display-name "Python (cse151b)"
```

Each session: run the PATH cell (`os.environ['PATH'] += ...`) and the `source .venv/bin/activate` cell before anything else.

## Running the Pipeline

Open `starter_code_cse151b_comp.ipynb` and run cells top to bottom. Key configuration at the top of Section 2:

```python
MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID      = "0"
DATA_PATH   = "data/public.jsonl"
OUTPUT_PATH = "results/starter_results.jsonl"
MAX_TOKENS  = 2048
```

Results land in `results/` as JSONL. Set `SAVE_EVAL = False` when running on the private test set (no ground truth).

## Architecture

### Data format (`data/public.jsonl`)
Each line is a JSON object:
- `id` — integer
- `question` — problem statement (may contain LaTeX)
- `options` — list of strings for MCQ; absent for free-form
- `answer` — gold letter string for MCQ; list of strings for free-form (multi-part answers are separate list elements)

### Inference pipeline (notebook)
1. **Prompt construction** (`build_prompt`): returns `(system_prompt, user_prompt)`. MCQ gets `SYSTEM_PROMPT_MCQ` (output letter in `\boxed{}`); free-form gets `SYSTEM_PROMPT_MATH` (step-by-step, answer in `\boxed{}`).
2. **Model loading**: Transformers path (active) uses INT4 BitsAndBytes + `device_map="cuda"`. vLLM path (commented out) uses INT8 + `gpu_memory_utilization=0.50`.
3. **Generation**: chat template applied via `tokenizer.apply_chat_template`, batch tokenized, then `llm.generate()`.
4. **Scoring**: MCQ uses regex extraction of the boxed letter; free-form uses `Judger.auto_judge()`.
5. **Output**: JSONL with `{id, is_mcq, gold, response, correct}` (evaluation) or `{id, is_mcq, response}` (submission).

### Judger (`judger.py` + `utils.py`)

`Judger` handles answer extraction and type-specific comparison. Answer types:

| Code | Type | Judge method |
|------|------|------|
| `NV` | Single numerical value | Numeric + SymPy evaluation |
| `EX` | Expression | SymPy symbolic simplification |
| `EQ` | Equation | Both sides parsed, difference simplified |
| `TF` | True/False | Exact string match after normalization |
| `INT` | Interval | Parses `\cup`-joined interval notation |
| `MCS` | MC single answer | Letter comparison |
| `MCM` | MC multiple answers | Letter-set comparison |
| `OL` | Ordered list | Element-wise equality |
| `UOL` | Unordered list | Set equality |
| `OE` | Open-ended | Case-insensitive string match |

**Answer extraction flow** (`Judger.extract_ans`): looks for `\boxed{}` first (strips `<think>...</think>` before searching), then falls back to last LaTeX formula or last number. `auto_judge` is used when `type_sequence` is unknown; `judge` is used when type is known per element.

`utils.py` provides: LaTeX normalization constants (`SIMPLE_RM_STRS`, `SIMPLE_REPLACE_MAP`, `LATEX_CMDS`), string-to-number map (`STR2NUM`), weekday/bool normalizers, and fix helpers (`fix_fracs`, `fix_sqrt`, `fix_a_slash_b`, `fix_inv_func`).

## Improving the Baseline

Levers to pull (in order of effort):
1. **System prompts** — edit `SYSTEM_PROMPT_MATH` / `SYSTEM_PROMPT_MCQ` in the notebook
2. **Sampling parameters** — `temperature`, `top_p`, `top_k`, `max_new_tokens`; majority voting across multiple samples
3. **Model swap** — change `MODEL_ID` to a larger or fine-tuned checkpoint
4. **Fine-tuning** — train on the public set; adjust `judger.py` scoring if needed
