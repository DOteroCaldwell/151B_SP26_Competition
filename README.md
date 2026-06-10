# CSE 151B Competition — Starter Code

Open **`starter_code_cse151b_comp.ipynb`** to get started.

The notebook covers environment setup, inference with Qwen3-4B-Thinking (INT8), and scoring against the public dataset.

## Contents

| File | Description |
|---|---|
| `starter_code_cse151b_comp.ipynb` | Main entry point for setup, inference, and public-dataset scoring |
| `fine_tune_qwen.ipynb` | QLoRA fine-tuning notebook for Qwen3-4B-Thinking |
| `inference.py` | Batch inference script for private/test-set generation and CSV export |
| `run_baseline.py` | Full public-set baseline runner with scoring and error analysis |
| `clean_dataset.py` | Dataset cleaning script for turning raw math data into normalized JSONL |
| `judger.py` | Response scoring logic |
| `utils.py` | Utilities used by `judger.py` |
| `data/public.jsonl` | Public dataset with ground-truth answers |
| `results/` | Output JSONL files written at runtime |

## Environment to run Inference
GPU: A30, Approx inference time: 3 hr 45 min

## How to run test set

The private-set pipeline is implemented in [inference.py](inference.py). It loads the hosted model weights from Hugging Face, reads `data/private.jsonl`, runs batched vLLM inference, and writes the final submission file as `private_test_results.csv` in the repository root.

Before running the script, ensure the private evaluation file is named `private.jsonl` and placed under `data/`. Then execute:

`uv run python inference.py`

The script will generate model responses for every example in the input JSONL and export the resulting `id,response` pairs into the submission CSV.

## Disclaimer

Some code in this repository may have been generated or assisted by AI tools.