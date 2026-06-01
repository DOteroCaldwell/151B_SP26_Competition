# CSE 151B Competition — Starter Code

Open **`starter_code_cse151b_comp.ipynb`** to get started.

The notebook covers environment setup, inference with Qwen3-4B-Thinking (INT8), and scoring against the public dataset.

## Contents

| File | Description |
|---|---|
| `starter_code_cse151b_comp.ipynb` | Main entry point |
| `judger.py` | Response scoring logic |
| `utils.py` | Utilities used by `judger.py` |
| `data/public.jsonl` | Public dataset with ground-truth answers |
| `results/` | Output JSONL files written at runtime |

## Environment to run Inference
GPU: A30, Approx inference time: 3 hr 45 min

## How to run test set

Weights are uploaded to HF. Only run inference file. Make sure data is called private.jsonl and is in data folder.
Final output is called private_test_results.csv

To run the model on the full private set, simply run 'uv run python inference.py'