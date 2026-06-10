#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **kwargs):
        return iterable

sys.path.insert(0, ".")
from judger import Judger


def find_default_results_path() -> Path:
    candidates = []
    for path in Path("results").glob("*.jsonl"):
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                first_line = next((line for line in f if line.strip()), "")
            if first_line and "gold" in json.loads(first_line):
                candidates.append(path)
        except (OSError, json.JSONDecodeError):
            continue

    if not candidates:
        raise FileNotFoundError("No results/*.jsonl file with a gold field was found.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_results(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "gold" not in record:
                raise ValueError(f"Line {line_number} in {path} is missing the gold field.")
            if "response" not in record:
                raise ValueError(f"Line {line_number} in {path} is missing the response field.")
            if "is_mcq" not in record:
                raise ValueError(f"Line {line_number} in {path} is missing the is_mcq field.")
            records.append(record)
    return records


def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def score_mcq(response: str, gold_letter: str) -> bool:
    return extract_letter(response) == gold_letter.strip().upper()


def score_records(records: list[dict]) -> list[dict]:
    try:
        judger = Judger(strict_extract=False)
    except ImportError as exc:
        raise SystemExit(
            "Failed to initialize Judger. Install the notebook grading dependency with:\n"
            "  pip install antlr4-python3-runtime==4.11.1"
        ) from exc

    scored = []

    for record in tqdm(records, total=len(records), desc="Scoring"):
        response = record["response"]
        gold = record["gold"]
        is_mcq = bool(record["is_mcq"])

        if is_mcq:
            correct = score_mcq(response, str(gold))
        else:
            gold_list = gold if isinstance(gold, list) else [gold]
            try:
                correct = judger.auto_judge(
                    pred=response,
                    gold=gold_list,
                    options=[[]] * len(gold_list),
                )
            except Exception:
                correct = False

        scored.append(
            {
                "id": record.get("id"),
                "is_mcq": is_mcq,
                "gold": gold,
                "response": response,
                "correct": correct,
            }
        )

    return scored


def acc(subset: list[dict]) -> float:
    return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0


def print_summary(results: list[dict]) -> None:
    mcq_res = [r for r in results if r["is_mcq"]]
    free_res = [r for r in results if not r["is_mcq"]]

    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
    print(f"  Free-form  : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
    print(f"  Overall    : {sum(r['correct'] for r in results):4d} / {len(results):4d}  ({acc(results):.2f}%)")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a results JSONL file against its gold answers.")
    parser.add_argument(
        "results_path",
        nargs="?",
        type=Path,
        help="Path to a results JSONL file. Defaults to newest results/*.jsonl with a gold field.",
    )
    args = parser.parse_args()

    results_path = args.results_path or find_default_results_path()
    records = load_results(results_path)
    print(f"Evaluating {len(records)} records from {results_path}")

    results = score_records(records)
    print_summary(results)


if __name__ == "__main__":
    main()
