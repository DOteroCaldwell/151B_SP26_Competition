import json
import os
import re
import sys
import csv
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
    "\nEXAMPLES:\n\n"
    r"""Q: Solve for $x>0$ in the following arithmetic sequence: $1^2, x^2, 3^2, \ldots$.
    A: The term $x^2$ is simply the average of $1^2 = 1$ and $3^2 = 9$, so $x^2 = (1 + 9)/2 = 5$.  Because $x > 0$, $x = \\boxed{sqrt(5)}\n
    Q: If the system of equations  \begin{align*} 3x+y&=a, 2x+5y&=2a, \end{align*} has a solution $(x,y)$ when $x=2$, compute $a$.
    A: Substituting in $x=2$, we obtain the equations \begin{align*} y+6&=a,\\ 5y+4&=2a.\end{align*} Multiplying the first equation by $5$ and subtracting it from the second equation, we find $$-26=-3a\Rightarrow a=\boxed{26/3}.
    Q: What is the area of the circle defined by $x^2-6x +y^2-14y +33=0$ that lies beneath the line $y=7$?
    A: Add $(-6/2)^2$ and $(-14/2)^2$ to both sides of the equation to get (x^2-6x +9) +(y^2-14y +49)=25, which in turn can be rewritten as $(x-3)^2 +(y-7)^2 =5^2$.  The center of this circle is $(3,7)$, so the line $y=7$ passes through the center of the circle.  Hence, the area of the circle that lies below $y=7$ is half the area of the circle.  The radius of the circle is sqrt(25) = 5$, so the circle has area $25*pi$.  Therefore, half the area of the circle is $\\boxed{(25*pi)/2$.
    Q: Simplify
    \[\frac{\sin{10^\circ}+\sin{20^\circ}}{\cos{10^\circ}+\cos{20^\circ}}.\] (If using angle in final answer, it should be positive and as small as possible.)
    A: From the product-to-sum identities,
    \[\frac{\sin{10^\circ}+\sin{20^\circ}}{\cos{10^\circ}+\cos{20^\circ}} = \frac{2 \sin 15^\circ \cos (-5^\circ)}{2 \cos 15^\circ \cos(-5^\circ)} = \frac{\sin 15^\circ}{\cos 15^\circ} = \boxed{tan(15)}.\]

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
def clean_response(value):
    if value is None:
        return ""
    return " ".join(str(value).splitlines())

def build_prompt(question: str, options: Optional[List[str]]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a question."""
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question


def run_inference(
    model_id: str = "alexcojo/Qwen3-Finetuned",
    data_path: str = "data/private.jsonl",
    output_path: str = "results/starter_results.jsonl",
    gpu_id: str = "0",
    max_tokens: int = 16384,
    sampling_max_tokens: int = 8192,
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
        system, user = build_prompt(item["question"], item.get("options"))
        
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

    # For cleaning the file
    with open(output_path, "r", encoding="utf-8") as infile, open(
        "private_test_results.csv", "w", encoding="utf-8", newline=""
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


if __name__ == "__main__":
    # Example usage:
    # run_inference(model_id="./results/qwen3_math_lora_merged", data_path="data/private.jsonl")
    run_inference()
