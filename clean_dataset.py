import json
import re
import os
import sys

def clean_text(text):
    if not isinstance(text, str):
        return str(text)
    # The Odyssey dataset seems to have "\\\\\n\\noindent" or similar at the end
    # We want to remove \noindent and any preceding backslashes or whitespace/newlines
    cleaned = re.sub(r'[\s\\]*\\noindent\s*', '', text)
    # Also remove trailing LaTeX line breaks like \\
    cleaned = re.sub(r'\\+\s*$', '', cleaned)
    return cleaned.strip()

def clean_question(text):
    if not isinstance(text, str):
        return str(text)
    # Removes "Problem 1:", "Problem 2:", "Problem_n:", "Problem_1", etc. at the beginning
    cleaned = re.sub(r'^Problem[\s_]*(\d+|n)[:\s\.]*', '', text, flags=re.IGNORECASE).strip()
    # Removes \end{problem} and \noindent from the end
    cleaned = re.sub(r'\\end\{problem\}', '', cleaned)
    cleaned = re.sub(r'[\s\\]*\\noindent\s*', '', cleaned)
    return cleaned.strip()

def process_dataset(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return

    items_processed = 0
    with open(output_path, 'w', encoding='utf-8') as out_f:
        with open(input_path, 'r', encoding='utf-8') as in_f:
            for i, line in enumerate(in_f):
                if not line.strip():
                    continue
                try:
                    raw_item = json.loads(line)
                    # Handle nested structure: {"Problem_1": {...}}
                    if isinstance(raw_item, dict) and len(raw_item) == 1:
                        key = list(raw_item.keys())[0]
                        if key.startswith('Problem'):
                            item = raw_item[key]
                        else:
                            item = raw_item
                    else:
                        item = raw_item

                    # Extract fields. Odyssey uses 'question', public.jsonl uses 'question'
                    question = item.get('question', item.get('problem', ''))
                    answer = item.get('answer', '')
                    
                    # Clean the question
                    cleaned_question = clean_question(question)
                    # Clean the answer
                    cleaned_answer = clean_text(answer)
                    
                    # public.jsonl uses ["answer"] for free-form
                    output_item = {
                        "id": i,
                        "question": cleaned_question,
                        "answer": [cleaned_answer] if isinstance(cleaned_answer, str) else cleaned_answer
                    }
                    
                    # Include other metadata if present
                    for k, v in item.items():
                        if k not in ['question', 'problem', 'answer', 'id']:
                            output_item[k] = v
                            
                    out_f.write(json.dumps(output_item, ensure_ascii=False) + '\n')
                    items_processed += 1
                except Exception as e:
                    print(f"Skipping line {i} due to error: {e}")

    print(f"Successfully processed {items_processed} items and saved to {output_path}")

if __name__ == "__main__":
    # Default filenames based on user request
    input_file = "final-oddysey-math.json"
    output_file = "final-oddysey-math-cleaned.jsonl"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
        
    process_dataset(input_file, output_file)
