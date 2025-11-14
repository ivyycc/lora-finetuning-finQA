#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess FinQA for program synthesis (consistent with evaluator + chat template).

What this script does
---------------------
- Loads FinQA JSON (train/val/test) where each item has: pre_text, post_text, table, qa{question, program, exe_ans}
- Builds a single-turn chat prompt that ends with **"Program:"** (no "Final Answer:")
- Converts the gold `qa.program` (list or string) to a linearized string of steps like:
    divide(100, 50) add(#0, 10)
  and appends a literal " EOF" token so the model learns to stop.
- Tokenizes with the **same tokenizer as the base model**, using `apply_chat_template(..., add_generation_prompt=True)`
- Creates tensors: input_ids, attention_mask, labels where labels are -100 for prompt tokens and the target program tokens for the answer
- Saves Hugging Face Datasets to disk for train and eval.

Notes
-----
- We purposely do NOT include the numeric final answer in labels; only the program is learned.
- At inference you will parse the generated text back into 5-token steps and then append "EOF" in the JSON file you feed to the evaluator.

Usage
-----
python preprocess_fixed.py \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --train_json /path/to/train.json \
  --dev_json   /path/to/dev.json \
  --save_dir   ./finqa_program_chat_v1
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Union

from datasets import Dataset, Features, Sequence, Value
from transformers import AutoTokenizer

# ----------------------
# Helpers
# ----------------------

def render_table(tbl: Union[List, Dict, str]) -> str:
    """Render table into a compact, model-friendly string."""
    if tbl is None:
        return ""
    if isinstance(tbl, str):
        return tbl
    lines = []
    if isinstance(tbl, list):
        for row in tbl:
            if isinstance(row, list):
                lines.append(" | ".join(str(c) for c in row))
            elif isinstance(row, dict):
                lines.append(" | ".join(f"{k}: {v}" for k, v in row.items()))
            else:
                lines.append(str(row))
    elif isinstance(tbl, dict):
        # some variants store a dict of rows/cols
        for k, v in tbl.items():
            lines.append(f"{k}: {v}")
    return "\n".join(lines)

def linearize_program(program: Union[str, List[str]]) -> str:
    """
    Convert FinQA program into a compact text the model should generate.
    Accepts either a token list like ['divide','(','100',',','50',')', ...] or a string.
    Produces: 'divide(100, 50) add(#0, 10)' (steps separated by spaces).
    """
    if program is None:
        return ""
    if isinstance(program, str):
        # Normalize whitespace
        s = " ".join(program.strip().split())
    else:
        # It's a token list; stitch into a string first
        s = " ".join(program)

    # Convert common spaced forms into op(arg1, arg2)
    # We capture operation names and two comma-separated arguments
    pattern = r'(add|subtract|multiply|divide|exp|greater|table_[a-z_]+)\s*\(\s*([^,\)]+)\s*,\s*([^,\)]+)\s*\)'
    steps = []
    for m in re.finditer(pattern, s, flags=re.IGNORECASE):
        op = m.group(1).lower()
        a1 = m.group(2).strip()
        a2 = m.group(3).strip()
        # remove thousands separators inside numbers
        a1 = a1.replace(",", "")
        a2 = a2.replace(",", "")
        steps.append(f"{op}({a1}, {a2})")
    return " ".join(steps)

def build_user_prompt(pre_text: List[str], post_text: List[str], table_obj: Any, question: str) -> str:
    pre = " ".join(pre_text) if isinstance(pre_text, list) else (pre_text or "")
    post = " ".join(post_text) if isinstance(post_text, list) else (post_text or "")
    table_str = render_table(table_obj)
    # Clear, minimal formatting
    prompt = (
        "You are a program synthesis model for financial question answering.\n"
        "Given the context (before/after text) and a table, write a reasoning Program as a sequence of operations.\n"
        "Each step must look like: op(arg1, arg2). Use #k to reference the result of step k (starting at #0 after first step).\n\n"
        "Context (before table):\n"
        f"{pre}\n\n"
        "Table:\n"
        f"{table_str}\n\n"
        "Context (after table):\n"
        f"{post}\n\n"
        f"Question: {question}\n\n"
        "Program:"  # <- IMPORTANT: ends with Program:
    )
    return prompt

def to_chat_template(tokenizer, user_prompt: str) -> str:
    """Wrap user prompt in a single-user-message chat template."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )

def pack_example(tokenizer, user_prompt: str, target_prog: str) -> Dict[str, List[int]]:
    """
    Tokenize prompt + labels (labels only for the program + ' EOF').
    Returns dict with input_ids, attention_mask, labels.
    """
    # Ensure we have some target text; attach ' EOF' so the model learns to stop.
    target_text = (target_prog.strip() + " EOF").strip()

    prompt_text = to_chat_template(tokenizer, user_prompt)

    prompt_enc = tokenizer(prompt_text, add_special_tokens=False)
    target_enc = tokenizer(target_text, add_special_tokens=False)

    input_ids = prompt_enc["input_ids"] + target_enc["input_ids"]
    attention_mask = prompt_enc["attention_mask"] + target_enc["attention_mask"]

    labels = [-100] * len(prompt_enc["input_ids"]) + target_enc["input_ids"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def load_jsonl_or_json(path: Path) -> List[Dict]:
    # Basic loader that accepts .json (list) or .jsonl
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            # sometimes wrapped
            data = data.get("data", [])
    except json.JSONDecodeError:
        # jsonl
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    return data

# ----------------------
# Main
# ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--train_json", type=str, required=True)
    ap.add_argument("--dev_json", type=str, required=True)
    ap.add_argument("--save_dir", type=str, required=True)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def convert_split(examples: List[Dict]) -> Dataset:
        rows = []
        for ex in examples:
            qa = ex.get("qa", {}) or {}
            question = qa.get("question", "")
            # Prefer 'table' if present; fall back to 'table_ori' if your dump uses that name.
            table_obj = ex.get("table", ex.get("table_ori", ""))

            user_prompt = build_user_prompt(ex.get("pre_text", ""), ex.get("post_text", ""), table_obj, question)

            prog = linearize_program(qa.get("program", ""))
            rows.append({
                "id": ex.get("id", ""),
                "prompt_text": user_prompt,
                "program_text": prog,
            })

        # Tokenize + pack
        def _map_pack(batch):
            packed = [pack_example(tokenizer, p, t) for p, t in zip(batch["prompt_text"], batch["program_text"])]
            return {
                "input_ids": [x["input_ids"] for x in packed],
                "attention_mask": [x["attention_mask"] for x in packed],
                "labels": [x["labels"] for x in packed],
                "id": batch["id"],
            }

        ds = Dataset.from_dict({
            "id": [r["id"] for r in rows],
            "prompt_text": [r["prompt_text"] for r in rows],
            "program_text": [r["program_text"] for r in rows],
        })
        ds = ds.map(_map_pack, batched=True, remove_columns=["prompt_text", "program_text"])
        return ds

    train_data = load_jsonl_or_json(Path(args.train_json))
    dev_data   = load_jsonl_or_json(Path(args.dev_json))

    ds_train = convert_split(train_data)
    ds_dev   = convert_split(dev_data)

    out = Path(args.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds_train.save_to_disk(str(out / "train"))
    ds_dev.save_to_disk(str(out / "dev"))

    print(f"Saved to: {out}/train and {out}/dev")
    print(ds_train)
    print(ds_dev)

if __name__ == "__main__":
    main()