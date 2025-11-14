#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified inference for FinQA program synthesis (compatible with base, merged, and PEFT adapters).

Examples:
# Zero-shot / merged model / small-data full FT
python inference_fixed.py \
  --model_path meta-llama/Llama-3.2-1B-Instruct \
  --test_json test.json \
  --out_json predictions_zeroshot.json \
  --max_new_tokens 384 --eg_k 1

# LoRA/QLoRA/Prefix/Prompt Tuning (unmerged adapter)
python inference_fixed.py \
  --base_model meta-llama/Llama-3.2-1B-Instruct \
  --adapter_path /path/to/adapter_dir \
  --test_json test.json \
  --out_json predictions_qlora.json \
  --max_new_tokens 384 --eg_k 1

# QLoRA with 4-bit inference (if you want true 4-bit at inference too)
python inference_fixed.py \
  --base_model meta-llama/Llama-3.2-1B-Instruct \
  --adapter_path /path/to/adapter_dir \
  --load_in_4bit \
  --test_json test.json \
  --out_json predictions_qlora_4bit.json \
  --max_new_tokens 384 --eg_k 1
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Union

import math
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional PEFT/4-bit imports guarded at runtime
try:
    from peft import PeftModel, PeftConfig
except Exception:
    PeftModel, PeftConfig = None, None
try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

# ---------------- Helpers ----------------

ALLOWED_OPS = {"add", "subtract", "multiply", "divide", "exp", "greater"}

def is_valid_op(op: str) -> bool:
    op = op.lower()
    return (op in ALLOWED_OPS) or op.startswith("table_")

def render_table(tbl: Union[List, Dict, str]) -> str:
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
        for k, v in tbl.items():
            lines.append(f"{k}: {v}")
    return "\n".join(lines)

def build_user_prompt(pre_text: List[str], post_text: List[str], table_obj: Any, question: str) -> str:
    pre = " ".join(pre_text) if isinstance(pre_text, list) else (pre_text or "")
    post = " ".join(post_text) if isinstance(post_text, list) else (post_text or "")
    table_str = render_table(table_obj)
    prompt = (
        "You are a program synthesis model for financial question answering.\n"
        "Given the context (before/after text) and a table, write a reasoning Program as a sequence of operations.\n"
        "Each step must look like: op(arg1, arg2). Use #k to reference prior step results (#0 is the first step's result). End with EOF.\n\n"
        "Context (before table):\n"
        f"{pre}\n\n"
        "Table:\n"
        f"{table_str}\n\n"
        "Context (after table):\n"
        f"{post}\n\n"
        f"Question: {question}\n\n"
        "Program:"
    )
    return prompt

def to_chat(tokenizer, user_prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )

def parse_raw_to_steps(text: str) -> List[List[str]]:
    # Keep only after "Program:" if present
    if "Program:" in text:
        text = text.split("Program:", 1)[1]
    # Cut off if model continued with other sections
    for stop in ["\n\nQuestion:", "\n\nContext:", "\n\nAnswer:", "\n\nFinal", "Explanation:"]:
        if stop in text:
            text = text.split(stop, 1)[0]
    s = " ".join(text.strip().split())
    if not s:
        return []
    # Regex for op(arg1, arg2)
    pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*([^,\)]+)\s*,\s*([^,\)]+)\s*\)'
    steps = []
    for m in re.finditer(pattern, s):
        op = m.group(1).strip()
        a1 = m.group(2).strip()
        a2 = m.group(3).strip()
        if not is_valid_op(op):
            continue
        if not a1.startswith("#") and not a1.startswith("const_"):
            a1 = a1.replace(",", "")
        if not a2.startswith("#") and not a2.startswith("const_"):
            a2 = a2.replace(",", "")
        steps.append([op, a1, a2])
    return steps

def steps_to_eval_tokens(steps):
    # FinQA evaluator expects 4 tokens per step: ['op(', 'arg1', 'arg2', ')']
    if not steps:
        return ["EOF"]
    tokens = []
    for (op, a1, a2) in steps:
        tokens.extend([f"{op}(", a1, a2, ")"])
    tokens.append("EOF")
    return tokens

def verify_eval_tokens(tokens):
    # Check 4-token step pattern + final EOF
    if not tokens or tokens[-1] != "EOF":
        return False
    if len(tokens) == 1:
        return True
    body = tokens[:-1]
    if len(body) % 4 != 0:
        return False
    for i in range(0, len(body), 4):
        t0, t1, t2, t3 = body[i:i+4]
        if not t0.endswith("("):
            return False
        op = t0[:-1].lower()
        if not (op in ALLOWED_OPS or op.startswith("table_")):
            return False
        if t3 != ")":
            return False
    return True

def load_json(path: Union[str, Path]) -> List[Dict]:
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("data", [])
    return data

# ---- Execution-guided utilities ----

def _exec_const(tok: str):
    if tok.startswith("const_"):
        try:
            return float(tok.split("_", 1)[1])
        except Exception:
            return None
    return None

def _exec_steps(steps):
    vals = []
    def val(x):
        if x.startswith("#"):
            idx = int(x[1:])
            return vals[idx]
        c = _exec_const(x)
        if c is not None:
            return c
        return float(x)

    for (op, a1, a2) in steps:
        try:
            x, y = val(a1), val(a2)
            if op == "add": z = x + y
            elif op == "subtract": z = x - y
            elif op == "multiply": z = x * y
            elif op == "divide": z = x / y if y != 0 else math.nan
            elif op == "exp": z = x ** y
            elif op == "greater": z = 1.0 if x > y else 0.0
            else: return None
            if math.isnan(z) or math.isinf(z): return None
            vals.append(z)
        except Exception:
            return None
    return vals[-1] if vals else None

def _tokens_to_steps(tokens):
    steps = []
    body = tokens[:-1]  # drop EOF
    for i in range(0, len(body), 4):
        op = body[i][:-1].lower()
        a1, a2 = body[i+1], body[i+2]
        steps.append([op, a1, a2])
    return steps

def eg_decode(model, tokenizer, inputs, max_new_tokens, K=5, temp=0.3, top_p=0.9, eos_id=None, pad_id=None):
    best_tokens = ["EOF"]
    for _ in range(K):
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temp,
            top_p=top_p,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )
        gen_text = tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        steps = parse_raw_to_steps(gen_text)
        tokens = steps_to_eval_tokens(steps)
        if not verify_eval_tokens(tokens):
            if len(tokens) > len(best_tokens):
                best_tokens = tokens
            continue
        val = _exec_steps(_tokens_to_steps(tokens))
        if val is not None:
            return tokens
        if len(tokens) > len(best_tokens):
            best_tokens = tokens
    return best_tokens

# ---------------- Model loading ----------------

def load_model_and_tokenizer(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Adapter path wins if provided
    if args.adapter_path:
        if PeftConfig is None:
            raise RuntimeError("peft not installed but --adapter_path specified.")
        base_name = args.base_model
        if base_name is None:
            # infer base from adapter config
            base_name = PeftConfig.from_pretrained(args.adapter_path).base_model_name_or_path

        tokenizer = AutoTokenizer.from_pretrained(base_name, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        # For stability: single device, fp16 compute by default
        four_bit_cfg = None
        if args.load_in_4bit:
            if BitsAndBytesConfig is None:
                raise RuntimeError("transformers BitsAndBytesConfig not available but --load_in_4bit set.")
            four_bit_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_name,
            quantization_config=four_bit_cfg,
            device_map=None,                   # single-device to avoid illegal access across shards
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
        ).to(device)

        model = PeftModel.from_pretrained(base_model, args.adapter_path)
        model = model.to(device)
        model.eval()
        print(f"✓ Loaded base: {base_name}")
        print(f"✓ Loaded adapter: {args.adapter_path}")
        return model, tokenizer, device

    # Otherwise, load a direct/merged model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None
    ).to(device)
    model.eval()
    print(f"✓ Loaded model: {args.model_path}")
    return model, tokenizer, device

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    # Pathing
    ap.add_argument("--model_path", type=str, help="Path to a direct/merged model")
    ap.add_argument("--base_model", type=str, default=None, help="Base model to load when using a PEFT adapter")
    ap.add_argument("--adapter_path", type=str, default=None, help="PEFT adapter directory (LoRA/QLoRA/Prefix/Prompt)")
    ap.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit inference when loading base+adapter")
    # Data & output
    ap.add_argument("--test_json", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)
    # Decoding
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--subset_size", type=int, default=None, help="Number of samples to test (subset)")
    ap.add_argument("--eg_k", type=int, default=1, help="#samples for EG; 1 = disabled (greedy)")
    ap.add_argument("--eg_temp", type=float, default=0.3, help="Sampling temperature for EG")
    ap.add_argument("--eg_top_p", type=float, default=0.9, help="Top-p for EG sampling")
    ap.add_argument("--input_max_len", type=int, default=2048, help="Truncation length for the prompt")

    args = ap.parse_args()

    if not args.model_path and not args.adapter_path:
        raise ValueError("Provide either --model_path (direct/merged) OR --adapter_path (+ optional --base_model).")

    model, tokenizer, device = load_model_and_tokenizer(args)

    test_items = load_json(args.test_json)
    if args.subset_size is not None:
        test_items = test_items[:args.subset_size]
        print(f"Testing on subset: {len(test_items)} samples")
    else:
        print(f"Loaded test items: {len(test_items)}")

    preds = []
    empties = 0
    valids = 0
    has_ops = 0

    for ex in tqdm(test_items, desc="Inference"):
        qa = ex.get("qa", {}) or {}
        q = qa.get("question", "")
        table_obj = ex.get("table", ex.get("table_ori", ""))
        user_prompt = build_user_prompt(ex.get("pre_text", ""), ex.get("post_text", ""), table_obj, q)
        chat_text = to_chat(tokenizer, user_prompt)

        inputs = tokenizer(
            chat_text,
            return_tensors="pt",
            truncation=True,
            max_length=args.input_max_len
        ).to(device)

        with torch.no_grad():
            if args.eg_k and args.eg_k > 1:
                tokens = eg_decode(
                    model, tokenizer, inputs,
                    max_new_tokens=args.max_new_tokens,
                    K=args.eg_k,
                    temp=args.eg_temp,
                    top_p=args.eg_top_p,
                    eos_id=tokenizer.eos_token_id,
                    pad_id=tokenizer.eos_token_id,
                )
            else:
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )
                gen_text = tokenizer.decode(
                    gen_ids[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
                steps = parse_raw_to_steps(gen_text)
                tokens = steps_to_eval_tokens(steps)

        if tokens == ["EOF"]:
            empties += 1
        if verify_eval_tokens(tokens):
            valids += 1
            if len(tokens) > 1:
                has_ops += 1

        preds.append({"id": ex.get("id", ""), "predicted": tokens})

    Path(args.out_json).write_text(json.dumps(preds, ensure_ascii=False, indent=2), encoding="utf-8")

    total = len(preds)
    print("\n================= STATISTICS =================")
    print(f"Total: {total}")
    print(f"Valid format: {valids} ({valids*100.0/total:.1f}%)")
    print(f"Has operations: {has_ops} ({has_ops*100.0/total:.1f}%)")
    print(f"Empty: {empties} ({empties*100.0/total:.1f}%)")
    print("==============================================")
    print(f"Saved predictions to: {args.out_json}")

if __name__ == "__main__":
    main()
