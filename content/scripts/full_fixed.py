#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full Finetune for FinQA — remote checkpoints to HF each epoch + final upload.
- Custom collator strips non-tensor fields before padding.
- remove_unused_columns=True in TrainingArguments.
- No persistent local saves: checkpoints are uploaded then deleted; final saved via tempdir.
"""

import argparse
import os
import datetime
from pathlib import Path
import tempfile

import wandb
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, LlamaForCausalLM,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)

from huggingface_hub import HfApi, login
from save_to_HF import PushToHubOnSave  # your callback that uploads checkpoint on save

ALLOWED_KEYS = {"input_ids", "attention_mask", "labels"}

class CleanLMDataCollator(DataCollatorForLanguageModeling):
    def torch_call(self, features):
        features = [{k: v for k, v in f.items() if k in ALLOWED_KEYS} for f in features]
        return super().torch_call(features)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train_bs", type=int, default=1)
    ap.add_argument("--eval_bs", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--early_stopping_patience", type=int, default=2)
    ap.add_argument("--early_stopping_threshold", type=float, default=0.0)
    ap.add_argument("--hf_token", type=str, required=True)
    ap.add_argument("--hf_repo", type=str, default="ivyycc/finqa-1b-experiments")
    ap.add_argument("--run_tag", type=str, default=None)      # if None → auto
    ap.add_argument("--max_shard_size", type=str, default="2GB")
    args = ap.parse_args()

    wandb.init(project="Research Project", name="full-ft", reinit=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model (full fine-tune)...")
    model = LlamaForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto"
    )
    # Memory & stability
    model.gradient_checkpointing_enable()

    # Data
    train_ds = load_from_disk(os.path.join(args.data_dir, "train"))
    dev_ds   = load_from_disk(os.path.join(args.data_dir, "dev"))
    data_collator = CleanLMDataCollator(tokenizer=tokenizer, mlm=False)

    # Build run tag if not provided
    if not args.run_tag:
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = Path(args.model_name).name
        args.run_tag = f"{base_name}_fullft_e{args.epochs}_lr{args.lr}_{stamp}"

    # Auth & ensure repo exists
    login(token=args.hf_token)
    api = HfApi()
    try:
        api.repo_info(args.hf_repo, repo_type="model")
    except Exception:
        user, name = args.hf_repo.split("/", 1)
        api.create_repo(name=name, repo_type="model", private=True)

    # Training args (epoch saves trigger remote upload via callback)
    training_args = TrainingArguments(
        output_dir="./_work_tmp",                 # scratch; checkpoints removed after upload
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.06,
        weight_decay=0.10,                        # full FT benefits from stronger WD
        max_grad_norm=1.0,                        # gradient clipping
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=None,                    # we'll delete locals after upload
        load_best_model_at_end=False,             # no local persistence
        bf16=torch.cuda.is_available(),
        fp16=args.fp16 and not torch.cuda.is_available(),
        optim="adamw_torch_fused",                # fast & stable on A100
        report_to=["wandb"],
        remove_unused_columns=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True
    )

    callbacks = [
        PushToHubOnSave(
            repo_id=args.hf_repo,
            run_tag=args.run_tag,
            hf_token=None,         # already logged in
            keep_local=False       # delete local checkpoint after upload
        ),
        EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold
        )
    ]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    trainer.train()

    # Metrics
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f" Peak VRAM usage: {peak_vram:.2f} GB")
    wandb.log({"peak_vram_gb": peak_vram})

    # Final upload (no persistent local save): save to tmp, push, auto-delete
    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir, safe_serialization=True, max_shard_size=args.max_shard_size)
        tokenizer.save_pretrained(tmpdir)
        final_subpath = f"{args.run_tag}/final"
        print(f"[upload-final] {tmpdir} → {args.hf_repo}/{final_subpath}")
        api.upload_folder(
            repo_id=args.hf_repo,
            folder_path=tmpdir,
            path_in_repo=final_subpath,
            commit_message=f"final weights {args.run_tag}",
            ignore_patterns=["**/.git/*", "**/.ipynb_checkpoints/*"],
        )

    print(f"✅ Remote checkpoints & final uploaded: https://huggingface.co/{args.hf_repo}/tree/main/{args.run_tag}")
    wandb.finish()

if __name__ == "__main__":
    main()
