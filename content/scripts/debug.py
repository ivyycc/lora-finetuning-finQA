#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Emergency Fix: Check and repair tokenized datasets
Run this BEFORE training!
"""

from datasets import load_from_disk
from transformers import AutoTokenizer
import torch

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
TRAIN_PATH= "/content/drive/MyDrive/Colab Notebooks/project_notebooks/finqa_program_tokenized_train_robust"  # Update this path
EVAL_PATH = "/content/drive/MyDrive/Colab Notebooks/project_notebooks/finqa_program_tokenized_eval_robust"
print("="*60)
print("EMERGENCY FIX: Checking tokenized data")
print("="*60)

# Load tokenizer to get vocab size
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
vocab_size = len(tokenizer)
print(f"\nModel vocabulary size: {vocab_size}")

# Load datasets
print("\nLoading datasets...")
train_dataset = load_from_disk(TRAIN_PATH)
eval_dataset = load_from_disk(EVAL_PATH)

print(f"Train size: {len(train_dataset)}")
print(f"Eval size: {len(eval_dataset)}")

def check_and_fix_dataset(dataset, name, vocab_size):
    """Check for invalid token IDs and fix them"""
    print(f"\n{'='*60}")
    print(f"CHECKING {name.upper()}")
    print(f"{'='*60}")
    
    issues_found = 0
    examples_with_issues = []
    
    for idx in range(min(100, len(dataset))):  # Check first 100
        example = dataset[idx]
        
        # Check input_ids
        input_ids = example['input_ids']
        invalid_inputs = [i for i in input_ids if i >= vocab_size or i < 0]
        
        # Check labels
        labels = example['labels']
        invalid_labels = [l for l in labels if l >= vocab_size and l != -100]
        
        if invalid_inputs or invalid_labels:
            issues_found += 1
            examples_with_issues.append(idx)
            
            if issues_found <= 3:  # Print first 3 issues
                print(f"\n❌ Issue in example {idx}:")
                if invalid_inputs:
                    print(f"   Invalid input_ids: {invalid_inputs[:5]}")
                if invalid_labels:
                    print(f"   Invalid labels: {invalid_labels[:5]}")
    
    if issues_found == 0:
        print(f"\n✅ {name}: NO ISSUES FOUND!")
        return False
    else:
        print(f"\n🚨 {name}: Found {issues_found} examples with issues")
        print(f"   Examples: {examples_with_issues[:10]}")
        return True

# Check both datasets
train_has_issues = check_and_fix_dataset(train_dataset, "Train", vocab_size)
eval_has_issues = check_and_fix_dataset(eval_dataset, "Eval", vocab_size)

# Summary
print("\n" + "="*60)
print("DIAGNOSIS SUMMARY")
print("="*60)

if train_has_issues or eval_has_issues:
    print("\n🚨 CRITICAL: Your tokenized data has INVALID TOKEN IDs!")
    print("\nThis causes the 'index out of bounds' error during training.")
    print("\n📋 SOLUTION:")
    print("1. Your preprocessing script has a bug")
    print("2. You need to re-run preprocessing with the FIXED version")
    print("3. DO NOT train with this data - it will crash!")
    
    print("\n🔧 WHAT TO DO:")
    print("1. Delete the current tokenized datasets")
    print("2. Use preprocess_enhanced.py or preprocess_fixed.py")
    print("3. Re-tokenize from scratch")
    print("4. Run THIS script again to verify")
    
    print("\n⚠️  LIKELY CAUSE:")
    print("Your original preprocessing created 'text' field incorrectly")
    print("or used wrong tokenizer settings.")
    
else:
    print("\n✅ ALL CHECKS PASSED!")
    print("Your data is valid. The error must be elsewhere.")
    print("\nOther possible causes:")
    print("1. Tokenizer mismatch (used different tokenizer)")
    print("2. Corrupted dataset files")
    print("3. Memory issues")

print("="*60)