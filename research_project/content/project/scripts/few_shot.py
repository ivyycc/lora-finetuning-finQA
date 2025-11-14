
import wandb


# %%
from datasets import load_from_disk

# --- Load tokenized datasets ---
tokenized_train = load_from_disk("/content/drive/MyDrive/Colab Notebooks/project_notebooks/finqa_program_tokenized_train_robust")
tokenized_eval = load_from_disk("/content/drive/MyDrive/Colab Notebooks/project_notebooks/finqa_program_tokenized_eval_robust")


# %% [markdown]
# Finetune config

# %%
from transformers import AutoTokenizer, LlamaForCausalLM, TrainingArguments, Trainer
import torch
import time


# ✅ Take a small subset for few-shot fine-tuning
fewshot_train = tokenized_train.select(range(200))  # 200 samples
fewshot_eval = tokenized_eval.select(range(40))     # 40 samples

model_name = "meta-llama/Llama-3.2-1B-Instruct"

# Load tokenizer from saved path (CONSISTENT with other methods)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding is consistent

print("=== CORRUPTED TOKENIZER ===")
print(f"EOS token: {tokenizer.eos_token} -> {tokenizer.eos_token_id}")
print(f"PAD token: {tokenizer.pad_token} -> {tokenizer.pad_token_id}")
print(f"Vocab size: {tokenizer.vocab_size}")

# Check fresh tokenizer
fresh_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
print("\n=== FRESH TOKENIZER ===")
print(f"EOS token: {fresh_tokenizer.eos_token} -> {fresh_tokenizer.eos_token_id}")
print(f"PAD token: {fresh_tokenizer.pad_token} -> {fresh_tokenizer.pad_token_id}")
print(f"Vocab size: {fresh_tokenizer.vocab_size}")

# Initialize W&B run (CONSISTENT with other methods)
wandb.init(
    project="Research Project",
    name="fewshot-1B-run1",  # change per model/run
    config={
        # Model info
        "model": "Llama-3.2-1B-Instruct",
        "method": "Few-Shot",

        # Training args
        "learning_rate": 2e-4,
        "batch_size": 1,
        "grad_accum": 4,  # Few-shot uses 8 (as in your original)
        "epochs": 2,
        "precision": "bf16",
        "few_shot_samples": 200,  # Specific to few-shot
        "few_shot_eval_samples": 50,

        # No LoRA-specific params
    }
)



# Load model (CONSISTENT precision with LoRA/DoRA)
model = LlamaForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # ✅ Consistent with LoRA/DoRA
    device_map="auto",
    low_cpu_mem_usage=True
)

# ✅ Enable gradient checkpointing to save VRAM
model.gradient_checkpointing_enable()

print("Few-shot model loaded successfully!")


# %% [markdown]
# Train model

# %%

training_args_fewshot = TrainingArguments(
    output_dir="./llama-fewshot-finetuned",
    per_device_train_batch_size=4,           # Increased from 1
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,           # Reduced from 4
    num_train_epochs=4,                      # Increased from 2
    learning_rate=2e-4,                      # More conservative
    bf16=True,
    fp16=False,

    # ✅ FIXED evaluation strategy:
    eval_strategy="epoch",                   # Critical fix!
    save_strategy="epoch",                   # Align with eval
    logging_steps=25,                        # Less frequent

    # Remove unnecessary:
    gradient_checkpointing=False,            # 1B model doesn't need this

    # Add standard regularization:
    weight_decay=0.01,
    warmup_steps=50,

    remove_unused_columns=False,
    report_to="wandb",
)

# Trainer
trainer_fewshot = Trainer(
    model=model,
    args=training_args_fewshot,
    train_dataset=fewshot_train,  # Using the subset
    eval_dataset=fewshot_eval,    # Using the subset
    tokenizer=tokenizer,  # ✅ Using the same tokenizer
)

# Training with timing and memory tracking (IDENTICAL to LoRA)
start_time = time.time()
torch.cuda.reset_peak_memory_stats()
trainer_fewshot.train()
end_time = time.time()

total_training_time = end_time - start_time
peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

print(f"✅ Training complete in {total_training_time/60:.2f} minutes")
print(f"💾 Peak VRAM usage: {peak_vram:.2f} GB")

# 🟩 Log custom metrics to W&B (IDENTICAL to LoRA)
wandb.log({
    "total_training_time_min": total_training_time / 60,
    "peak_vram_gb": peak_vram
})

# Save model and tokenizer (IDENTICAL structure to LoRA)
save_path = "/content/drive/MyDrive/Colab Notebooks/project_notebooks/llama-fewshot-finetuned"
trainer_fewshot.save_model(save_path)
tokenizer.save_pretrained(save_path)  # ✅ Save the same tokenizer
print(f"✅ Model saved at {save_path}")

wandb.finish()  # ✅ End W&B run

# %%


# %% [markdown]
# validate

# %%


# %% [markdown]
# test

# %%


# %% [markdown]
# 


