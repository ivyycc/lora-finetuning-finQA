# FinQA Program Synthesis with PEFT on Llama (LoRA / QLoRA / DoRA / QDoRA)

> Reproducible fine‑tuning and inference pipeline for **program‑based QA** on **FinQA**, using Llama‑family models with parameter‑efficient methods (LoRA, QLoRA, DoRA, QDoRA). Includes clean data collators, stable inference, optional W&B tracking, and helpers to push checkpoints to the Hugging Face Hub.

<div align="center">
  
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-ffcc4d.svg)](https://huggingface.co/transformers)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA%2FDoRA%2FQLoRA-6f42c1.svg)](https://github.com/huggingface/peft)
[![Colab A100](https://img.shields.io/badge/Colab-A100-success.svg)](https://colab.research.google.com/)
[![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-optional-orange.svg)](https://wandb.ai/)

</div>

---

## ✨ What’s inside
- **Unified training** for LoRA/DoRA/QLoRA on Llama‑1B/3B (bf16/4‑bit aware).
- **Stable inference** script (`inference_fixed.py`) that works with:
  - Base or merged checkpoints ✅
  - Unmerged PEFT adapters (LoRA/DoRA/QLoRA) ✅
  - 4‑bit inference (optional) ✅
- **Clean collator** that strips non‑tensor fields and avoids common HF/PEFT pitfalls.
- **Push to Hub** each epoch and on final save (optional).
- Colab‑friendly defaults and **CUDA memory hygiene** (grad checkpointing recommended).

> If you trained with **bf16** on an **A100**, you’re good — the scripts auto‑detect mixed precision and quantization.

---

## 📦 Repository layout
> Filenames may differ slightly in your repo; adjust paths accordingly.

```
.
├─ data/
│  ├─ FinQA_train.json
│  ├─ FinQA_dev.json
│  └─ FinQA_test.json
├─ scripts/
│  ├─ preprocess.py
│  ├─ train_dora.py              # or train_peft.py (LoRA/DoRA/QLoRA)
│  ├─ inference_fixed.py
│  └─ save_to_HF.py              # PushToHubEachEpoch helper
├─ models/
│  └─ llama-3b-finqa-lora/       # example output (merged or adapters)
├─ README.md
└─ requirements.txt (optional)
```

---

## 🛠️ Setup

### 1) Environment
```bash
python -m venv .venv && source .venv/bin/activate  # (Linux/Mac)
# or on Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121   # pick your CUDA
pip install transformers peft bitsandbytes datasets accelerate wandb huggingface_hub
```

> If `bitsandbytes` fails on Windows/CPU, disable 4‑bit and run fp16/bf16 instead.

### 2) (Optional) Login to the Hub & W&B
```bash
huggingface-cli login
wandb login
```

---

## 🧹 Data preprocessing (FinQA)
Make sure your JSON follows the preprocessed schema used in training. If you have a `preprocess.py`, run:

```bash
python scripts/preprocess.py \
  --input_json data/FinQA_train.json \
  --output_dataset ./data/processed/train
```

If you already have HF `DatasetDict` folders (e.g., `load_from_disk`), point the training script to them directly.

---

## 🚂 Training

Below is a **DoRA** example; swap flags for LoRA/QLoRA as needed.

```bash
python scripts/train_dora.py \
  --base_model meta-llama/Llama-3.2-3B-Instruct \
  --dataset_dir ./data/processed/train \
  --output_dir ./models/llama-3b-finqa-dora \
  --epochs 3 --per_device_train_batch_size 4 \
  --lr 1e-4 --weight_decay 0.0 \
  --gradient_accumulation_steps 4 \
  --use_bf16 true \
  --use_4bit true \               # QLoRA-style training (set false for standard DoRA/LoRA)
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --logging_steps 20 --save_strategy epoch \
  --push_to_hub true \
  --report_to wandb
```

**Notes**
- For **LoRA**: set `--use_4bit false` (unless you want true QLoRA) and keep LoRA params.
- For **QLoRA**: keep `--use_4bit true` and ensure `prepare_model_for_kbit_training` is applied.
- For **DoRA/QDoRA**: ensure your script swaps the LoRA re‑parameterization to DoRA.
- Enable **gradient checkpointing** to reduce memory.
- Set `remove_unused_columns=True` in `TrainingArguments` to avoid collator errors.

---

## 🔮 Inference (`inference_fixed.py`)

### A) Base / merged checkpoint
```bash
python scripts/inference_fixed.py \
  --model_path ./models/llama-3b-finqa-dora \
  --test_json data/FinQA_test.json \
  --out_json outputs/predictions_merged.json \
  --max_new_tokens 384 \
  --eg_k 1
```

### B) Unmerged PEFT adapter (LoRA/DoRA/QLoRA)
```bash
python scripts/inference_fixed.py \
  --base_model meta-llama/Llama-3.2-3B-Instruct \
  --adapter_path ./models/llama-3b-finqa-dora-adapter \
  --test_json data/FinQA_test.json \
  --out_json outputs/predictions_adapter.json \
  --max_new_tokens 384 \
  --eg_k 1 \
  --load_in_4bit true         # optional: true 4‑bit inference
```

**Script guarantees**
- Uses `tokenizer.apply_chat_template(..., add_generation_prompt=True)`
- Works whether you fine‑tuned in bf16 or not.
- Cleans feature dicts to only keep `input_ids`, `attention_mask`, `labels`.
- Deterministic `torch.manual_seed` for reproducibility (where applicable).

---

## 📊 Evaluation
Add your scorer here (example):
```bash
python scripts/score_finqa.py \
  --pred_json outputs/predictions_adapter.json \
  --gold_json data/FinQA_test.json \
  --out_csv outputs/metrics.csv
```

Insert your results:

| Model (Params) | Method | Quant | Dev Acc | Test Acc | Notes |
| --- | --- | --- | ---: | ---: | --- |
| Llama‑3.2‑1B | LoRA | fp16 |  |  |  |
| Llama‑3.2‑3B | DoRA | bf16 |  |  |  |
| Llama‑3.2‑3B | QDoRA | 4‑bit |  |  |  |

---

## 🧩 Repro tips
- **Colab A100 + bf16** is the smoothest path.
- If you see logs like _“Unable to register cuDNN factory”_, they’re usually harmless TF/XLA noise when PyTorch is active.
- For CUDA OOM, try:
  - Reduce `per_device_train_batch_size`
  - Increase `gradient_accumulation_steps`
  - Enable grad checkpointing
  - Disable Flash‑Attention if your stack doesn’t support it
- Ensure `bitsandbytes` matches your CUDA (12.1 vs 11.8).

---

## 🤝 Push to Hub
Training script includes a callback that can push each epoch and a final snapshot:
```python
from save_to_HF import PushToHubEachEpoch
trainer.add_callback(PushToHubEachEpoch(repo_id="your-username/llama-finqa-dora"))
```

---

## 🗂️ Config cheatsheet
Common flags you may want to surface in your `argparse`:
- `--base_model`, `--model_path`, `--adapter_path`
- `--use_bf16`, `--use_4bit`, `--gradient_checkpointing`
- `--lora_r`, `--lora_alpha`, `--lora_dropout` (or DoRA variants)
- `--max_new_tokens`, `--do_sample`, `--temperature`, `--top_p`
- `--eg_k` (few‑shot example count)
- `--save_strategy epoch`, `--logging_steps`, `--eval_strategy steps`

---

## 📁 Dataset citation
- **FinQA**: *Chen et al., 2021*. Please cite the original paper/dataset when publishing results.

```bibtex
@inproceedings{chen2021finqa,
  title={FinQA: A Large-Scale Financial Question Answering Dataset over Structured Tables and Text},
  author={...},
  booktitle={EMNLP},
  year={2021}
}
```

---

## 🙏 Acknowledgements
- Hugging Face Transformers, PEFT, Datasets, Accelerate
- bitsandbytes (4‑bit quantization)
- FinQA authors and maintainers
- Colab team for generous A100 time 

---

## 💬 Questions / Issues
If something breaks, please open an issue with:
1. Full command + git commit hash
2. `pip freeze` (Torch/Transformers/PEFT/bitsandbytes versions)
3. GPU + CUDA version, and whether you used 4‑bit or bf16
