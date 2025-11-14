
# FinQA Program Synthesis with PEFT on LLaMA (LoRA / QLoRA / DoRA / QDoRA)

> A reproducible pipeline for **program-based financial question answering** on **FinQA**, using LLaMA-1B and LLaMA-3B models with parameter-efficient fine-tuning (LoRA, QLoRA, DoRA, QDoRA). Includes stable preprocessing, unified training, consistent inference, and optional W&B + Hugging Face integration.

<div align="center">

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)
![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-ffcc4d.svg)
![PEFT](https://img.shields.io/badge/PEFT-LoRA%2FDoRA%2FQLoRA-6f42c1.svg)
![Colab A100](https://img.shields.io/badge/Colab-A100-success.svg)
![W\&B](https://img.shields.io/badge/Weights%20%26%20Biases-optional-orange.svg)

</div>

---

## ✨ What's Included

* **Unified training scripts** for LoRA, DoRA, QLoRA, and QDoRA on LLaMA-1B and LLaMA-3B
* **Stable inference** supporting merged & unmerged adapters + optional 4-bit
* **FinQA-aligned preprocessing** (program linearization, table rendering, chat-template)
* **Clean collator** removing non-tensor fields
* **Hugging Face Hub integration**
* **W&B logging** for runtime, VRAM, power, and losses
* **Colab-friendly (A100, bf16, 4-bit)**

---

## 🧪 What Was Actually Experimented (Ground Truth)

### LLaMA-1B

* **LoRA:** r=8/16, α=16/32
* **DoRA:** r=4, α=8
* **QLoRA:** r=4, α=8
* **QDoRA:** r=4, α=8
* **Few-shot:** 200 examples (3.2% of training set)
* **Full fine-tuning** (baseline)

### LLaMA-3B

* **QLoRA:** r=16, α=32
* **QDoRA:** r=16, α=32
* **LoRA/DoRA attempts produced unstable program outputs**
  → Documented in report as known inference instability.

---

## 📦 Repository Structure

```
main_project/
├── data/
│   ├── train.json
│   ├── dev.json
│   └── test.json
│
├── scripts/
│   ├── preprocess_fixed.py
│   ├── train_lora.py
│   ├── dora_fixed.py
│   ├── qlora_fixed.py
│   ├── qdora_fixed.py
│   ├── full_fixed.py
│   ├── few_shot_fixed.py
│   ├── inference_fixed.py
│   ├── evaluate_fixed.py
│   └── save_to_HF.py
│
├── predictions/                # Partial results; full set stored on Drive
│
├── models/                     # Not stored locally; pushed to HuggingFace
│
├── README.md
└── requirements.txt
```

---

## 🛠️ Setup

### 1. Create environment and install deps

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft bitsandbytes datasets accelerate wandb huggingface_hub
```

### 2. Authenticate (optional)

```bash
huggingface-cli login
wandb login
```

---

## 🔄 Preprocessing (FinQA)

```bash
python scripts/preprocess_fixed.py \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --train_json data/train.json \
  --dev_json data/dev.json \
  --save_dir data/processed
```

✔ Converts FinQA format into chat-based LM examples
✔ Linearizes symbolic reasoning programs
✔ Saves HF Dataset structure (train/dev)

---

## 🚂 Training

### Example: QDoRA on LLaMA-3B

```bash
python scripts/qdora_fixed.py \
  --base_model meta-llama/Llama-3.2-3B-Instruct \
  --dataset_dir data/processed/train \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --use_bf16 true --use_4bit true \
  --epochs 3 --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --push_to_hub true --report_to wandb
```

**Notes**

* LoRA → set `--use_4bit false`
* QLoRA → keep `--use_4bit true`, NF4 quantization
* DoRA/QDoRA → special reparameterization inside script
* Gradient checkpointing recommended for 3B

---

## 🔮 Inference

### Using a merged checkpoint

```bash
python scripts/inference_fixed.py \
  --model_path models/llama-3b-finqa-qdora \
  --test_json data/test.json \
  --out_json predictions/qdora3b.json
```

### Using an unmerged adapter

```bash
python scripts/inference_fixed.py \
  --base_model meta-llama/Llama-3.2-3B-Instruct \
  --adapter_path models/llama3b_qdora_adapter \
  --test_json data/test.json
```

> ⚠ Some LoRA/DoRA-3B checkpoints produced malformed programs
> (repetition, truncation). This is a known behavior in low-rank reasoning.

---

## 📊 Evaluation

```bash
python scripts/evaluate_fixed.py \
  --pred_json predictions/output.json \
  --gold_json data/test.json
```

Outputs:

* **Program Accuracy (PA)**
* **Execution Accuracy (EA)**

---

## 📈 Optional: System Metrics via W&B

Logged automatically:

* Peak VRAM
* Runtime per epoch
* GPU power / energy (NVML)
* Loss curves

Full visualizations are available in the **W&B dashboard** and in the **appendix of the report**.

---

## 🧩 Future Work

A custom **South African financial QA dataset** is under development to test model generalization on local financial documents and reporting formats.
The experiments for the SA finQA dataset are under the folder SA_Dataset_Experiments. Preprocessing was largely completed, and QA pair generation was still underway. 
The sample outputs from the QA-pair generation indicated that prompt needed to be refined and the preprocessing pipeline needed more optimization (especially in the chunking phase). 

The QA-pairs generated followed a simple Context-Question-Answer formatting. Since FinQA was the dataset used in the main project, future implementations could consider creating a program-based
QA set based on South African financial documents instead. 

---

## 📚 Dataset Citation

```bibtex
@inproceedings{chen2021finqa,
  title={FinQA: A Large-Scale Financial Question Answering Dataset over Structured Tables and Text},
  author={Chen, Shiyue and others},
  booktitle={EMNLP},
  year={2021}
}
```

---

## 🙏 Acknowledgements

* FinQA dataset authors
* Hugging Face Transformers, PEFT, Datasets, Accelerate
* bitsandbytes NF4 quantization
* Google Colab A100 compute resources

---

## 💬 Issues / Questions

Please open an issue with:

1. Exact command
2. `pip freeze`
3. GPU + CUDA version
4. Whether 4-bit or bf16 was used


