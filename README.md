# lora-finetuning-finQA
COMPARATIVE STUDY ON LORA, QLORA, AND DORA FOR EFFICIENT FINE-TUNING OF LLMS IN FINANCIAL QUESTION ANSWERING TASKS

financial-qa-finetuning/
├── data/                     # Downloaded raw PDFs or HTML
├── processed/                # Cleaned and chunked data
├── summaries/                # Extractive and abstractive summaries
├── qa_pairs/                 # JSON or CSV of question-answer pairs
├── models/                   # Finetuned model checkpoints
├── scripts/                  # Chunking, summarization, QA gen
├── notebooks/                # Jupyter notebooks (optional)
├── requirements.txt
├── README.md
└── main.py

pip install torch transformers datasets peft accelerate
pip install sentencepiece nltk scikit-learn pandas
pip install rank_bm25 faiss-cpu # for vector DB and retrieval
pip install git+https://github.com/huggingface/peft.git
pip install llama-index==0.10.35
pip install tiktoken # if using OpenAI tokenizer
pip install beautifulsoup4 requests

pip install summa # for TextRank
pip install transformers[sentencepiece] # for PEGASUS

pip install bitsandbytes xformers #If using FLAN5 on HF




