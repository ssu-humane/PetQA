# PetQA: Benchmarking Veterinary Knowledge and Clinical Reasoning

[![arXiv](https://img.shields.io/badge/arXiv-2609.04598-b31b1b.svg)](https://arxiv.org/abs/2609.04598)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-Dataset-FFD21E)](https://huggingface.co/datasets/humane-lab/PetQA)

This repository provides the unified answer-generation and evaluation code for PetQA. For details about the benchmark and experiments, see our [paper](https://arxiv.org/abs/2609.04598). 

The [dataset](https://huggingface.co/datasets/humane-lab/PetQA) is available on the Hugging Face Hub.


## 🗂️ Repository Structure

```text
PetQA/
├── generation/     # vLLM, Transformers, and API generation backends
├── evaluation/     # Automatic and LLM-as-a-judge evaluation
├── prompts/        # Generation and evaluation prompts
├── scripts/        # Command-line entry points
├── utils/          # Shared configuration, data, batch, and I/O utilities
├── requirements.txt
└── README.md
```



## ⚙️ Setup

Python 3.12 is recommended.

```bash
pip install -r requirements.txt
```

For API-based generation or LLM-as-a-judge evaluation, set the relevant environment variable:

```bash
export OPENAI_API_KEY="..."
export GOOGLE_API_KEY="..."
```


## ▶️ Answer Generation

Choose a generation backend and modality:

```bash
bash scripts/run_generation.sh {vllm|transformers|api} \
  --modality {text|multimodal} [args]
```

### Zero-shot examples

```bash
# Text-only generation
bash scripts/run_generation.sh vllm --modality text \
  --model_name qwen-3-8b --strategy zeroshot

# Multimodal generation
bash scripts/run_generation.sh vllm --modality multimodal \
  --model_name qwen-3-vl-8b --strategy zeroshot

# English multimodal generation with an API model
bash scripts/run_generation.sh api --modality multimodal \
  --model_name gpt-4.1 --strategy zeroshot --language en
```

### RAG example

```bash
bash scripts/run_generation.sh vllm --modality text \
  --model_name qwen-3-8b --strategy rag --top_k 6 \
  --retrieved_paragraphs_path /path/to/retrieved_paragraphs.json
```

**RAG resources:** To run RAG experiments, users must provide passages retrieved from sources they are authorized to use.

The supplied retrieval file should contain one entry per test item:

```json
[
  {
    "q_id": 1,
    "a_id": 0,
    "retrieved_paragraphs": [
      {"content": "..."}
    ]
  }
]
```

### Clinical condition extraction

```bash
python -m generation.predict_disease --modality text \
  --model_name qwen-3-8b --strategy zeroshot
```

Generated answers are saved under `outputs/<modality>/results/generated_answers/`.


## 📊 Evaluation

Automatic evaluation computes ROUGE and BERTScore:

```bash
bash scripts/run_evaluation.sh auto --modality text \
  --model_name qwen-3-8b --strategy zeroshot
```

LLM-as-a-judge evaluation measures helpfulness and factuality:

```bash
bash scripts/run_evaluation.sh judge --modality multimodal \
  --model_name qwen-3-vl-8b --strategy zeroshot
```

Evaluation commands use the same model, strategy, language, and `top_k` values as the corresponding generation command.



## 📝 Citation

```bibtex
@misc{kim2026petqa,
  title         = {PetQA: Benchmarking Veterinary Knowledge and Clinical Reasoning},
  author        = {Taegyun Kim and Youngwook Ham and Jungwook Rhim and
                   Ju-Hyun An and Sungkyu Park and Kunwoo Park},
  year          = {2026},
  eprint        = {2609.04598},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2609.04598}
}
```
