# PetQA: Benchmarking Veterinary Knowledge and Clinical Reasoning

[![arXiv](https://img.shields.io/badge/arXiv-2609.04598-b31b1b.svg)](https://arxiv.org/abs/2609.04598)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-Dataset-FFD21E)](https://huggingface.co/datasets/humane-lab/PetQA)

PetQA is the first long-form question-answering dataset for evaluating veterinary knowledge and clinical reasoning about dogs and cats. It includes text-only and multimodal questions, paired with reference answers written by veterinarians verified by the source platform.

This repository provides the unified answer-generation and evaluation code for PetQA. The [dataset](https://huggingface.co/datasets/humane-lab/PetQA) is available on the Hugging Face Hub. For details about the benchmark and experiments, see our [paper](https://arxiv.org/abs/2609.04598). 



## 🐾 Dataset
### Overview
PetQA was collected from the pet medical consultation board of Naver Knowledge iN, covering posts published from 2014 to 2024. Preprocessing removed duplicate questions and posts with corrupted images. GPT-4o-mini was then used to filter irrelevant or uninformative content, personal information, promotional material, and noisy expressions while preserving essential clinical details.

After preprocessing, the final dataset contains 18,827 question–answer pairs. We split each modality into training, validation, and test sets.

| Modality | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Text | 6,076 | 2,000 | 2,000 | 10,076 |
| Multimodal | 4,751 | 2,000 | 2,000 | 8,751 |
| **Total** | **10,827** | **4,000** | **4,000** | **18,827** |

The test sets for benchmarking experiments are referred to as **PetQA-Bench**. PetQA-Bench is annotated under expert-reviewed guidelines and used to benchmark models' veterinary knowledge and clinical reasoning. The training and validation sets are used for supervised fine-tuning.

### PetQA-Bench annotation

Each PetQA-Bench sample was annotated along two dimensions: **question type** and **clinical conditions**. These labels were designed to assess model performance across different clinical scenarios and determine whether model responses contain key clinical conditions.

To assess data quality, a veterinary expert with a Ph.D. reviewed 100 randomly sampled question–answer pairs using two criteria: relevance and correctness. Relevance assessed whether each answer appropriately addressed the question, while correctness assessed its factual accuracy. The review found that 99% of the sampled answers were relevant and 85% were factually correct. These results support the overall reliability of the dataset.

### Multilingual extension

To broaden its global applicability, we used GPT-4o-mini to translate the original Korean benchmark into five additional languages: English, Chinese, German, Arabic, and Indonesian.




## 📥 Dataset Usage

```python
from datasets import load_dataset

data = load_dataset('humane-lab/PetQA')
```
The dataset is shared under the [CC BY-NC-ND 4.0 license](https://creativecommons.org/licenses/by-nc-nd/4.0/).




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
