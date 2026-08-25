# PetQA: Benchmarking Veterinary Knowledge and Clinical Reasoning

PetQA is a Korean long-form question-answering benchmark for evaluating veterinary knowledge and clinical reasoning in large language models (LLMs) and large vision-language models (LVLMs). This repository contains its unified generation and evaluation code.

## Data access

PetQA data is available only to approved users. Read [`data/instructions.md`](data/instructions.md) for the application and manual approval process. Do not open an issue or commit a received download URL to this repository.

After approval, place the delivered files in this layout:

```text
data/
├── TEXT_ONLY/
│   ├── train.json
│   ├── validation.json
│   ├── test.json
│   ├── test_translated_en.json
│   ├── test_translated_de.json
│   ├── test_translated_zh.json
│   ├── test_translated_ar.json
│   └── test_translated_id.json
└── TEXT_w_IMAGE/
    ├── train.json
    ├── validation.json
    ├── test.json
    ├── test_translated_en.json
    ├── test_translated_de.json
    ├── test_translated_zh.json
    ├── test_translated_ar.json
    ├── test_translated_id.json
    └── images/
        └── <image-id>.<ext>
```

Each item in `test.json` has the following shape:

```json
{
  "q_id": 1,
  "a_id": 0,
  "question": "...",
  "gold_answer": "...",
  "gold_question_type": "진단",
  "gold_disease": "...",
  "animal_type": "dog"
}
```

Multimodal items add an `image` field containing an image ID. Translated files retain the original fields and add `question_<language>` and `gold_answer_<language>`.

## Setup

Python 3.12 is recommended.

```bash
pip install -r requirements.txt
```

For API generation or LLM evaluation, set the relevant `OPENAI_API_KEY` or `GOOGLE_API_KEY` environment variable.

## Generation

Choose one backend and one modality:

```bash
bash scripts/run_generation.sh {vllm|transformers|api} --modality {text|multimodal} [args]
```

### Endpoint and output names

All commands use these endpoint forms:

- Text zero-shot: `<model>_zeroshot[_<language>]`
- Text RAG: `<model>_RAG-<top_k>`
- Multimodal zero-shot: `<model>_zeroshot_multimodal[_<language>]`
- Multimodal RAG: `<model>_RAG-<top_k>_multimodal`

Because source documents may be copyright-protected, the repository and approved data package do not include a RAG corpus or retrieved passages; supply passages produced from sources you are authorized to use.

Examples:

```bash
# Text zero-shot
bash scripts/run_generation.sh vllm --modality text \
  --model_name qwen-3-8b --strategy zeroshot

# Multimodal zero-shot
bash scripts/run_generation.sh vllm --modality multimodal \
  --model_name qwen-3-vl-8b --strategy zeroshot

# Text RAG
bash scripts/run_generation.sh vllm --modality text \
  --model_name qwen-3-8b --strategy rag --top_k 6 \
  --retrieved_paragraphs_path /path/to/retrieved_paragraphs.json

# English multimodal API generation
bash scripts/run_generation.sh api --modality multimodal \
  --model_name gpt-4.1 --strategy zeroshot --language en
```

### Disease extraction

Disease extraction reads all generated runs for the selected endpoint and writes corresponding files under `outputs/<modality>/results/predicted_diseases/`:

```bash
python -m generation.predict_disease --modality text \
  --model_name qwen-3-8b --strategy zeroshot
```

## Evaluation

Automatic and LLM-as-a-judge evaluation resolve exactly the same endpoints:

```bash
bash scripts/run_evaluation.sh auto --modality text \
  --model_name qwen-3-8b --strategy zeroshot

bash scripts/run_evaluation.sh judge --modality multimodal \
  --model_name qwen-3-vl-8b --strategy zeroshot
```

## Citation

Citation information will be added upon publication.
