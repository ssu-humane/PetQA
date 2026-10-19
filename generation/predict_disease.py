import argparse
import os
import re

# Disease extraction also runs vLLM through its Python API, potentially with
# tensor parallel workers, so use the CUDA-safe multiprocessing method here.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from utils.config import (
    SUPPORTED_LANGUAGES,
    SUPPORTED_MODALITIES,
    SUPPORTED_STRATEGIES,
    build_endpoint,
    get_modality_spec,
)
from utils.data import disease_prompt_path, load_prompt
from utils.io import (
    collect_answer_paths,
    load_json,
    output_paths,
    save_json,
    set_seed,
    setup_logging,
)


PREFIX_RE = re.compile(
    r"^(?:\s*(?:질병명|Disease|disease|답변)\s*[:：\-\s]+)+", re.IGNORECASE
)


def has_gold_disease_label(item):
    value = item.get("gold_disease", item.get("gold_diseases"))
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def extract_disease(text):
    if not text:
        return None
    value = PREFIX_RE.sub("", str(text).strip()).strip().strip("'\"` ")
    if value.lower() in {"null", "없음", "해당없음"}:
        return None
    diseases = [part.strip() for part in value.split(",")]
    diseases = [part for part in diseases if part and part.lower() != "null"]
    return ", ".join(diseases) if diseases else None


def build_disease_prompts(data, tokenizer, prompt):
    rendered = []
    for item in data:
        rendered.append(
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": prompt.format(answer=item["pred_answer"]),
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return rendered


def predict_diseases(
    llm,
    tokenizer,
    data,
    prompt,
    batch_size,
    max_tokens,
):
    from vllm import SamplingParams

    indices = [
        index
        for index, item in enumerate(data)
        if item.get("gold_question_type") == "진단" and has_gold_disease_label(item)
    ]
    selected = [data[index] for index in indices]
    prompts = build_disease_prompts(selected, tokenizer, prompt)
    sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens, seed=42)
    for start in range(0, len(prompts), batch_size):
        outputs = llm.generate(
            prompts[start : start + batch_size],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        batch_indices = indices[start : start + batch_size]
        for index, output in zip(batch_indices, outputs):
            data[index]["pred_disease"] = extract_disease(output.outputs[0].text)
    return data


def build_parser():
    parser = argparse.ArgumentParser(description="Extract disease names from generated PetQA answers with vLLM.")
    parser.add_argument("--modality", choices=SUPPORTED_MODALITIES, required=True)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--strategy", choices=SUPPORTED_STRATEGIES, default="zeroshot")
    parser.add_argument("--language", choices=SUPPORTED_LANGUAGES, default="ko")
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--judge_model_name", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    spec = get_modality_spec(args.modality)
    args.model_name = args.model_name or spec.default_models["disease"]
    args.judge_model_name = (args.judge_model_name or spec.default_models["disease_judge"])
    args.batch_size = (
        args.batch_size
        if args.batch_size is not None
        else spec.default_batch_sizes["disease"]
    )
    endpoint = build_endpoint(
        args.model_name,
        args.strategy,
        args.language,
        args.top_k,
        spec,
    )
    paths = output_paths(spec)
    inputs = collect_answer_paths(paths.generated_answers, endpoint)
    if not inputs:
        return

    import torch
    from vllm import LLM

    set_seed(42)
    logger = setup_logging()
    llm = LLM(
        model=spec.model_mapping[args.judge_model_name],
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tokenizer = llm.get_tokenizer()
    prompt = load_prompt(disease_prompt_path())
    for input_path in inputs:
        data = load_json(input_path)
        destination = paths.predicted_diseases / input_path.name
        save_json(
            predict_diseases(
                llm,
                tokenizer,
                data,
                prompt,
                args.batch_size,
                args.max_tokens,
            ),
            destination,
        )
        logger.info("Disease extraction finished: %s", destination)


if __name__ == "__main__":
    main()
