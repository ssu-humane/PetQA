import argparse
import json
import os
import re
import statistics
import time

from utils.batch import GeminiBatchClient, OpenAIBatchClient
from utils.config import (
    SUPPORTED_LANGUAGES,
    SUPPORTED_MODALITIES,
    SUPPORTED_STRATEGIES,
    build_endpoint,
    get_modality_spec,
)
from utils.data import evaluation_prompt_path, load_prompt, sample_key
from utils.io import (
    batch_file_paths,
    collect_answer_paths,
    format_time,
    load_json,
    output_paths,
    save_json,
)


SEED = 42


def parse_helpfulness(text):
    match = re.search(r"\[\[([1-5])\]\]", text)
    if not match:
        raise ValueError(f"Helpfulness score not found: {text[:200]}")
    analysis = text[: match.start()].strip()
    if analysis.startswith("Rating:"):
        analysis = analysis.removeprefix("Rating:").strip()
    return int(match.group(1)), analysis


def _openai_text(row):
    return str(row["response"]["body"]["choices"][0]["message"]["content"])


def _gemini_text(row):
    parts = row["response"]["candidates"][0]["content"]["parts"]
    return "\n".join(str(part.get("text", "")) for part in parts).strip()


def _parse_json_response(text):
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    return json.loads(value)


def build_helpfulness_rows(
    data, prompt, model_id
):
    rows = []
    for item in data:
        user_prompt = prompt.format(
            question=item["question"], pred_answer=item["pred_answer"]
        )
        rows.append(
            {
                "custom_id": sample_key(item),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_id,
                    "messages": [{"role": "user", "content": user_prompt}],
                    "max_tokens": 1024,
                    "temperature": 0,
                    "seed": SEED,
                },
            }
        )
    return rows


def build_factuality_rows(
    data, prompt
):
    rows = []
    for item in data:
        user_prompt = prompt.format(
            question=item["question"],
            gold_answer=item["gold_answer"],
            pred_answer=item["pred_answer"],
        )
        rows.append(
            {
                "key": sample_key(item),
                "request": {
                    "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                    "generation_config": {
                        "maxOutputTokens": 8192,
                        "temperature": 0,
                        "seed": SEED,
                        "responseMimeType": "application/json",
                        "thinkingConfig": {"thinkingBudget": 0},
                    },
                },
            }
        )
    return rows


def _result_base(item):
    return {
        "q_id": item["q_id"],
        "a_id": item["a_id"],
        "question_type": item.get("gold_question_type", ""),
        "question": item["question"],
        "gold_answer": item["gold_answer"],
        "pred_answer": item["pred_answer"],
    }


def evaluate_helpfulness(
    batch_client,
    data,
    prompt,
    output_path,
    judge_name,
    judge_model_id,
    paths,
):
    input_path, batch_output = batch_file_paths(paths, output_path.stem, f"judge_helpfulness_{judge_name}")
    rows = build_helpfulness_rows(data, prompt, judge_model_id)
    output_rows = batch_client.run(
        rows,
        input_path=input_path,
        output_path=batch_output,
        description=f"PetQA helpfulness: {output_path.stem}",
    )
    answers = {str(row.get("custom_id", "")): _openai_text(row) for row in output_rows}
    results = []
    for item in data:
        key = sample_key(item)
        score, feedback = parse_helpfulness(answers[key])
        results.append(
            {
                **_result_base(item),
                "helpfulness_score": score,
                "helpfulness_feedback": feedback,
            }
        )
    save_json(results, output_path)
    return results


def evaluate_factuality(
    batch_client,
    data,
    prompt,
    output_path,
    judge_name,
    judge_model_id,
    paths,
):
    input_path, batch_output = batch_file_paths(paths, output_path.stem, f"judge_factuality_{judge_name}")
    rows = build_factuality_rows(data, prompt)
    output_rows = batch_client.run(
        rows,
        model=judge_model_id,
        input_path=input_path,
        output_path=batch_output,
        display_name=f"PetQA factuality: {output_path.stem}",
    )
    answers = {
        str(row.get("key", "")): _parse_json_response(_gemini_text(row))
        for row in output_rows
    }
    results = []
    for item in data:
        key = sample_key(item)
        results.append({**_result_base(item), **answers[key]})
    save_json(results, output_path)
    return results


def helpfulness_score(results):
    return statistics.mean(int(item["helpfulness_score"]) for item in results)


def factuality_scores(results):
    precision = []
    recall = []
    f1 = []
    for item in results:
        predicted_count = int(item["facts_count_predicted_answer"])
        predicted_support = int(item["support_predicted_answer"])
        reference_count = int(item["facts_count_reference_answer"])
        reference_support = int(item["support_reference_answer"])
        p = predicted_support / predicted_count if predicted_count else 0.0
        r = reference_support / reference_count if reference_count else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    return statistics.mean(precision), statistics.mean(recall), statistics.mean(f1)


def print_summary(name, scores, digits):
    mean = statistics.mean(scores)
    suffix = (
        f" ± {statistics.stdev(scores):.{digits}f}" if len(scores) > 1 else ""
    )
    print(f"{name}: {mean:.{digits}f}{suffix}")


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate PetQA with LLM judges.")
    parser.add_argument("--modality", choices=SUPPORTED_MODALITIES, required=True)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--strategy", choices=SUPPORTED_STRATEGIES, default="zeroshot")
    parser.add_argument("--language", choices=SUPPORTED_LANGUAGES, default="ko")
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--helpfulness_judge", default="gpt-4o")
    parser.add_argument("--factuality_judge", default="gemini-2.5-flash")
    parser.add_argument("--metric", choices=("helpfulness", "factuality", "all"), default="all")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    spec = get_modality_spec(args.modality)
    args.model_name = args.model_name or spec.default_models["evaluation"]
    endpoint = build_endpoint(args.model_name, args.strategy, args.language, args.top_k, spec)
    paths = output_paths(spec)
    input_paths = collect_answer_paths(paths.generated_answers, endpoint)

    started = time.time()
    helpfulness_runs = []
    factuality_runs = []
    openai_batch = None
    gemini_batch = None

    if args.metric in {"helpfulness", "all"}:
        prompt = load_prompt(evaluation_prompt_path("helpfulness", args.language))
        for input_path in input_paths:
            destination = paths.llm_judge_helpfulness / input_path.name
            if destination.is_file():
                results = load_json(destination)
            else:
                if openai_batch is None:
                    import openai

                    api_key = os.getenv("OPENAI_API_KEY")
                    openai_batch = OpenAIBatchClient(openai.OpenAI(api_key=api_key))
                data = load_json(input_path)
                results = evaluate_helpfulness(
                    openai_batch,
                    data,
                    prompt,
                    destination,
                    args.helpfulness_judge,
                    spec.model_mapping[args.helpfulness_judge],
                    paths,
                )
            helpfulness_runs.append(helpfulness_score(results))

    if args.metric in {"factuality", "all"}:
        prompt = load_prompt(evaluation_prompt_path("factuality", args.language))
        for input_path in input_paths:
            destination = paths.llm_judge_factuality / input_path.name
            if destination.is_file():
                results = load_json(destination)
            else:
                if gemini_batch is None:
                    from google import genai

                    api_key = os.getenv("GOOGLE_API_KEY")
                    gemini_batch = GeminiBatchClient(genai.Client(api_key=api_key))
                data = load_json(input_path)
                results = evaluate_factuality(
                    gemini_batch,
                    data,
                    prompt,
                    destination,
                    args.factuality_judge,
                    spec.model_mapping[args.factuality_judge],
                    paths,
                )
            factuality_runs.append(factuality_scores(results)[2])

    if helpfulness_runs:
        print_summary("Helpfulness", helpfulness_runs, 3)
    if factuality_runs:
        print_summary("Factuality", factuality_runs, 3)
    print(f"TOTAL TIME: {format_time(time.time() - started)}")


if __name__ == "__main__":
    main()
