import argparse
import re
import statistics
import time

from utils.config import (
    SUPPORTED_LANGUAGES,
    SUPPORTED_MODALITIES,
    SUPPORTED_STRATEGIES,
    build_endpoint,
    get_modality_spec,
)
from utils.io import (
    collect_answer_paths,
    format_time,
    load_json,
    output_paths,
    save_json,
    set_seed,
)


def normalize_text(value):
    return "" if value is None else str(value).strip()


def tokenize_for_rouge(values, language):
    if language == "ko":
        from konlpy.tag import Okt

        okt = Okt()
        return [" ".join(okt.morphs(normalize_text(value))) for value in values]
    return [
        " ".join(re.findall(r"\w+|[^\w\s]", normalize_text(value).lower()))
        for value in values
    ]


def compute_rouge(predictions, references, language):
    from rouge import Rouge

    predictions = tokenize_for_rouge(predictions, language)
    references = tokenize_for_rouge(references, language)
    scorer = Rouge()
    scores = [
        scorer.get_scores(prediction, reference)[0]["rouge-l"]["f"]
        if prediction and reference
        else 0.0
        for prediction, reference in zip(predictions, references)
    ]
    return statistics.mean(scores) if scores else 0.0, scores


def compute_bertscore(predictions, references, language):
    from bert_score import score

    predictions = [normalize_text(value) for value in predictions]
    references = [normalize_text(value) for value in references]
    if not predictions:
        return 0.0, []
    options = (
        {"lang": "ko"}
        if language == "ko"
        else {"model_type": "bert-base-multilingual-cased", "num_layers": 9}
    )
    scores = score(predictions, references, **options)[2].tolist()
    return statistics.mean(scores), scores


def evaluate_run(data, output_path, language):
    predictions = [item["pred_answer"] for item in data]
    references = [item["gold_answer"] for item in data]
    rouge, rouge_scores = compute_rouge(predictions, references, language)
    bertscore, bert_scores = compute_bertscore(predictions, references, language)
    for item, rouge_score, bert_score in zip(data, rouge_scores, bert_scores):
        item.update(rouge=rouge_score, bertscore=bert_score)
    save_json(data, output_path)
    return rouge, bertscore


def print_summary(name, scores):
    mean = statistics.mean(scores)
    suffix = f" ± {statistics.stdev(scores):.3f}" if len(scores) > 1 else ""
    print(f"{name}: {mean:.3f}{suffix}")


def build_parser():
    parser = argparse.ArgumentParser(description="Compute automatic PetQA metrics.")
    parser.add_argument("--modality", choices=SUPPORTED_MODALITIES, required=True)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--strategy", choices=SUPPORTED_STRATEGIES, default="zeroshot")
    parser.add_argument("--language", choices=SUPPORTED_LANGUAGES, default="ko")
    parser.add_argument("--top_k", type=int, default=6)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    spec = get_modality_spec(args.modality)
    args.model_name = args.model_name or spec.default_models["evaluation"]
    endpoint = build_endpoint(args.model_name, args.strategy, args.language, args.top_k, spec)
    paths = output_paths(spec)
    input_paths = collect_answer_paths(paths.generated_answers, endpoint)

    set_seed(42)
    started = time.time()
    rouge_runs = []
    bertscore_runs = []
    for input_path in input_paths:
        data = load_json(input_path)
        rouge, bertscore = evaluate_run(data, paths.automatic_metrics / input_path.name, args.language)
        rouge_runs.append(rouge)
        bertscore_runs.append(bertscore)
    if rouge_runs:
        print_summary("ROUGE-L", rouge_runs)
        print_summary("BERTScore", bertscore_runs)
    print(f"TOTAL TIME: {format_time(time.time() - started)}")


if __name__ == "__main__":
    main()
