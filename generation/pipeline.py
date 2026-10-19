import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from utils.config import (
    PROJECT_ROOT,
    SUPPORTED_LANGUAGES,
    SUPPORTED_MODALITIES,
    SUPPORTED_STRATEGIES,
    ModalitySpec,
    build_endpoint,
    get_modality_spec,
    repeated_endpoint,
)
from utils.data import (
    format_generation_prompt,
    load_generation_data,
    load_generation_prompt,
    load_retrieved_samples,
    resolve_image_path,
    resolve_retrieved_paragraphs_path,
)
from utils.io import (
    OutputPaths,
    answer_path,
    load_results,
    output_paths,
    save_json,
    set_seed,
    setup_logging,
)


@dataclass(frozen=True)
class PreparedGenerationItem:
    source: dict
    prompt: str
    image_path: Path | None = None


@dataclass
class GenerationContext:
    args: argparse.Namespace
    backend_name: str
    modality: ModalitySpec
    model_name: str
    model_path: str
    strategy: str
    language: str
    top_k: int
    batch_size: int
    max_tokens: int
    output_paths: OutputPaths
    project_root: Path
    base_endpoint: str
    endpoint: str
    logger: logging.Logger


@dataclass(frozen=True)
class GeneratedBatch:
    items: list
    answers: list


class GenerationBackend(Protocol):
    def initialize(self, context):
        """Load the model or remote client."""

    def generate(self, items, context):
        """Yield completed batches in the same order as ``items``."""


def build_generation_parser(backend_name, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--modality", choices=SUPPORTED_MODALITIES, required=True)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--strategy", choices=SUPPORTED_STRATEGIES, default="zeroshot")
    parser.add_argument("--language", choices=SUPPORTED_LANGUAGES, default="ko")
    parser.add_argument("--test_data_path", default=None)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--retrieved_paragraphs_path", default=None)
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=512)
    return parser


def _resolved_context(
    args,
    backend_name,
    project_root,
):
    spec = get_modality_spec(args.modality)
    model_name = args.model_name or spec.default_models[backend_name]
    top_k = int(args.top_k)
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else spec.default_batch_sizes.get(backend_name, 100)
    )
    args.model_name = model_name
    args.batch_size = batch_size
    base_endpoint = build_endpoint(
        model_name,
        args.strategy,
        args.language,
        top_k,
        spec,
    )
    root = Path(project_root).resolve()
    return GenerationContext(
        args=args,
        backend_name=backend_name,
        modality=spec,
        model_name=model_name,
        model_path=spec.model_mapping[model_name],
        strategy=args.strategy,
        language=args.language,
        top_k=top_k,
        batch_size=batch_size,
        max_tokens=int(args.max_tokens),
        output_paths=output_paths(spec, root),
        project_root=root,
        base_endpoint=base_endpoint,
        endpoint=base_endpoint,
        logger=setup_logging(),
    )


def prepare_generation_items(
    data,
    context,
):
    prompt = load_generation_prompt(
        context.strategy, context.language, context.project_root
    )
    retrieved_samples = None
    if context.strategy == "rag":
        retrieved_path = resolve_retrieved_paragraphs_path(
            context.modality,
            context.args.retrieved_paragraphs_path,
            context.project_root,
        )
        retrieved_samples = load_retrieved_samples(retrieved_path)

    image_dir = context.modality.data_dir(context.project_root) / "images"
    prepared = []
    for item in data:
        rendered_prompt = format_generation_prompt(
            prompt,
            item,
            context.strategy,
            retrieved_samples,
            context.top_k,
        )
        image_path = (
            resolve_image_path(item, image_dir) if context.modality.uses_images else None
        )
        prepared.append(
            PreparedGenerationItem(
                source=dict(item), prompt=rendered_prompt, image_path=image_path
            )
        )
    return prepared


def _append_generated_batch(
    results,
    generated,
):
    for item, answer in zip(generated.items, generated.answers):
        row = dict(item.source)
        row.update(
            pred_answer="" if answer is None else str(answer).strip(),
            pred_question_type=None,
            pred_disease=None,
        )
        results.append(row)


def run_generation(
    backend,
    args,
    *,
    backend_name,
    project_root=PROJECT_ROOT,
):
    """Run generation, checkpointing every yielded batch and resuming by row."""

    context = _resolved_context(args, backend_name, project_root)
    set_seed(42)
    data = load_generation_data(
        context.modality,
        context.language,
        context.args.test_data_path,
        context.project_root,
    )
    prepared = prepare_generation_items(data, context)
    backend.initialize(context)

    written_paths = []
    for run_index in range(context.args.num_repeats):
        context.endpoint = repeated_endpoint(context.base_endpoint, run_index)
        destination = answer_path(context.output_paths, context.endpoint)
        written_paths.append(destination)
        results, start_index = load_results(destination)
        context.logger.info("ENDPOINT: %s", context.endpoint)
        if start_index == len(prepared):
            context.logger.info("Completed result already exists; skipping %s", destination)
            continue

        remaining = prepared[start_index:]
        for generated in backend.generate(remaining, context):
            _append_generated_batch(results, generated)
            save_json(results, destination)
    context.logger.info("Answer generation completed.")
    return written_paths
