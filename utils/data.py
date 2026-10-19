import mimetypes
import string
from collections.abc import Mapping
from pathlib import Path

from utils.config import PROJECT_ROOT
from utils.io import load_json


DATA_ACCESS_GUIDANCE = (
    "Approved data is required. See data/instructions.md for the application process."
)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def load_prompt(path):
    prompt_path = Path(path)
    return prompt_path.read_text(encoding="utf-8")


def resolve_test_data_path(
    spec,
    language="ko",
    test_data_path=None,
    project_root=PROJECT_ROOT,
):
    if test_data_path is not None:
        path = Path(test_data_path).expanduser().resolve()
    else:
        filename = "test.json" if language == "ko" else f"test_translated_{language}.json"
        path = spec.data_dir(project_root) / filename
    return path


def prepare_test_data_for_language(
    data, language="ko"
):
    if language == "ko":
        return [dict(item) for item in data]

    question_key = f"question_{language}"
    answer_key = f"gold_answer_{language}"
    normalized = []
    missing = []
    for index, item in enumerate(data):
        question = item.get(question_key)
        answer = item.get(answer_key)
        if not isinstance(question, str) or not question.strip():
            missing.append(str(item.get("q_id", index)))
            continue
        if not isinstance(answer, str) or not answer.strip():
            missing.append(str(item.get("q_id", index)))
            continue
        row = dict(item)
        row["question"] = question
        row["gold_answer"] = answer
        normalized.append(row)
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"Samples missing {question_key}/{answer_key} translation fields: {preview}"
        )
    return normalized


def load_generation_data(
    spec,
    language="ko",
    test_data_path=None,
    project_root=PROJECT_ROOT,
):
    path = resolve_test_data_path(spec, language, test_data_path, project_root)
    data = load_json(path)
    return prepare_test_data_for_language(data, language)


def generation_prompt_path(
    strategy,
    language,
    project_root=PROJECT_ROOT,
):
    prompt_dir = Path(project_root) / "prompts"
    if strategy == "rag":
        return prompt_dir / "rag.txt"
    return prompt_dir / "zeroshot" / f"{language}.txt"


def evaluation_prompt_path(
    metric,
    language,
    project_root=PROJECT_ROOT,
):
    suffix = "" if language == "ko" else "_en"
    return Path(project_root) / "prompts" / f"{metric}{suffix}.txt"


def disease_prompt_path(project_root=PROJECT_ROOT):
    return Path(project_root) / "prompts" / "clinical_condition.txt"


def load_generation_prompt(
    strategy,
    language,
    project_root=PROJECT_ROOT,
):
    path = generation_prompt_path(strategy, language, project_root)
    prompt = load_prompt(path)
    fields = {
        field_name
        for _, field_name, _, _ in string.Formatter().parse(prompt)
        if field_name
    }
    required = {"question"}
    if strategy == "rag":
        required.add("paragraphs")
    missing = sorted(required - fields)
    if missing:
        placeholders = ", ".join(f"{{{name}}}" for name in missing)
        raise ValueError(f"Prompt placeholders missing in {path}: {placeholders}")
    return prompt


def sample_key(item):
    return f"{item.get('q_id')}_{item.get('a_id')}"


def resolve_retrieved_paragraphs_path(
    spec,
    explicit_path=None,
    project_root=PROJECT_ROOT,
):
    path = (
        Path(explicit_path).expanduser().resolve()
        if explicit_path is not None
        else spec.data_dir(project_root) / "retrieved_paragraphs.json"
    )
    return path


def load_retrieved_samples(path):
    samples = load_json(path)
    return {sample_key(item): item for item in samples}


def build_rag_paragraphs(
    item,
    retrieved_samples,
    top_k,
):
    key = sample_key(item)
    sample = retrieved_samples.get(key)
    if sample is None:
        raise ValueError(f"RAG document missing for sample {key}")
    documents = sample.get("retrieved_paragraphs")
    if not isinstance(documents, list) or not documents:
        raise ValueError(f"RAG documents are empty for sample {key}")

    paragraphs = []
    for index, document in enumerate(documents[:top_k], start=1):
        content = document.get("content") if isinstance(document, Mapping) else None
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"RAG document content missing for sample {key}")
        paragraphs.append(f"[문서 {index}]\n{content}")
    return "\n\n".join(paragraphs)


def format_generation_prompt(
    prompt,
    item,
    strategy="zeroshot",
    retrieved_samples=None,
    top_k=6,
):
    values = {"question": item["question"]}
    if strategy == "rag":
        values["paragraphs"] = build_rag_paragraphs(item, retrieved_samples, top_k)
    return prompt.format(**values)


def resolve_image_path(item, image_dir):
    image_id = item.get("image")
    if image_id is None or str(image_id).strip() == "":
        return None
    image_dir = Path(image_dir)
    raw_name = str(image_id)
    candidates = [image_dir / raw_name] if Path(raw_name).suffix.lower() in IMAGE_EXTENSIONS else []
    candidates.extend(image_dir / f"{raw_name}{extension}" for extension in IMAGE_EXTENSIONS)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Image not found for sample {sample_key(item)} (image={image_id}) in {image_dir}. "
        f"{DATA_ACCESS_GUIDANCE}"
    )


def image_mime_type(path):
    return mimetypes.guess_type(str(path))[0] or "application/octet-stream"
