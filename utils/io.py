import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path

from utils.config import PROJECT_ROOT, get_modality_spec


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    batches: Path
    results: Path
    generated_answers: Path
    automatic_metrics: Path
    llm_judge_helpfulness: Path
    llm_judge_factuality: Path
    predicted_diseases: Path


def output_paths(
    modality,
    project_root=PROJECT_ROOT,
):
    spec = get_modality_spec(modality)
    root = Path(project_root) / "outputs" / spec.name
    results = root / "results"
    return OutputPaths(
        root=root,
        batches=root / "batches",
        results=results,
        generated_answers=results / "generated_answers",
        automatic_metrics=results / "automatic_metrics",
        llm_judge_helpfulness=results / "llm_judge_helpfulness",
        llm_judge_factuality=results / "llm_judge_factuality",
        predicted_diseases=results / "predicted_diseases",
    )


def load_json(path):
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, path):
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def write_jsonl(rows, path):
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_results(path):
    file_path = Path(path)
    if not file_path.exists():
        return [], 0
    results = load_json(file_path)
    return results, len(results)


def answer_path(paths, endpoint):
    return paths.generated_answers / f"{endpoint}.json"


def collect_answer_paths(directory, endpoint):
    directory = Path(directory)
    candidates = []
    base = directory / f"{endpoint}.json"
    if base.is_file():
        candidates.append((0, base))
    pattern = re.compile(rf"^{re.escape(endpoint)}_(\d+)\.json$")
    if directory.is_dir():
        for path in directory.glob(f"{endpoint}_*.json"):
            match = pattern.match(path.name)
            if match:
                candidates.append((int(match.group(1)), path))
    return [path for _, path in sorted(candidates, key=lambda pair: pair[0])]


def batch_file_paths(
    paths, endpoint, purpose
):
    stem = f"{purpose}_{endpoint}"
    return paths.batches / f"{stem}_input.jsonl", paths.batches / f"{stem}_output.jsonl"


def set_seed(seed=42):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed seeds every device and does so lazily.  Calling
    # torch.cuda.is_available() here would initialize the CUDA driver and make
    # subsequent fork-based multiprocessing unsafe.
    torch.manual_seed(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("google_genai.models").setLevel(logging.WARNING)
    return logging.getLogger("petqa")


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining = int(seconds % 60)
    if hours:
        return f"{hours}h {minutes}m {remaining}s"
    if minutes:
        return f"{minutes}m {remaining}s"
    return f"{remaining}s"
