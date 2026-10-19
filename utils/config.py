from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_MODALITIES = ("text", "multimodal")
SUPPORTED_LANGUAGES = ("ko", "en", "de", "zh", "ar", "id")
SUPPORTED_STRATEGIES = ("zeroshot", "rag")
_API_MODELS = {
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4o": "gpt-4o-2024-08-06",
    "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-2.5-pro": "gemini-2.5-pro",
}

_TEXT_MODELS = {
    **_API_MODELS,
    "hcx-seed-1.5b": "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
    "hcx-seed-vision-3b": "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B",
    "a.x-4.0-light": "skt/A.X-4.0-Light",
    "a.x-4.0-vl-light": "skt/A.X-4.0-VL-Light",
    "qwen-3-8b": "Qwen/Qwen3-8B",
    "qwen-3-32b": "Qwen/Qwen3-32B",
    "qwen-3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen-3-vl-32b": "Qwen/Qwen3-VL-32B-Instruct",
    "gemma-2-9b": "google/gemma-2-9b-it",
    "gemma-2-27b": "google/gemma-2-27b-it",
    "gemma-3-12b": "google/gemma-3-12b-it",
    "gemma-3-27b": "google/gemma-3-27b-it",
    "medgemma-27b-text": "google/medgemma-27b-text-it",
    "medgemma-27b": "google/medgemma-27b-it",
}

_MULTIMODAL_MODELS = {
    **_API_MODELS,
    "gemma-3-12b": "google/gemma-3-12b-it",
    "gemma-3-27b": "google/gemma-3-27b-it",
    "qwen-3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen-3-vl-32b": "Qwen/Qwen3-VL-32B-Instruct",
    "hcx-seed-vision-3b": "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B",
    "a.x-4.0-vl-light": "skt/A.X-4.0-VL-Light",
    "medgemma-27b": "google/medgemma-27b-text-it",
}


@dataclass(frozen=True)
class ModalitySpec:
    """All modality-specific choices needed by the shared pipeline."""

    name: str
    data_directory: str
    endpoint_suffix: str
    uses_images: bool
    model_mapping: Mapping[str, str]
    default_models: Mapping[str, str]
    default_batch_sizes: Mapping[str, int]
    default_tensor_parallel_size: int
    default_gpu_memory_utilization: float

    def data_dir(self, project_root=PROJECT_ROOT):
        return Path(project_root) / "data" / self.data_directory


MODALITY_SPECS = MappingProxyType(
    {
        "text": ModalitySpec(
            name="text",
            data_directory="TEXT_ONLY",
            endpoint_suffix="",
            uses_images=False,
            model_mapping=MappingProxyType(_TEXT_MODELS),
            default_models=MappingProxyType(
                {
                    "vllm": "qwen-3-8b",
                    "transformers": "a.x-4.0-vl-light",
                    "api": "gpt-4.1-mini",
                    "evaluation": "gpt-4.1-mini",
                    "disease": "qwen-3-8b",
                    "disease_judge": "qwen-3-32b",
                }
            ),
            default_batch_sizes=MappingProxyType(
                {"vllm": 100, "transformers": 50, "disease": 200}
            ),
            default_tensor_parallel_size=2,
            default_gpu_memory_utilization=0.8,
        ),
        "multimodal": ModalitySpec(
            name="multimodal",
            data_directory="TEXT_w_IMAGE",
            endpoint_suffix="_multimodal",
            uses_images=True,
            model_mapping=MappingProxyType(_MULTIMODAL_MODELS),
            default_models=MappingProxyType(
                {
                    "vllm": "qwen-3-vl-8b",
                    "transformers": "a.x-4.0-vl-light",
                    "api": "gpt-4.1",
                    "evaluation": "qwen-3-vl-8b",
                    "disease": "gpt-4.1",
                    "disease_judge": "qwen-3-32b",
                }
            ),
            default_batch_sizes=MappingProxyType(
                {"vllm": 200, "transformers": 5, "disease": 200}
            ),
            default_tensor_parallel_size=1,
            default_gpu_memory_utilization=0.9,
        ),
    }
)


def get_modality_spec(modality):
    if isinstance(modality, ModalitySpec):
        return modality
    return MODALITY_SPECS[modality]


def build_endpoint(
    model_name,
    strategy,
    language,
    top_k,
    modality,
):
    """Build the single canonical endpoint name used by every command."""

    spec = get_modality_spec(modality)
    if language != "ko" and strategy != "zeroshot":
        raise ValueError("Non-Korean endpoints support only the zeroshot strategy.")
    tag = "zeroshot" if strategy == "zeroshot" else f"RAG-{top_k}"
    endpoint = f"{model_name}_{tag}{spec.endpoint_suffix}"
    if language != "ko":
        endpoint += f"_{language}"
    return endpoint


def repeated_endpoint(endpoint, run_index):
    return endpoint if run_index == 0 else f"{endpoint}_{run_index}"
