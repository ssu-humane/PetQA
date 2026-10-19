import os

# vLLM defaults to ``fork`` when used as a library.  ``spawn`` is CUDA-safe,
# and this module has the required ``if __name__ == "__main__"`` guard.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from generation.pipeline import (
    GeneratedBatch,
    build_generation_parser,
    run_generation,
)


THINKING_MODEL_PREFIXES = ("qwen-3",)


def supports_thinking(model_name):
    return model_name.startswith(THINKING_MODEL_PREFIXES)


def render_model_inputs(
    items,
    tokenizer,
    context,
):
    rendered = []
    chat_options = {"tokenize": False, "add_generation_prompt": True}
    if supports_thinking(context.model_name):
        chat_options["enable_thinking"] = False
    for item in items:
        if context.modality.uses_images:
            content = []
            if item.image_path is not None:
                content.append({"type": "image"})
            content.append({"type": "text", "text": item.prompt})
        else:
            content = item.prompt
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], **chat_options
        )
        rendered.append((prompt, item.image_path))
    return rendered


def build_model_inputs(
    items,
    tokenizer,
    context,
):
    rendered = render_model_inputs(items, tokenizer, context)
    if not context.modality.uses_images:
        return [prompt for prompt, _ in rendered]

    from PIL import Image

    model_inputs = []
    for prompt, image_path in rendered:
        model_input = {"prompt": prompt}
        if image_path is not None:
            with Image.open(image_path) as image:
                model_input["multi_modal_data"] = {
                    "image": image.convert("RGB").copy()
                }
        model_inputs.append(model_input)
    return model_inputs


class VLLMBackend:
    def __init__(self):
        self.llm = None
        self.tokenizer = None
        self.sampling_params = None

    def initialize(self, context):
        import torch
        from vllm import LLM, SamplingParams

        if "gemma-2" in context.model_name:
            os.environ["VLLM_FLASH_ATTN_VERSION"] = "2"
        tensor_parallel_size = (
            context.args.tensor_parallel_size
            if context.args.tensor_parallel_size is not None
            else context.modality.default_tensor_parallel_size
        )
        gpu_memory_utilization = (
            context.args.gpu_memory_utilization
            if context.args.gpu_memory_utilization is not None
            else context.modality.default_gpu_memory_utilization
        )
        options = {
            "model": context.model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "trust_remote_code": True,
            "dtype": torch.bfloat16,
            "max_model_len": context.args.max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
        }
        if context.modality.uses_images:
            options["limit_mm_per_prompt"] = {"image": 1, "video": 0}
        self.llm = LLM(**options)
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=context.max_tokens,
            seed=42,
            repetition_penalty=1.05,
        )

    def generate(
        self,
        items,
        context,
    ):
        for start in range(0, len(items), context.batch_size):
            batch = items[start : start + context.batch_size]
            model_inputs = build_model_inputs(batch, self.tokenizer, context)
            outputs = self.llm.generate(
                model_inputs,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            answers = [output.outputs[0].text for output in outputs]
            yield GeneratedBatch(batch, answers)


def build_parser():
    parser = build_generation_parser("vllm", "Generate PetQA answers with vLLM.")
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=None)
    parser.add_argument("--max_model_len", type=int, default=8192)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_generation(VLLMBackend(), args, backend_name="vllm")


if __name__ == "__main__":
    main()
