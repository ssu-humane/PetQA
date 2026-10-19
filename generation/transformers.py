import re
from concurrent.futures import ThreadPoolExecutor

from generation.pipeline import (
    GeneratedBatch,
    build_generation_parser,
    run_generation,
)


def _load_image(path):
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB").copy()


def build_generation_payload(
    items,
    context,
    *,
    image_workers=8,
):
    payload = []
    for item in items:
        if context.modality.uses_images:
            content = []
            if item.image_path is not None:
                content.append({"type": "image"})
            content.append({"type": "text", "text": item.prompt})
        else:
            content = item.prompt
        payload.append(
            {
                "item": item,
                "messages": [{"role": "user", "content": content}],
                "image_path": item.image_path,
                "image": None,
            }
        )

    with_images = [entry for entry in payload if entry["image_path"] is not None]
    if with_images:
        workers = min(image_workers, len(with_images))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            images = list(
                executor.map(_load_image, [entry["image_path"] for entry in with_images])
            )
        for entry, image in zip(with_images, images):
            entry["image"] = image
    return payload


def clean_answer(text):
    value = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL).strip()
    return re.sub(
        r"^\s*(assistant|user)\s*\n+", "", value, flags=re.IGNORECASE
    ).strip()


class TransformersBackend:
    def __init__(self):
        self.model = None
        self.processor = None

    def initialize(self, context):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.model = AutoModelForCausalLM.from_pretrained(
            context.model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(
            context.model_path, trust_remote_code=True
        )
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    def _device(self):
        import torch

        return "cuda" if torch.cuda.is_available() else getattr(
            self.model, "device", "cpu"
        )

    def _generate_subset(
        self, subset, context
    ):
        if not subset:
            return []
        import torch

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        chunk_size = 1 if subset[0]["image"] is not None else context.batch_size
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(subset), chunk_size):
                chunk = subset[start : start + chunk_size]
                texts = [
                    self.processor.apply_chat_template(
                        entry["messages"],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for entry in chunk
                ]
                if chunk[0]["image"] is not None:
                    inputs = self.processor(
                        images=[entry["image"] for entry in chunk],
                        text=texts,
                        padding=True,
                        return_tensors="pt",
                    )
                else:
                    inputs = self.processor(
                        text=texts, padding=True, return_tensors="pt"
                    )
                if hasattr(inputs, "to"):
                    inputs = inputs.to(self._device())
                else:
                    inputs = {
                        key: value.to(self._device())
                        if isinstance(value, torch.Tensor)
                        else value
                        for key, value in inputs.items()
                    }
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=context.max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    repetition_penalty=1.05,
                )
                input_ids = inputs["input_ids"]
                trimmed = generated[:, input_ids.shape[1] :]
                predictions.extend(
                    clean_answer(answer)
                    for answer in self.processor.batch_decode(
                        trimmed, skip_special_tokens=True
                    )
                )
        return predictions

    def generate(
        self,
        items,
        context,
    ):
        for start in range(0, len(items), context.batch_size):
            batch = items[start : start + context.batch_size]
            payload = build_generation_payload(
                batch, context, image_workers=context.args.image_workers
            )
            indexed = list(enumerate(payload))
            with_image = [pair for pair in indexed if pair[1]["image"] is not None]
            without_image = [pair for pair in indexed if pair[1]["image"] is None]
            answers = [""] * len(batch)
            for pairs in (with_image, without_image):
                subset_answers = self._generate_subset(
                    [entry for _, entry in pairs], context
                )
                for (index, _), answer in zip(pairs, subset_answers):
                    answers[index] = answer
            yield GeneratedBatch(batch, answers)


def build_parser():
    parser = build_generation_parser(
        "transformers", "Generate PetQA answers with Transformers."
    )
    parser.add_argument("--image_workers", type=int, default=8)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_generation(TransformersBackend(), args, backend_name="transformers")


if __name__ == "__main__":
    main()
