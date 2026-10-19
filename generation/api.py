import base64
import os
from pathlib import Path

from generation.pipeline import (
    GeneratedBatch,
    build_generation_parser,
    run_generation,
)
from utils.batch import GeminiBatchClient, OpenAIBatchClient
from utils.data import image_mime_type, sample_key
from utils.io import batch_file_paths


def parse_model_family(model_name):
    return "openai" if model_name.startswith("gpt") else "gemini"


def encode_image(image_path):
    return base64.b64encode(Path(image_path).read_bytes()).decode("ascii")


def build_openai_payload(
    model_id,
    user_prompt,
    image_path=None,
    *,
    max_tokens=512,
):
    if image_path is None:
        content = user_prompt
    else:
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        f"data:{image_mime_type(image_path)};base64,"
                        f"{encode_image(image_path)}"
                    ),
                    "detail": "low",
                },
            },
            {"type": "text", "text": user_prompt},
        ]
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 42,
    }


def build_google_payload(
    model_name,
    user_prompt,
    image_path=None,
    *,
    max_tokens=512,
):
    parts = []
    if image_path is not None:
        parts.append(
            {
                "inlineData": {
                    "mimeType": image_mime_type(image_path),
                    "data": encode_image(image_path),
                }
            }
        )
    parts.append({"text": user_prompt})
    return {
        "contents": [{"role": "user", "parts": parts}],
        "generation_config": {
            "maxOutputTokens": max_tokens,
            "temperature": 0,
            "seed": 42,
            "thinkingConfig": {
                "thinkingBudget": 128 if model_name == "gemini-2.5-pro" else 0
            },
        },
    }


def parse_openai_answers(rows):
    return {
        str(row["custom_id"]): str(
            row["response"]["body"]["choices"][0]["message"]["content"]
        )
        for row in rows
    }


def google_batch_text(row):
    parts = row["response"]["candidates"][0]["content"]["parts"]
    return "\n".join(str(part.get("text", "")) for part in parts).strip()


class APIBackend:
    def __init__(self):
        self.family = ""
        self.batch_client: Any = None

    def initialize(self, context):
        self.family = parse_model_family(context.model_name)
        if self.family == "openai":
            import openai

            api_key = os.getenv("OPENAI_API_KEY")
            self.batch_client = OpenAIBatchClient(
                openai.OpenAI(api_key=api_key), logger=context.logger
            )
        else:
            from google import genai

            api_key = os.getenv("GOOGLE_API_KEY")
            self.batch_client = GeminiBatchClient(
                genai.Client(api_key=api_key), logger=context.logger
            )

    def generate(
        self,
        items,
        context,
    ):
        input_path, output_path = batch_file_paths(
            context.output_paths, context.endpoint, "generation"
        )
        if self.family == "openai":
            rows = [
                {
                    "custom_id": sample_key(item.source),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": build_openai_payload(
                        context.model_path,
                        item.prompt,
                        item.image_path,
                        max_tokens=context.max_tokens,
                    ),
                }
                for item in items
            ]
            output_rows = self.batch_client.run(
                rows,
                input_path=input_path,
                output_path=output_path,
                description=f"PetQA generation: {context.endpoint}",
            )
            answer_map = parse_openai_answers(output_rows)
        else:
            rows = [
                {
                    "key": sample_key(item.source),
                    "request": build_google_payload(
                        context.model_name,
                        item.prompt,
                        item.image_path,
                        max_tokens=context.max_tokens,
                    ),
                }
                for item in items
            ]
            output_rows = self.batch_client.run(
                rows,
                model=context.model_path,
                input_path=input_path,
                output_path=output_path,
                display_name=f"PetQA generation: {context.endpoint}",
            )
            answer_map = {
                str(row["key"]): google_batch_text(row) for row in output_rows
            }
        answers = [answer_map[sample_key(item.source)] for item in items]
        yield GeneratedBatch(items, answers)


def build_parser():
    return build_generation_parser("api", "Generate PetQA answers with a batch API.")


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_generation(APIBackend(), args, backend_name="api")


if __name__ == "__main__":
    main()
