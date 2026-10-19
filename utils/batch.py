import logging
import time
from pathlib import Path

from utils.io import read_jsonl, write_jsonl


class OpenAIBatchClient:
    """Submit, poll, and download a JSONL OpenAI batch."""

    TERMINAL_STATES = {"completed", "failed", "expired", "cancelled"}

    def __init__(
        self,
        client,
        *,
        poll_interval=60,
        sleep=time.sleep,
        logger=None,
    ):
        self.client = client
        self.poll_interval = poll_interval
        self.sleep = sleep
        self.logger = logger or logging.getLogger("petqa")

    def run(
        self,
        rows,
        *,
        input_path,
        output_path,
        endpoint="/v1/chat/completions",
        description="PetQA batch",
    ):
        input_path = Path(input_path)
        output_path = Path(output_path)
        write_jsonl(rows, input_path)
        with input_path.open("rb") as file:
            uploaded = self.client.files.create(file=file, purpose="batch")
        job = self.client.batches.create(
            input_file_id=uploaded.id,
            endpoint=endpoint,
            completion_window="24h",
            metadata={"description": description},
        )
        while True:
            job = self.client.batches.retrieve(job.id)
            if job.status in self.TERMINAL_STATES:
                break
            self.logger.info("OpenAI batch status: %s", job.status)
            self.sleep(self.poll_interval)
        if job.status != "completed":
            raise RuntimeError(f"OpenAI batch ended with status: {job.status}")

        response = self.client.files.content(job.output_file_id)
        content = response.content if hasattr(response, "content") else response
        if isinstance(content, str):
            content = content.encode("utf-8")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(content)
        return read_jsonl(output_path)


class GeminiBatchClient:
    """Submit, poll, and download a JSONL Gemini batch."""

    TERMINAL_STATES = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }

    def __init__(
        self,
        client,
        *,
        poll_interval=60,
        sleep=time.sleep,
        logger=None,
    ):
        self.client = client
        self.poll_interval = poll_interval
        self.sleep = sleep
        self.logger = logger or logging.getLogger("petqa")

    @staticmethod
    def _state_name(job):
        state = job.state
        return state.name if hasattr(state, "name") else str(state)

    def run(
        self,
        rows,
        *,
        model,
        input_path,
        output_path,
        display_name="PetQA batch",
    ):
        from google.genai import types

        input_path = Path(input_path)
        output_path = Path(output_path)
        write_jsonl(rows, input_path)
        uploaded = self.client.files.upload(
            file=input_path,
            config=types.UploadFileConfig(display_name=display_name, mime_type="jsonl"),
        )
        job = self.client.batches.create(
            model=model,
            src=uploaded.name,
            config={"display_name": display_name},
        )
        while True:
            job = self.client.batches.get(name=job.name)
            state = self._state_name(job)
            if state in self.TERMINAL_STATES:
                break
            self.logger.info("Gemini batch status: %s", state)
            self.sleep(self.poll_interval)
        if state != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"Gemini batch ended with status: {state}")

        content = self.client.files.download(file=job.dest.file_name)
        if isinstance(content, bytes):
            text = content.decode("utf-8")
        else:
            text = str(content)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        return read_jsonl(output_path)
