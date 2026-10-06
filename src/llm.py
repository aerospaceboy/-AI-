from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path


# Server-side hiccups and rate limits are worth retrying; client errors are not.
RETRYABLE_HTTP_CODES = frozenset({408, 429, 500, 502, 503, 504})
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0

TRUTHFULNESS_RULES = """Only use information found in the provided transcript, OCR results,
timestamps and frame metadata. Never invent code, commands, parameters, file names,
errors, solutions, project status, or results. If information cannot be verified, mark
it as [未确认]. Preserve OCR text verbatim and cite timestamps for technical facts."""


class LLMProvider(ABC):
    @property
    def cache_key(self) -> str:
        """Stable, secret-free identity used to invalidate generated summaries."""
        return type(self).__name__

    @abstractmethod
    def summarize(self, content: str, instruction: str) -> str:
        raise NotImplementedError


class NoLLMProvider(LLMProvider):
    @property
    def cache_key(self) -> str:
        return "none"

    def summarize(self, content: str, instruction: str) -> str:
        return ""


class OpenAICompatibleProvider(LLMProvider):
    # Reasoning models can spend minutes on a long OCR-heavy chunk; 180s was
    # measured to be too short for DashScope qwen3.8-flash summarization.
    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: int = 600) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or ""
        self.timeout = timeout

    @property
    def cache_key(self) -> str:
        return f"openai-compatible:{self.base_url}:{self.model}"

    def _complete(self, body: bytes) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_error: Exception | None = None
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions", data=body, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return payload["choices"][0]["message"]["content"].strip()
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in RETRYABLE_HTTP_CODES:
                    break
            except OSError as exc:
                # Covers URLError, read timeouts and dropped connections.
                last_error = exc
            except (KeyError, IndexError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"OpenAI-compatible request failed: {exc}") from exc
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
        raise RuntimeError(f"OpenAI-compatible request failed: {last_error}")

    def summarize(self, content: str, instruction: str) -> str:
        body = json.dumps({
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": TRUTHFULNESS_RULES},
                {"role": "user", "content": f"{instruction}\n\nSOURCE MATERIAL:\n{content}"},
            ],
        }, ensure_ascii=False).encode("utf-8")
        return self._complete(body)

    def describe_image(self, image_path: Path, instruction: str) -> str:
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        body = json.dumps({
            "model": self.model,
            "temperature": 0.1,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                    {"type": "text", "text": instruction},
                ],
            }],
        }, ensure_ascii=False).encode("utf-8")
        return self._complete(body)


def provider_from_options(
    disabled: bool,
    base_url: str | None,
    model: str | None,
    api_key: str | None,
) -> LLMProvider:
    if disabled:
        return NoLLMProvider()
    url = base_url or os.getenv("VIDEO2AI_BASE_URL") or "https://api.openai.com/v1"
    selected_model = model or os.getenv("VIDEO2AI_MODEL")
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not selected_model:
        return NoLLMProvider()
    return OpenAICompatibleProvider(url, selected_model, key)
