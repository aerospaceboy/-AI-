from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod


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
    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: int = 180) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or ""
        self.timeout = timeout

    @property
    def cache_key(self) -> str:
        return f"openai-compatible:{self.base_url}:{self.model}"

    def summarize(self, content: str, instruction: str) -> str:
        body = json.dumps({
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": TRUTHFULNESS_RULES},
                {"role": "user", "content": f"{instruction}\n\nSOURCE MATERIAL:\n{content}"},
            ],
        }, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return payload["choices"][0]["message"]["content"].strip()
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"OpenAI-compatible request failed: {exc}") from exc


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
