from pathlib import Path

from src.llm import LLMProvider
from src.summarizer import generate_summary


class FakeProvider(LLMProvider):
    def __init__(self, identity: str) -> None:
        self.identity = identity
        self.calls = 0

    @property
    def cache_key(self) -> str:
        return self.identity

    def summarize(self, content: str, instruction: str) -> str:
        self.calls += 1
        return f"summary-{self.calls}"


def _timeline(text: str) -> list[dict]:
    return [{"start": 0, "end": 10, "speech": [text], "ocr": [], "frames": []}]


def test_chunk_summary_cache_uses_content_and_provider(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks"

    first = FakeProvider("model-a")
    generate_summary(_timeline("alpha"), chunks, first)
    assert first.calls == 2  # one chunk plus the final synthesis

    same = FakeProvider("model-a")
    generate_summary(_timeline("alpha"), chunks, same)
    assert same.calls == 1  # chunk reused; final synthesis still runs

    changed_content = FakeProvider("model-a")
    generate_summary(_timeline("beta"), chunks, changed_content)
    assert changed_content.calls == 2

    changed_provider = FakeProvider("model-b")
    generate_summary(_timeline("beta"), chunks, changed_provider)
    assert changed_provider.calls == 2
