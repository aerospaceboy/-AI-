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


def test_concurrent_chunk_summaries_keep_source_order(tmp_path: Path) -> None:
    import time as _time

    class OrderCheckProvider(LLMProvider):
        def __init__(self) -> None:
            self.final_input: str | None = None

        @property
        def cache_key(self) -> str:
            return "order-check"

        def summarize(self, content: str, instruction: str) -> str:
            if "时间:" not in content:
                self.final_input = content
                return "final"
            # Make later chunks finish first; the final synthesis must still
            # receive the chunk summaries in timeline order.
            _time.sleep(0.3 if "00:00:00" in content else 0.0)
            return "S-" + content.splitlines()[0].split("时间: ")[1][:8]

    timeline = [
        {"start": start, "end": start + 10, "speech": [f"chunk {start}"], "ocr": [], "frames": []}
        for start in (0, 10, 20)
    ]
    provider = OrderCheckProvider()
    generate_summary(timeline, tmp_path / "chunks", provider, max_workers=3)
    assert provider.final_input == "S-00:00:00\n\n---\n\nS-00:00:10\n\n---\n\nS-00:00:20"


def test_empty_chunks_skip_the_llm(tmp_path: Path) -> None:
    class CountingProvider(FakeProvider):
        final_input = ""

        def summarize(self, content: str, instruction: str) -> str:
            self.calls += 1
            if "时间:" not in content:
                self.final_input = content
            return f"summary-{self.calls}"

    timeline = [
        {"start": 0, "end": 10, "speech": [], "ocr": [], "frames": ["frames/a.jpg"]},
        {"start": 10, "end": 20, "speech": ["hello"], "ocr": [], "frames": []},
    ]
    provider = CountingProvider("counting")
    generate_summary(timeline, tmp_path / "chunks", provider)
    assert provider.calls == 2  # one real chunk plus the final synthesis
    assert "没有语音讲解" in provider.final_input
    assert "summary-1" in provider.final_input
