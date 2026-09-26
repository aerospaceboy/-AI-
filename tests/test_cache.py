from pathlib import Path

from src.utils import read_json, write_json
from src.pipeline import _cached, _mark_done


def test_existing_cache_is_readable(tmp_path: Path) -> None:
    cached = {"has_audio": False, "segments": []}
    path = tmp_path / "raw" / "transcript.json"
    write_json(path, cached)
    assert read_json(path) == cached


def test_cache_signature_invalidates_changed_options(tmp_path: Path) -> None:
    artifact = tmp_path / "result.json"
    artifact.write_text("{}", encoding="utf-8")
    state = {}
    _mark_done(state, "asr", "signature-a")
    assert _cached(state, "asr", "signature-a", artifact)
    assert not _cached(state, "asr", "signature-b", artifact)
