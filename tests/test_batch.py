from pathlib import Path

import pytest

import src.cli as cli


def test_batch_directory_processes_each_video(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "b.mkv").write_bytes(b"x")
    (tmp_path / "a.mp4").write_bytes(b"x")
    (tmp_path / "note.txt").write_bytes(b"x")
    (tmp_path / "sub").mkdir()

    calls: list[str] = []

    def fake_run(options) -> Path:
        calls.append(options.video.name)
        assert options.output == options.video.with_name(options.video.stem + "_ai")
        if options.video.name == "b.mkv":
            raise RuntimeError("boom")
        return options.output

    monkeypatch.setattr(cli, "run", fake_run)
    assert cli.main([str(tmp_path), "--no-llm"]) == 1
    assert calls == ["a.mp4", "b.mkv"]


def test_single_video_keeps_explicit_output(monkeypatch, tmp_path: Path) -> None:
    video = tmp_path / "one.mp4"
    video.write_bytes(b"x")
    output = tmp_path / "custom"

    def fake_run(options) -> Path:
        assert options.video == video
        assert options.output == output
        return output

    monkeypatch.setattr(cli, "run", fake_run)
    assert cli.main([str(video), "--output", str(output), "--no-llm"]) == 0


def test_batch_directory_rejects_output(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "a.mp4").write_bytes(b"x")
    with pytest.raises(SystemExit, match="--output"):
        cli.main([str(tmp_path), "--output", str(tmp_path / "out"), "--no-llm"])


def test_batch_directory_requires_video_files(tmp_path: Path) -> None:
    (tmp_path / "readme.txt").write_bytes(b"x")
    with pytest.raises(SystemExit, match="No video files"):
        cli.main([str(tmp_path), "--no-llm"])
