from src.utils import format_timestamp


def test_time_format() -> None:
    assert format_timestamp(0) == "00:00:00"
    assert format_timestamp(3725.9) == "01:02:05"
    assert format_timestamp(-1) == "00:00:00"
    assert format_timestamp(59.9999, include_ms=True) == "00:01:00.000"
