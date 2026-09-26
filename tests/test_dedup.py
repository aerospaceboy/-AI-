from src.ocr import is_duplicate_ocr


def test_exact_ocr_duplicate() -> None:
    lines = ["File Edit View", "parameter DATA_WIDTH = 8"]
    assert is_duplicate_ocr(lines, lines)


def test_small_code_change_is_kept() -> None:
    before = ["parameter DATA_WIDTH = 8", "parameter DEPTH = 16"]
    after = ["parameter DATA_WIDTH = 9", "parameter DEPTH = 16"]
    assert not is_duplicate_ocr(before, after)
