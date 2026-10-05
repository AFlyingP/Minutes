from minutes.ingest.ocr import needs_ocr


def test_image_page_with_little_text_needs_ocr() -> None:
    assert needs_ocr("Page 3", 0.95)


def test_text_page_with_logo_does_not_need_ocr() -> None:
    assert not needs_ocr("a" * 500, 0.1)


def test_little_text_without_large_image_does_not_need_ocr() -> None:
    assert not needs_ocr("a" * 10, 0.3)
    assert not needs_ocr("", 0.0)


def test_threshold_is_50_characters_and_half_page() -> None:
    assert needs_ocr("a" * 49, 0.5)
    assert not needs_ocr("a" * 50, 0.5)
    assert not needs_ocr("a" * 49, 0.49)
    assert needs_ocr(" \t\n" + "a " * 49, 0.5)
    assert not needs_ocr(" \t\n" + "a " * 50, 0.5)


def test_garbled_text_needs_ocr() -> None:
    assert needs_ocr("\ufffd" * 20 + "a" * 80, 0.0)
    assert not needs_ocr("\ufffd" * 19 + "a" * 81, 0.0)
    assert needs_ocr("\ue000" * 10 + "\uf8ff" * 10 + "a" * 80 + " \t\n", 0.0)
    assert not needs_ocr("\udfff" * 20 + "\uf900" * 20 + "a" * 60, 0.0)
