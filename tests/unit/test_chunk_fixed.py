from minutes.ingest.chunk import fixed_chunks


def test_windows_are_1200_with_200_overlap() -> None:
    assert fixed_chunks("x" * 2500) == [(0, 1200), (1000, 2200), (2000, 2500)]


def test_short_text_gives_one_chunk() -> None:
    assert fixed_chunks("Motion carried.") == [(0, 15)]


def test_blank_text_gives_no_chunk() -> None:
    assert fixed_chunks("") == []
    assert fixed_chunks(" \n\n ") == []
