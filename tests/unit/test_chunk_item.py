from datetime import date

from minutes.ingest.chunk import header_for, item_chunks


def test_lines_pack_up_to_1200_characters() -> None:
    text = "a" * 500 + "\n" + "b" * 500 + "\n" + "c" * 500
    lines = [(0, 500), (501, 1001), (1002, 1502)]

    chunks = item_chunks(lines, text)

    assert [text[start:end] for start, end in chunks] == [
        "a" * 500 + "\n" + "b" * 500,
        "c" * 500,
    ]


def test_long_line_is_cut_every_1200() -> None:
    text = "x" * 2500

    assert item_chunks([(0, len(text))], text) == [(0, 1200), (1200, 2400), (2400, 2500)]


def test_whitespace_only_chunk_is_dropped() -> None:
    text = " " * 400

    assert item_chunks([(0, len(text))], text) == []


def test_chunks_are_contiguous_and_cover_the_item_slice() -> None:
    text = "a" * 500 + "\n" + "b" * 500 + "\n" + "c" * 500
    lines = [(0, 500), (501, 1001), (1002, 1502)]

    chunks = item_chunks(lines, text)

    assert chunks == [(0, 1001), (1002, 1502)]
    assert text[chunks[0][1] : chunks[1][0]] == "\n"
    assert (
        "".join(text[start:end] for start, end in chunks)
        == "a" * 500 + "\n" + "b" * 500 + "c" * 500
    )


def test_header_format_and_200_character_limit() -> None:
    header = header_for(
        "Birch",
        "City Council",
        date(2024, 3, 12),
        "minutes",
        "9.B",
        "Adopt RESOLUTION 2024-07",
    )

    assert header == "Birch City Council 2024-03-12 minutes | 9.B Adopt RESOLUTION 2024-07"
    long_header = header_for("Birch", "City Council", date(2024, 3, 12), "minutes", "", "x" * 400)
    assert len(long_header) == 200
