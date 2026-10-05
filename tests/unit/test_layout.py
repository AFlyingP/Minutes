from minutes.ingest.layout import Table, Word, build_text, normalize_text


def test_single_column_lines_in_top_order() -> None:
    words = [
        Word("three", 90, 140, 120, 150),
        Word("two", 90, 120, 110, 130),
        Word("line", 50, 100, 70, 110),
        Word("line", 50, 140, 70, 150),
        Word("one", 90, 100, 110, 110),
        Word("line", 50, 120, 70, 130),
    ]
    text, _ = build_text(words, [], 600, 800)
    assert text == "line one\nline two\nline three"


def test_words_within_3pt_join_one_line() -> None:
    words = [
        Word("third", 50, 104.0, 80, 114),
        Word("second", 90, 102.5, 120, 112.5),
        Word("first", 50, 100.0, 80, 110),
    ]
    text, _ = build_text(words, [], 600, 800)
    assert text == "first second\nthird"


def test_two_column_reading_order() -> None:
    left = [Word(f"left{i}", 100, 100 + 10 * i, 350, 108 + 10 * i) for i in range(30)]
    right = [Word(f"right{i}", 650, 105 + 10 * i, 900, 113 + 10 * i) for i in range(30)]
    text, _ = build_text(list(reversed(right + left)), [], 1000, 1000)
    assert text.splitlines() == [word.text for word in left + right]


def test_full_width_line_splits_bands() -> None:
    above = [Word("above-left", 100, 50, 350, 60), Word("above-right", 650, 50, 900, 60)]
    heading = Word("heading", 100, 70, 900, 78)
    left = [Word(f"left{i}", 100, 100 + 10 * i, 350, 108 + 10 * i) for i in range(30)]
    right = [Word(f"right{i}", 650, 100 + 10 * i, 900, 108 + 10 * i) for i in range(30)]
    text, _ = build_text([*right, heading, *above, *left], [], 1000, 1000)
    assert text.splitlines() == [word.text for word in [*above, heading, *left, *right]]

    table = Table(
        (100, 450, 900, 490), [[(100, y, 500, y + 20), (500, y, 900, y + 20)] for y in (450, 470)]
    )
    table_words = [Word("Member", 110, 452, 160, 462), Word("Vote", 650, 452, 700, 462)]
    below = [Word("below-left", 100, 500, 350, 510), Word("below-right", 650, 500, 900, 510)]
    text, _ = build_text(left + right + table_words + below, [table], 1000, 1000)
    assert text.splitlines() == [word.text for word in left + right] + [
        "Member | Vote",
        " | ",
        "below-left",
        "below-right",
    ]


def test_no_gutter_when_side_has_fewer_than_25_words() -> None:
    for left_count, right_count in ((24, 30), (30, 24)):
        left = [Word(f"left{i}", 100, 100 + 10 * i, 350, 108 + 10 * i) for i in range(left_count)]
        right = [
            Word(f"right{i}", 650, 105 + 10 * i, 900, 113 + 10 * i) for i in range(right_count)
        ]
        words = left + right
        text, _ = build_text(words, [], 1000, 1000)
        assert text.splitlines() == [word.text for word in sorted(words, key=lambda word: word.top)]


def test_gutter_must_lie_in_middle_40_percent() -> None:
    for left_edge, right_edge in ((150, 200), (250, 450), (550, 750), (800, 850)):
        left = [Word(f"left{i}", 50, 100 + 10 * i, left_edge, 108 + 10 * i) for i in range(30)]
        right = [Word(f"right{i}", right_edge, 105 + 10 * i, 950, 113 + 10 * i) for i in range(30)]
        words = left + right
        text, _ = build_text(words, [], 1000, 1000)
        assert text.splitlines() == [word.text for word in sorted(words, key=lambda word: word.top)]


def test_table_rows_join_cells_with_pipe() -> None:
    rows: list[list[tuple[float, float, float, float]]] = [
        [(50, y, 150, y + 20), (150, y, 250, y + 20)] for y in (100, 120, 140)
    ]
    table = Table((50, 100, 250, 160), rows)
    words = [
        Word(text, x, y, x + 30, y + 8)
        for y, row in zip(
            (105, 125, 145), (("Member", "Vote"), ("Gray", "Aye"), ("Hale", "Aye")), strict=True
        )
        for x, text in zip((55, 155), row, strict=True)
    ]
    text, boxes = build_text(list(reversed(words)), [table], 600, 800)
    assert text == "Member | Vote\nGray | Aye\nHale | Aye"
    assert len(boxes) == len(words)
    empty_text, _ = build_text(words[:-1], [table], 600, 800)
    assert empty_text == "Member | Vote\nGray | Aye\nHale | "
    for ignored in (Table(table.bbox, rows[:1]), Table(table.bbox, [[row[0]] for row in rows])):
        text, boxes = build_text(words, [ignored], 600, 800)
        assert text == "Member Vote\nGray Aye\nHale Aye"
        assert len(boxes) == len(words)

    unassigned = [
        Word("unassigned", 270, 125, 290, 133),
        Word("right-edge", 240, 125, 260, 133),
        Word("bottom-edge", 55, 155, 85, 165),
    ]
    text, boxes = build_text(words + unassigned, [Table((50, 100, 300, 170), rows)], 600, 800)
    assert len(boxes) == len(words) + len(unassigned)
    boxed_words = [text[box[0] : box[1]] for box in boxes]
    for word in unassigned:
        assert word.text in text.split()
        assert boxed_words.count(word.text) == 1


def test_boxes_cover_each_word_and_match_text() -> None:
    words = [
        Word("outside", 40.123, 50.321, 85.432, 60.654),
        Word("caf\u0065\u0301", 60.123, 105.321, 90.432, 115.654),
        Word("\u2018yes\u2019", 165.123, 125.321, 200.432, 135.654),
        Word("next\u00a0word", 60.123, 125.321, 130.432, 135.654),
        Word("\u201cVote\u201d", 165.123, 105.321, 200.432, 115.654),
    ]
    table = Table(
        (50, 100, 250, 140), [[(50, y, 150, y + 20), (150, y, 250, y + 20)] for y in (100, 120)]
    )
    text, boxes = build_text(words, [table], 612, 792)
    assert text == "outside\ncaf\u00e9 | \"Vote\"\nnext word | 'yes'"
    assert len(boxes) == len(words)
    expected = {normalize_text(word.text): word for word in words}
    covered = set()
    for start, end, x0, y0, x1, y1 in boxes:
        word = expected[text[start:end]]
        covered.add(word)
        assert (x0, y0, x1, y1) == (
            round(word.x0 / 612, 4),
            round(word.top / 792, 4),
            round(word.x1 / 612, 4),
            round(word.bottom / 792, 4),
        )
        assert all(
            0 <= coordinate <= 1 and coordinate == round(coordinate, 4)
            for coordinate in (x0, y0, x1, y1)
        )
    assert covered == set(words)
    assert build_text([], [], 612, 792) == ("", [])


def test_normalize_text_replaces_nbsp_and_curly_quotes() -> None:
    text = "\u201cCafe\u0301\u201d\u00a0\u2018can\u2019t\u2019\t  stop \t\n  next\tline\t "
    assert normalize_text(text) == "\"Caf\u00e9\" 'can't' stop\n next line"
