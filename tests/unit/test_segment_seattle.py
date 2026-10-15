from minutes.ingest.segment import ItemSpan, segment_document


def parse(text: str) -> list[ItemSpan]:
    return segment_document("legistar", [(1, text)])


def test_headings_become_section_items() -> None:
    items = parse("A. CALL TO ORDER\nB. ROLL CALL")

    assert [(item.kind, item.identifier, item.title) for item in items] == [
        ("section", "A", "CALL TO ORDER"),
        ("section", "B", "ROLL CALL"),
    ]


def test_anchor_with_title_starts_business_item() -> None:
    items = parse("2. CB 120746 AN ORDINANCE appropriating money")
    business = [item for item in items if item.kind == "business"]

    assert len(business) == 1
    assert business[0].identifier == "CB 120746"
    assert business[0].title.startswith("AN ORDINANCE appropriating money")


def test_bare_anchor_takes_preceding_title_block() -> None:
    items = parse("A. ITEMS\nAN ORDINANCE relating to X;\nauthorizing Y.\n1. CB 120725")
    business = [item for item in items if item.kind == "business"]

    assert len(business) == 1
    assert business[0].start_offset == len("A. ITEMS\n")
    assert business[0].title == "AN ORDINANCE relating to X; authorizing Y."


def test_title_stops_at_next_bare_anchor_block() -> None:
    items = parse(
        "1. CB 120725 AN ORDINANCE for the first item\n"
        "A RESOLUTION for the second item\n"
        "2. Res 32128"
    )
    business = [item for item in items if item.kind == "business"]

    assert [item.title for item in business] == [
        "AN ORDINANCE for the first item",
        "A RESOLUTION for the second item",
    ]


def test_bare_anchor_without_title_start_uses_anchor_line_as_title() -> None:
    items = parse("A. ITEMS\n1. CB 120725")
    business = [item for item in items if item.kind == "business"]

    assert business[0].title == "1. CB 120725"


def test_identifier_in_running_text_starts_nothing() -> None:
    items = parse("A. OTHER BUSINESS\nTitle mentions Clerk File 314512.\nSee CB 120725 mid-line")

    assert not [item for item in items if item.kind == "business"]


def test_repeated_identifier_starts_nothing() -> None:
    items = parse("A. ITEMS\n1. CB 120725 A RESOLUTION\n2. CB 120725 duplicate")

    assert [item.identifier for item in items if item.kind == "business"] == ["CB 120725"]


def test_item_spans_units() -> None:
    items = segment_document(
        "legistar",
        [
            (1, "A. ITEMS"),
            (2, "1. CB 120725 AN ORDINANCE\ncontinuing"),
            (3, "Page header\n2. Res 32128 A RESOLUTION"),
        ],
    )
    first = next(item for item in items if item.identifier == "CB 120725")

    assert first.start_unit == 2
    assert first.end_unit == 3
    assert first.end_offset == len("Page header\n")


def test_worked_example_s1_counts_five() -> None:
    items = parse(
        "E. INTRODUCTION AND REFERRAL CALENDAR:\nIRC 426\n"
        "G. CONSENT CALENDAR\n1. Min 462\n2. CB 120746 AN ORDINANCE\n"
        "H. ITEMS\n1. CB 120725 A RESOLUTION\n"
        "J. RESOLUTIONS\n2. Res 32128 A RESOLUTION re Clerk File 314512"
    )

    assert [item.identifier for item in items if item.kind == "business"] == [
        "IRC 426",
        "Min 462",
        "CB 120746",
        "CB 120725",
        "Res 32128",
    ]


def test_document_without_anchors_is_one_preamble_item() -> None:
    items = parse("Opening text\nNo agenda identifiers")

    assert [(item.kind, item.identifier, item.title) for item in items] == [
        ("section", "PREAMBLE", "Preamble")
    ]


def test_title_cut_at_committee_recommendation_and_300_characters() -> None:
    with_stop = parse(
        "1. CB 120725 AN ORDINANCE changing the code\n"
        "The Committee recommends passage.\nAdditional text"
    )
    long_title = parse("1. CB 120726 " + "A RESOLUTION " + "x" * 350)

    assert next(item for item in with_stop if item.kind == "business").title == (
        "AN ORDINANCE changing the code"
    )
    assert len(next(item for item in long_title if item.kind == "business").title) == 300
