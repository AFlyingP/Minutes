from minutes.ingest.segment import ItemRow, ItemSpan, link_items, segment_document


def parse(text: str) -> list[ItemSpan]:
    return segment_document("granicus", [(1, text)])


def test_top_level_headings_are_sections() -> None:
    items = parse("1. CALL TO ORDER\n2. ROLL CALL\n12. ADJOURNMENT")

    assert [(item.kind, item.identifier) for item in items] == [
        ("section", "1."),
        ("section", "2."),
        ("section", "12."),
    ]


def test_sub_items_are_business() -> None:
    items = parse("13. CONSENT\n13.A Adopt a Resolution")
    business = [item for item in items if item.kind == "business"]

    assert [(item.identifier, item.title) for item in business] == [("13.A", "Adopt a Resolution")]


def test_sub_item_with_trailing_dot() -> None:
    items = parse(
        "7. CONSENT\n"
        "7.A. Review and approve June 21, 2023 meeting minutes.\n"
        "7.A. Draft Meeting Minutes 06.21.2023.pdf"
    )
    business = [item for item in items if item.kind == "business"]

    assert [(item.identifier, item.title) for item in business] == [
        ("7.A", "Review and approve June 21, 2023 meeting minutes.")
    ]


def test_bare_sub_item_takes_title_from_next_lines() -> None:
    items = parse(
        "15. GENERAL BUSINESS\n"
        "15.A Adopt a Resolution\n"
        "15.B\n"
        "Approve the Fireworks Show Agreement.\n"
        "15.B.pdf\n"
        "16. CITY MANAGER REPORTS"
    )
    business = [item for item in items if item.kind == "business"]

    assert [item.identifier for item in business] == ["15.A", "15.B"]
    assert next(item for item in business if item.identifier == "15.B").title.startswith(
        "Approve the Fireworks Show Agreement."
    )


def test_two_letter_sub_item() -> None:
    items = parse("9. CONSENT\n9.AA Approve the amended plan")

    assert [item.identifier for item in items if item.kind == "business"] == ["9.AA"]


def test_repeated_sub_item_identifier_starts_nothing() -> None:
    items = parse("9. CONSENT\n9.A Approve the plan\n9.A Approve it again")

    assert [item.identifier for item in items if item.kind == "business"] == ["9.A"]


def test_title_cut_at_motion_line() -> None:
    items = parse("9.B Approve the project\nCouncilmember Gray MOTIONED to approve the project")

    assert next(item for item in items if item.kind == "business").title == "Approve the project"


def test_worked_example_l2_counts_sixteen() -> None:
    lines = [f"{number}. HEADING {number}" for number in range(1, 16)]
    lines.extend(("1.A Closed session", "1.B Closed session", "10.A Presentation"))
    lines.extend(f"13.{letter} Consent item {letter}" for letter in "ABCDEFGHIJ")
    lines.extend(("14.A Public hearing", "15.A General business", "15.B General business"))

    items = parse("\n".join(lines))

    assert sum(item.kind == "business" for item in items) == 16


def test_lowercase_numbered_sentence_is_not_a_heading() -> None:
    items = parse("2. the applicant shall submit the form")

    assert len(items) == 1
    assert items[0].identifier == "PREAMBLE"


def test_link_by_title_similarity() -> None:
    agenda = [
        ItemRow(
            1,
            "business",
            "13.B",
            "Adopt a Resolution Awarding a Construction Contract for the Brentford Circle Trail "
            "Crossings Project",
            1,
        ),
        ItemRow(2, "business", "13.C", "Approve a park maintenance agreement", 2),
    ]
    minutes = [
        ItemRow(
            3,
            "business",
            "11.B",
            "Adopt RESOLUTION 2025-157 Awarding a Construction Contract for the Brentford Circle "
            "Trail Crossings Project",
            1,
        ),
        ItemRow(4, "business", "11.C", "Authorize a software procurement", 2),
    ]

    assert link_items(agenda, minutes, "granicus") == [(3, 1)]


def test_link_ties_go_to_lowest_agenda_ordinal() -> None:
    agenda = [
        ItemRow(2, "business", "13.B", "Approve the same city construction project", 2),
        ItemRow(1, "business", "13.A", "Approve the same city construction project", 1),
    ]
    minutes = [ItemRow(3, "business", "11.A", "Approve the same city construction project", 1)]

    assert link_items(agenda, minutes, "granicus") == [(3, 1)]


def test_each_agenda_item_linked_at_most_once() -> None:
    agenda = [ItemRow(1, "business", "13.A", "Approve a major construction project", 1)]
    minutes = [
        ItemRow(2, "business", "11.A", "Approve a major construction project", 1),
        ItemRow(3, "business", "11.B", "Approve a major construction project", 2),
    ]

    assert link_items(agenda, minutes, "granicus") == [(2, 1)]
