import pytest

from minutes.extract.verify import quoted_amounts, quoted_counts, verify, ws


def vote(quote: str, ayes: int, noes: int = 0) -> dict[str, object]:
    return {"quote": quote, "ayes": ayes, "noes": noes, "abstain": 0, "absent": 0, "members": []}


def test_quote_must_be_substring_after_whitespace_collapse() -> None:
    text = "The motion  passed today."
    assert verify("motion", {"quote": "motion\npassed", "mover": None, "seconder": None}, text).ok
    result = verify("motion", {"quote": "motion succeeded", "mover": None, "seconder": None}, text)
    assert result.error == "quote_not_found"
    assert result.quote_start is result.quote_end is None


def test_vote_counts_from_seattle_form() -> None:
    quote = "In Favor: 4 - Abbott, Chen, Diaz, Evans\nOpposed: 1 - Flores"
    assert verify("vote", vote(quote, 4, 1), quote).ok
    assert verify("vote", vote(quote, 5, 1), quote).error == "vote_count_mismatch"


def test_vote_counts_from_name_lists_and_none() -> None:
    quote = "Ayes: Gray, Hale, Ivers, Jones\nNoes: King\nAbsent: None\nAbstain: None"
    assert quoted_counts(quote) == {"ayes": 4, "noes": 1, "abstain": 0, "absent": 0}
    assert verify("vote", vote(quote, 4, 1), quote).ok
    assert verify("vote", vote(quote, 3, 1), quote).error == "vote_count_mismatch"


def test_vote_counts_from_table_lines() -> None:
    quote = "\n".join(f"{name} | Aye" for name in ("Gray", "Hale", "Ivers", "Jones", "King"))
    assert verify("vote", vote(quote, 5), quote).ok
    assert verify("vote", vote(quote, 4), quote).error == "vote_count_mismatch"


def test_vote_members_must_match_counts() -> None:
    quote = "Ayes: Gray, Hale, Ivers, Jones"
    data = vote(quote, 4) | {
        "members": [{"name": name, "value": "aye"} for name in ("Gray", "Hale", "Ivers")]
        + [{"name": "Jones", "value": "no"}]
    }
    assert verify("vote", data, quote).error == "vote_members_mismatch"
    data["ayes"] = 3
    assert verify("vote", data, quote).error == "vote_count_mismatch"


def test_missing_label_means_zero() -> None:
    quote = "Ayes: Gray"
    assert verify("vote", vote(quote, 1, 1), quote).error == "vote_count_mismatch"


def test_amount_must_appear_in_quote() -> None:
    quote = "The contract costs $184,500, with a budget of $1.2 million."
    assert quoted_amounts(quote) == [184500.0, 1200000.0]
    for amount in (184500.0, 1200000.0, 184500.004):
        assert verify("amount", {"quote": quote, "amount_usd": amount}, quote).ok
    assert (
        verify("amount", {"quote": quote, "amount_usd": 184000.0}, quote).error
        == "amount_not_in_quote"
    )
    assert quoted_amounts("$2 billion and $3 thousand") == [2e9, 3000.0]


def test_amount_suffixes_match_values() -> None:
    for quote, amount in (
        ("The $14k increase is within the police departments budget.", 14000),
        ("The project costs $3M.", 3_000_000),
        ("The project costs $1B.", 1_000_000_000),
        ("The project costs $2bn.", 2_000_000_000),
        ("The project costs $1.5 million.", 1_500_000),
    ):
        assert verify("amount", {"quote": quote, "amount_usd": amount}, quote).ok


def test_amount_suffix_requires_word_boundary() -> None:
    quote = "The $5 milestones are listed in the budget."
    assert quoted_amounts(quote) == [5.0]
    assert verify("amount", {"quote": quote, "amount_usd": 5}, quote).ok
    assert (
        verify(
            "amount", {"quote": "The $14k increase.", "amount_usd": 14}, "The $14k increase."
        ).error
        == "amount_not_in_quote"
    )


def test_ordinance_identifier_must_appear_in_quote() -> None:
    quote = "Introduce ORDINANCE 1042 Amending Title 15."
    assert verify("ordinance", {"quote": quote, "identifier": "ordinance\n1042"}, quote).ok
    assert (
        verify("ordinance", {"quote": quote, "identifier": "1043"}, quote).error
        == "identifier_not_in_quote"
    )


def test_motion_and_statement_names_must_appear_in_quote() -> None:
    quote = "Councilmember Gray motioned to approve, seconded by Hale."
    data = {"quote": quote, "mover": "Councilmember Gray", "seconder": "Hale"}
    assert verify("motion", data, quote).ok
    assert verify("motion", data | {"mover": "Smith"}, quote).error == "name_not_in_quote"
    assert verify("motion", data | {"seconder": "Smith"}, quote).error == "name_not_in_quote"
    assert verify("statement", {"quote": quote, "speaker": "GRAY"}, quote).ok
    assert (
        verify("statement", {"quote": quote, "speaker": "Smith"}, quote).error
        == "name_not_in_quote"
    )


def test_quote_offsets_point_at_original_text() -> None:
    text = "  Preface\tCouncilmember  Gray\n motioned to approve.  Afterword  "
    quote = "Gray motioned\nto approve."
    result = verify("motion", {"quote": quote, "mover": "Gray", "seconder": None}, text)
    assert result.ok
    assert result.quote_start == text.index("Gray")
    assert result.quote_end == text.index("  Afterword")
    assert ws(text[result.quote_start : result.quote_end]) == ws(quote)


def test_unknown_kind_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unknown fact kind"):
        verify("unknown", {}, "Some source text")


def test_vote_counts_add_labels_and_table_rows() -> None:
    quote = "Ayes: 2 \u2013 Gray, Hale\nNoes: None\nIvers | yes\nJones | Nay\nKing | Excused"
    assert quoted_counts(quote) == {"ayes": 3, "noes": 1, "abstain": 0, "absent": 1}


def test_vote_counts_from_case_insensitive_labels() -> None:
    quote = "In favor: 1 - Moore Opposed: 5 - Hollingsworth, Kettle, Nelson, Strauss, Woo"
    assert verify("vote", vote(quote, 1, 5), quote).ok


def test_vote_tally_overrides_roll_call_members() -> None:
    quote = (
        "Approved by the following roll call vote: E.Johnson \u2013 Aye, J.Johnson \u2013 Aye, "
        "Herzog \u2013 Aye, Jackson - Aye, "
        "Hutchinson \u2013 Aye, Manning \u2013 Aye and Gilbert \u2013 Aye (7-0-0-0)"
    )
    members = [
        {"name": name, "value": "aye"}
        for name in (
            "E.Johnson",
            "J.Johnson",
            "Herzog",
            "Jackson",
            "Hutchinson",
            "Manning",
            "Gilbert",
        )
    ]
    assert quoted_counts(quote) == {"ayes": 7, "noes": 0, "abstain": 0, "absent": 0}
    assert verify("vote", vote(quote, 7) | {"members": members}, quote).ok


def test_vote_counts_from_dash_roll_call() -> None:
    quote = (
        "By the following Vote (Approved): Joiner- Aye, Andreatta -Aye, Brown \u2013 Aye, "
        "Lauritsen \u2013 Aye and Mayor Karleskint- Aye."
    )
    assert quoted_counts(quote) == {"ayes": 5, "noes": 0, "abstain": 0, "absent": 0}
    assert verify("vote", vote(quote, 5), quote).ok


def test_vote_tally_allows_partial_abstainer_members() -> None:
    quote = (
        "Commissioners, Herzog, J.Johnson and Manning abstained. Unanimously Approved "
        "with no opposition (4-0-3-0)."
    )
    members = [{"name": name, "value": "abstain"} for name in ("Herzog", "J.Johnson", "Manning")]
    assert verify("vote", vote(quote, 4) | {"abstain": 3, "members": members}, quote).ok
    assert (
        verify("vote", vote(quote, 5) | {"abstain": 3, "members": members}, quote).error
        == "vote_count_mismatch"
    )


def test_vote_members_reject_counts_above_or_different_from_full_list() -> None:
    quote = "Approved by roll call: Gray - Aye (7-0-0-0)"
    eight_ayes = [{"name": str(index), "value": "aye"} for index in range(8)]
    assert verify("vote", vote(quote, 7) | {"members": eight_ayes}, quote).error == (
        "vote_members_mismatch"
    )

    seven_members = [{"name": str(index), "value": "aye"} for index in range(6)] + [
        {"name": "6", "value": "no"}
    ]
    assert verify("vote", vote(quote, 7) | {"members": seven_members}, quote).error == (
        "vote_members_mismatch"
    )


def test_hyphenated_words_do_not_count_as_dash_votes() -> None:
    quote = "e-Notice was sent to Smith-Noble. Ayes: Gray, Hale"
    assert quoted_counts(quote) == {"ayes": 2, "noes": 0, "abstain": 0, "absent": 0}
    assert verify("vote", vote(quote, 2), quote).ok
