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
