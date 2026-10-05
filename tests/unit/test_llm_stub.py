from typing import Any

import pytest

from minutes.errors import BudgetExceeded, ConfigError
from minutes.llm import Budget
from minutes.llm.prompts import ItemText, SourceSpan, extract_user, synth_user
from minutes.llm.stub import StubClient

BIRCH_9B = """9.B Adopt RESOLUTION 2024-07 Awarding a Construction Contract to
Granite Works Inc. for the Oak Street Sidewalk Repair Project in the
Amount of $184,500.
Councilmember Gray motioned to approve, motion was seconded by
Councilmember Hale. Passed by the following roll call vote:
Ayes: Gray, Hale, Ivers, Jones
Noes: King
Absent: None
Abstain: None"""


def source(number: int, text: str) -> SourceSpan:
    return SourceSpan(number, "Birch", "City Council", "2024-03-12", "minutes", "p. 2", text)


def synth(sources: list[SourceSpan]) -> dict[str, Any]:
    user = synth_user("How did the council vote?", sources)
    result = StubClient().chat(purpose="synth", messages=[{"role": "user", "content": user}])
    assert result.parsed is not None
    return result.parsed


def extract(text: str) -> list[dict[str, Any]]:
    item = ItemText(1, 7, "9.B", "Adopt RESOLUTION 2024-07", text, 1)
    user = extract_user("Birch", "City Council", "2024-03-12", "minutes", [item])
    result = StubClient().chat(purpose="extract", messages=[{"role": "user", "content": user}])
    assert result.parsed is not None
    records: list[dict[str, Any]] = result.parsed["records"]  # type: ignore[assignment]
    return records


def test_synth_reports_vote_counts_from_first_vote_source() -> None:
    sources = [
        source(1, "9. CONSENT\n9.A Approval of Meeting Minutes of February 27, 2024."),
        source(
            2, "Passed by the following roll call vote:\nAyes: Gray, Hale, Ivers, Jones\nNoes: King"
        ),
    ]
    assert synth(sources)["sentences"] == [
        {"text": "The roll call vote was 4 ayes and 1 noes.", "citations": [2]}
    ]


def test_synth_without_vote_quotes_first_source() -> None:
    text = (
        "10.A Adopt a Resolution Approving the Tentative Subdivision Map for the\n"
        "Willow Creek Project. The Planning Commission reviewed the map on February 20 and "
        "recommended approval with the conditions listed in the staff report."
    )
    sentence = synth([source(1, text)])["sentences"][0]
    flat = text.replace("\n", " ")
    assert sentence["citations"] == [1]
    assert sentence["text"] == flat[:160].rsplit(" ", 1)[0] + "."
    assert len(sentence["text"]) <= 161


def test_extract_vote_from_label_lines() -> None:
    text = (
        "1. CB 3001 AN ORDINANCE relating to sidewalk repair.\n"
        "The Council Bill (CB) was passed by the following vote:\n"
        "In Favor: 4 - Abbott, Chen, Diaz, Evans\n"
        "Opposed: 1 - Flores"
    )
    votes = [r for r in extract(text) if r["kind"] == "vote"]
    assert len(votes) == 1
    vote = votes[0]["vote"]
    assert (vote["ayes"], vote["noes"]) == (4, 1)
    assert len(vote["members"]) == 5
    assert vote["outcome"] == "passed"
    assert vote["subject_identifier"] == "CB 3001"
    assert votes[0]["quote"] == "In Favor: 4 - Abbott, Chen, Diaz, Evans\nOpposed: 1 - Flores"


def test_extract_vote_from_table_lines() -> None:
    rows = "\n".join(f"{name} | Aye" for name in ("Gray", "Hale", "Ivers", "Jones", "King"))
    votes = [r for r in extract("10.A Adopt RESOLUTION 2024-08.\nMember | Vote\n" + rows)]
    vote = next(r["vote"] for r in votes if r["kind"] == "vote")
    assert (vote["ayes"], vote["noes"]) == (5, 0)
    assert vote["members"][0] == {"name": "Gray", "value": "aye"}


def test_extract_amount_and_motion() -> None:
    records = extract(BIRCH_9B)
    amounts = [r for r in records if r["kind"] == "amount"]
    motions = [r for r in records if r["kind"] == "motion"]
    assert [r["amount"]["amount_usd"] for r in amounts] == [184500.0]
    assert amounts[0]["quote"] == "Amount of $184,500."
    assert len(motions) == 1
    assert (motions[0]["motion"]["mover"], motions[0]["motion"]["seconder"]) == ("Gray", "Hale")
    assert motions[0]["motion"]["outcome"] == "passed"


def test_extract_ordinance_once_per_identifier() -> None:
    ordinances = [r["ordinance"] for r in extract(BIRCH_9B) if r["kind"] == "ordinance"]
    assert ordinances == [
        {
            "identifier": "RESOLUTION 2024-07",
            "legislation_type": "resolution",
            "title": "Adopt RESOLUTION 2024-07",
            "status": "unknown",
        }
    ]


def test_unknown_purpose_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="stub has no rule for purpose judge"):
        StubClient().chat(purpose="judge", messages=[{"role": "user", "content": ""}])


def test_budget_refuses_a_call_that_would_pass_the_limit() -> None:
    budget = Budget(0.05)
    budget.check(0.03)
    budget.add(0.03)
    with pytest.raises(BudgetExceeded):
        budget.check(0.03)
