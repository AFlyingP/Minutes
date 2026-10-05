import pytest
from pydantic import ValidationError

from minutes.labels.schema import (
    AgendaCount,
    AgentTask,
    Label,
    LabelIn,
    QuestionAnswerable,
    QuestionUnanswerable,
    parse_payload,
)

PASSAGE: dict[str, object] = {
    "document_id": "birch-minutes-201",
    "unit_index": 2,
    "start": 0,
    "end": 12,
    "text": "Page 2 of 3\n",
}
QUESTION: dict[str, object] = {
    "question": "How did the council vote?",
    "answerable": True,
    "question_type": "vote",
    "answer": "4",
    "answer_type": "number",
    "passages": [PASSAGE],
}
ABSENT: dict[str, object] = {
    "question": "Was the quarry approved?",
    "answerable": False,
    "reason": "entity_not_in_corpus",
    "absence_searches": [{"query": query, "hits": 0} for query in ("quarry", "mine", "stone")],
}


def test_answerable_question_requires_1_to_3_passages() -> None:
    for count in (1, 2, 3):
        assert isinstance(
            parse_payload("question", QUESTION | {"passages": [PASSAGE] * count}),
            QuestionAnswerable,
        )
    for count in (0, 4):
        with pytest.raises(ValidationError):
            parse_payload("question", QUESTION | {"passages": [PASSAGE] * count})


def test_unanswerable_question_requires_reason_from_closed_list() -> None:
    for reason in (
        "entity_not_in_corpus",
        "date_outside_corpus",
        "fact_not_recorded",
        "false_premise",
    ):
        assert isinstance(
            parse_payload("question", ABSENT | {"reason": reason}), QuestionUnanswerable
        )
    with pytest.raises(ValidationError):
        parse_payload("question", ABSENT | {"reason": "unknown"})
    with pytest.raises(ValidationError):
        parse_payload("question", {key: value for key, value in ABSENT.items() if key != "reason"})
    with pytest.raises(ValidationError):
        parse_payload("question", ABSENT | {"absence_searches": []})


def test_extra_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        parse_payload("question", QUESTION | {"extra": True})
    with pytest.raises(ValidationError):
        parse_payload("question", QUESTION | {"passages": [PASSAGE | {"extra": True}]})
    with pytest.raises(ValidationError):
        LabelIn.model_validate(
            {
                "type": "question",
                "city": "birch",
                "author": "human",
                "payload": QUESTION,
                "id": "q-birch-001",
            }
        )
    with pytest.raises(ValidationError):
        parse_payload(
            "extraction",
            {
                "kind": "amount",
                "passage": PASSAGE,
                "record": {"amount_usd": 10, "purpose": "repair", "payee": None, "extra": True},
            },
        )


def test_agenda_count_payload_shape() -> None:
    payload: dict[str, object] = {
        "meeting_id": "birch-201",
        "count": 1,
        "items": [{"identifier": "9.B", "title": "Sidewalk repair", "start_page": 2}],
    }
    assert isinstance(parse_payload("agenda_count", payload), AgendaCount)
    for changes in (
        {"count": -1},
        {"items": [{"identifier": "9.B", "title": "x" * 121, "start_page": 2}]},
        {"items": [{"identifier": "9.B", "title": "repair", "start_page": 0}]},
    ):
        with pytest.raises(ValidationError):
            parse_payload("agenda_count", payload | changes)


def test_agent_task_requires_two_documents() -> None:
    payload: dict[str, object] = {
        "question": "Which plans were approved?",
        "answer": "Both",
        "answer_type": "free_text",
        "document_ids": ["birch-agenda-201", "birch-minutes-201"],
    }
    assert isinstance(parse_payload("agent_task", payload), AgentTask)
    with pytest.raises(ValidationError):
        parse_payload("agent_task", payload | {"document_ids": ["birch-agenda-201"]})


@pytest.mark.parametrize(
    "kind,record",
    [
        ("motion", {"text": "approve", "mover": None, "seconder": None, "outcome": "passed"}),
        (
            "vote",
            {
                "subject_identifier": None,
                "ayes": 4,
                "noes": 1,
                "abstain": 0,
                "absent": 0,
                "members": [{"name": "Gray", "value": "aye"}],
                "outcome": "passed",
            },
        ),
        (
            "ordinance",
            {
                "identifier": "1042",
                "legislation_type": "ordinance",
                "title": "Building code",
                "status": "introduced",
            },
        ),
        ("amount", {"amount_usd": 184500, "purpose": "Sidewalk repair", "payee": None}),
        ("statement", {"speaker": "Gray", "role": None, "summary": "Approved the contract"}),
    ],
)
def test_extraction_record_matches_kind(kind: str, record: dict[str, object]) -> None:
    payload: dict[str, object] = {"kind": kind, "passage": PASSAGE, "record": record}
    assert parse_payload("extraction", payload)
    with pytest.raises(ValidationError):
        parse_payload("extraction", payload | {"record": {}})
    with pytest.raises(ValidationError):
        parse_payload("extraction", payload | {"record": record | {"extra": True}})


def test_label_envelope_bounds_and_unknown_type() -> None:
    for changes in ({"note": "x" * 1001}, {"author": "unknown"}, {"type": "unknown"}):
        with pytest.raises(ValidationError):
            LabelIn.model_validate(
                {"type": "question", "city": "birch", "author": "human", "payload": QUESTION}
                | changes
            )
    with pytest.raises(ValueError, match="unknown label type"):
        parse_payload("unknown", {})


def test_record_error_path_starts_with_record() -> None:
    with pytest.raises(ValidationError) as error:
        parse_payload("extraction", {"kind": "vote", "passage": PASSAGE, "record": {"ayes": -1}})
    locations = [detail["loc"] for detail in error.value.errors()]
    assert ("record", "ayes") in locations
    assert ("record", "noes") in locations
    assert all(location[0] == "record" for location in locations)


def test_label_fields_are_in_envelope_order() -> None:
    assert list(Label.model_fields) == [
        "id",
        "type",
        "city",
        "author",
        "created_utc",
        "human_reviewed",
        "note",
        "payload",
    ]
