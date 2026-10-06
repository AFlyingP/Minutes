import random
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import psycopg
import pytest
from psycopg.rows import TupleRow

from minutes.config import load_cities
from minutes.errors import (
    CorpusError,
    LabelsFrozenError,
    LabelValidationError,
    NotFoundError,
    ValidationError,
)
from minutes.labels import store
from minutes.labels.schema import LabelIn, Passage

pytestmark = pytest.mark.integration
NOW = datetime(2024, 5, 1, 12, 30, 45, tzinfo=UTC)


@pytest.fixture(autouse=True)
def source_corpus(fixture_corpus: str, monkeypatch: pytest.MonkeyPatch) -> Callable[[Path], bool]:
    is_frozen = store.is_frozen
    monkeypatch.setattr(store, "is_frozen", lambda *_: False)
    return is_frozen


def passage(
    conn: psycopg.Connection[TupleRow],
    document: str = "birch-minutes-201",
    unit: int = 2,
    text: str | None = None,
) -> dict[str, object]:
    row = conn.execute(
        "SELECT text FROM units WHERE document_id = %s AND unit_index = %s", (document, unit)
    ).fetchone()
    assert row is not None
    unit_text: str = row[0]
    selected = text if text is not None else unit_text[:50]
    start = unit_text.index(selected)
    return {
        "document_id": document,
        "unit_index": unit,
        "start": start,
        "end": start + len(selected),
        "text": selected,
    }


def question(passages: list[dict[str, object]], **changes: object) -> LabelIn:
    payload: dict[str, object] = {
        "question": "How many supported the proposal?",
        "answerable": True,
        "question_type": "vote",
        "answer": "4",
        "answer_type": "number",
        "passages": passages,
    }
    return LabelIn(type="question", city="birch", author="human", payload=payload | changes)


def messages(conn: psycopg.Connection[TupleRow], label: LabelIn) -> list[str]:
    with pytest.raises(LabelValidationError) as error:
        store.create(conn, label, now=lambda: NOW)
    result: list[str] = error.value.args[0]
    return result


def test_create_assigns_sequential_id_and_utc(conn: psycopg.Connection[TupleRow]) -> None:
    case = question([passage(conn)])
    first = store.create(conn, case, now=lambda: NOW)
    second = store.create(conn, case, now=lambda: NOW)
    assert first.id == "q-birch-001"
    assert second.id == "q-birch-002"
    assert first.created_utc == NOW
    assert not first.human_reviewed
    assert store.list_labels(conn, "question", "birch") == [first, second]
    assert store.list_labels(conn, "extraction", "birch") == []


def test_frozen_labels_reject_create_at_store_level(
    conn: psycopg.Connection[TupleRow],
    tmp_path: Path,
    source_corpus: Callable[[Path], bool],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(store, "is_frozen", source_corpus)
    (tmp_path / "manifest.json").write_text('{"frozen": true}', encoding="utf-8")
    with pytest.raises(LabelsFrozenError, match="labels are frozen"):
        store.create(conn, question([passage(conn)]), labels_dir=tmp_path)
    assert store.list_labels(conn) == []


def test_passage_text_must_match_stored_unit_text(conn: psycopg.Connection[TupleRow]) -> None:
    p = passage(conn)
    p["text"] = str(p["text"])[:-1] + "!"
    assert messages(conn, question([p])) == ["passage 1: text does not match stored unit text"]
    with pytest.raises(LabelValidationError):
        store.validate_passage(conn, Passage.model_validate(p))


@pytest.mark.parametrize(
    "changes", [{"start": -1}, {"end": 10000}, {"start": 50, "end": 50}, {"unit_index": 999}]
)
def test_passage_offsets_out_of_range_rejected(
    conn: psycopg.Connection[TupleRow], changes: dict[str, object]
) -> None:
    assert "passage 1: offsets out of range" in messages(conn, question([passage(conn) | changes]))


def test_passage_cannot_reference_other_city_document(conn: psycopg.Connection[TupleRow]) -> None:
    p = passage(conn, "alder-minutes-101", 1)
    assert messages(conn, question([p])) == ["passage 1: unknown document alder-minutes-101"]
    for changes in ({"document_id": "missing-document"}, {"document_id": "birch-minutes-201"}):
        if changes["document_id"] == "birch-minutes-201":
            conn.execute(
                "UPDATE documents SET status = 'failed' WHERE id = %s", (changes["document_id"],)
            )
        assert any(
            "unknown document" in message
            for message in messages(
                conn, question([passage(conn, "alder-minutes-101", 1) | changes])
            )
        )


@pytest.mark.parametrize("length,valid", [(9, False), (10, True), (2000, True), (2001, False)])
def test_passage_length_bounds(
    conn: psycopg.Connection[TupleRow], length: int, valid: bool
) -> None:
    conn.execute(
        "UPDATE units SET text = %s WHERE document_id = 'birch-minutes-201' AND unit_index = 2",
        ("x" * 2100,),
    )
    p = passage(conn, text="x" * length)
    if valid:
        store.create(conn, question([p]), now=lambda: NOW)
    else:
        assert messages(conn, question([p])) == ["passage 1: length must be 10 to 2000 characters"]


def test_question_copying_six_consecutive_words_rejected(
    conn: psycopg.Connection[TupleRow],
) -> None:
    p = passage(conn, "birch-minutes-201", 1, "for the Oak Street Sidewalk Repair Project in the")
    assert messages(
        conn, question([p], question="Who endorsed FOR the Oak Street Sidewalk Repair?")
    ) == ["question copies more than 5 consecutive words from passage 1"]


def test_five_consecutive_words_allowed(conn: psycopg.Connection[TupleRow]) -> None:
    p = passage(conn, "birch-minutes-201", 1, "for the Oak Street Sidewalk Repair Project in the")
    assert store.create(
        conn,
        question([p], question="Who endorsed the Oak Street Sidewalk Repair?"),
        now=lambda: NOW,
    )


def test_question_copying_second_passage_is_rejected(
    conn: psycopg.Connection[TupleRow],
) -> None:
    first = passage(conn)
    second = passage(
        conn, "birch-minutes-201", 1, "for the Oak Street Sidewalk Repair Project in the"
    )
    assert messages(
        conn, question([first, second], question="Who endorsed FOR the Oak Street Sidewalk Repair?")
    ) == ["question copies more than 5 consecutive words from passage 2"]


def test_multi_passage_requires_two_pages_of_one_meeting(
    conn: psycopg.Connection[TupleRow],
) -> None:
    p = passage(conn)
    error = "multi_passage needs passages on two different pages of one meeting"
    for passages in ([p], [p, p], [p, passage(conn, "birch-minutes-202", 1)]):
        assert error in messages(conn, question(passages, question_type="multi_passage"))
    assert store.create(
        conn,
        question([p, passage(conn, "birch-agenda-201", 2)], question_type="multi_passage"),
        now=lambda: NOW,
    )


def test_answer_must_match_answer_type(conn: psycopg.Connection[TupleRow]) -> None:
    p = passage(conn)
    for answer in ("March 12", "2024-02-30", "2024-3-12"):
        assert messages(conn, question([p], answer_type="date", answer=answer)) == [
            "answer does not match answer_type date"
        ]
    assert store.create(
        conn, question([p], answer_type="date", answer="2024-03-12"), now=lambda: NOW
    )
    for answer in ("four", "1,000", "4\n"):
        assert messages(conn, question([p], answer=answer)) == [
            "answer does not match answer_type number"
        ]
    for answer in ("-4", "4.5"):
        assert store.create(conn, question([p], answer=answer), now=lambda: NOW)
    for answer in ("", "x" * 301):
        assert messages(conn, question([p], answer_type="name", answer=answer)) == [
            "answer does not match answer_type name"
        ]


def unanswerable() -> LabelIn:
    return LabelIn(
        type="question",
        city="birch",
        author="human",
        payload={
            "question": "Was the quarry approved?",
            "answerable": False,
            "reason": "entity_not_in_corpus",
            "absence_searches": [
                {"query": "Granite Works", "hits": 0},
                {"query": "quarry", "hits": 0},
                {"query": "stone mine", "hits": 0},
            ],
        },
    )


def test_absence_search_hits_must_equal_stored_count(conn: psycopg.Connection[TupleRow]) -> None:
    case = unanswerable()
    assert messages(conn, case) == ["absence search 1: hits is 0, stored text has 3"]
    case.payload["absence_searches"] = [
        {"query": "Granite Works", "hits": 3},
        {"query": "quarry", "hits": 0},
        {"query": "stone mine", "hits": 0},
    ]
    assert store.create(conn, case, now=lambda: NOW)


def test_agenda_count_in_city_with_too_few_agendas_is_a_validation_message(
    conn: psycopg.Connection[TupleRow],
) -> None:
    case = LabelIn(
        type="agenda_count",
        city="birch",
        author="human",
        payload={
            "meeting_id": "birch-201",
            "count": 1,
            "items": [{"identifier": "9.B", "title": "Sidewalk repair", "start_page": 2}],
        },
    )
    assert messages(conn, case) == ["agenda_count: city birch needs at least 10 downloaded agendas"]
    assert store.list_labels(conn) == []


def test_agenda_count_must_be_sampled_meeting_and_match_items(
    conn: psycopg.Connection[TupleRow], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        store,
        "sample_meetings",
        lambda _conn, city: (
            [f"{city}-201", f"{city}-202"] if city == "birch" else ["alder-101", "alder-102"]
        ),
    )
    case = LabelIn(
        type="agenda_count",
        city="birch",
        author="human",
        payload={
            "meeting_id": "birch-201",
            "count": 1,
            "items": [{"identifier": "9.B", "title": "Sidewalk repair", "start_page": 2}],
        },
    )
    assert messages(conn, case.model_copy(update={"payload": case.payload | {"count": 2}})) == [
        "agenda_count: count does not match items"
    ]
    for changes, expected in (
        ({"meeting_id": "unknown-meeting"}, "unknown meeting"),
        ({"meeting_id": "alder-101"}, "unknown meeting"),
        (
            {"items": [{"identifier": "9.B", "title": "repair", "start_page": 999}]},
            "start_page out of range",
        ),
        (
            {"count": 2, "items": [{"identifier": "9.B", "title": "repair", "start_page": 2}] * 2},
            "identifiers must be distinct",
        ),
    ):
        assert f"agenda_count: {expected}" in messages(
            conn, case.model_copy(update={"payload": case.payload | changes})
        )
    created = store.create(conn, case, now=lambda: NOW)
    assert store.update(conn, created.id, case).id == created.id
    assert messages(conn, case) == ["agenda_count: meeting already has a label"]
    store.delete(conn, created.id)
    monkeypatch.setattr(store, "sample_meetings", lambda *_: ["birch-202"])
    assert messages(conn, case) == ["agenda_count: meeting is not sampled"]
    conn.execute("UPDATE documents SET status = 'failed' WHERE id = 'birch-agenda-201'")
    assert "agenda_count: meeting has no downloaded agenda" in messages(conn, case)


def test_extraction_vote_record_must_match_passage_counts(
    conn: psycopg.Connection[TupleRow],
) -> None:
    p = passage(conn, text="Ayes: Gray, Hale, Ivers, Jones")
    record: dict[str, object] = {
        "subject_identifier": None,
        "ayes": 5,
        "noes": 0,
        "abstain": 0,
        "absent": 0,
        "members": [],
        "outcome": "passed",
    }
    case = LabelIn(
        type="extraction",
        city="birch",
        author="human",
        payload={"kind": "vote", "passage": p, "record": record},
    )
    assert messages(conn, case) == ["extraction: vote_count_mismatch"]
    case.payload["record"] = record | {"ayes": 4}
    assert store.create(conn, case, now=lambda: NOW)


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({"document_id": "missing-document"}, "unknown document missing-document"),
        ({"document_id": "alder-minutes-101"}, "unknown document alder-minutes-101"),
        ({"unit_index": 999}, "offsets out of range"),
        ({"end": 10000}, "offsets out of range"),
        ({"text": "wrong source text"}, "text does not match stored unit text"),
        ({"end": 9, "text": "Ayes: Gra"}, "length must be 10 to 2000 characters"),
    ],
)
def test_extraction_passage_rules_are_checked(
    conn: psycopg.Connection[TupleRow], changes: dict[str, object], expected: str
) -> None:
    p = passage(conn, text="Ayes: Gray, Hale, Ivers, Jones")
    if changes.get("end") == 9:
        changes = changes | {"end": Passage.model_validate(p).start + 9}
    case = LabelIn(
        type="extraction",
        city="birch",
        author="human",
        payload={
            "kind": "vote",
            "passage": p | changes,
            "record": {
                "subject_identifier": None,
                "ayes": 4,
                "noes": 0,
                "abstain": 0,
                "absent": 0,
                "members": [],
                "outcome": "passed",
            },
        },
    )
    assert f"passage 1: {expected}" in messages(conn, case)
    assert store.list_labels(conn) == []


def test_extraction_passage_must_be_agenda_or_minutes(conn: psycopg.Connection[TupleRow]) -> None:
    p = passage(conn, "birch-transcript-201", 1)
    case = LabelIn(
        type="extraction",
        city="birch",
        author="human",
        payload={
            "kind": "statement",
            "passage": p,
            "record": {"speaker": "SPEAKER 1", "role": None, "summary": "Opening the meeting"},
        },
    )
    assert "extraction: document kind must be agenda or minutes" in messages(conn, case)
    case.payload["passage"] = passage(conn, "birch-agenda-201", 1)
    case.payload["record"] = {"speaker": "BIRCH", "role": None, "summary": "City agenda"}
    assert store.create(conn, case, now=lambda: NOW)


def test_all_messages_are_collected(conn: psycopg.Connection[TupleRow]) -> None:
    p = passage(conn) | {"text": "wrong source text"}
    assert messages(conn, question([p, p])) == [
        "passage 1: text does not match stored unit text",
        "passage 2: text does not match stored unit text",
    ]
    assert store.list_labels(conn) == []


def test_text_search_counts_units_case_insensitively(conn: psycopg.Connection[TupleRow]) -> None:
    result = store.text_search(conn, "birch", "granite works")
    assert result.unit_count == 3
    assert result == store.text_search(conn, "birch", "GRANITE WORKS")
    assert [(document, unit) for document, unit, _ in result.matches] == sorted(
        (document, unit) for document, unit, _ in result.matches
    )
    assert all("granite works" in snippet.lower() for _, _, snippet in result.matches)
    assert store.text_search(conn, "alder", "granite works").unit_count == 0
    with pytest.raises(ValidationError):
        store.text_search(conn, "birch", "x")
    conn.execute("UPDATE documents SET status = 'failed' WHERE id = 'birch-agenda-201'")
    assert store.text_search(conn, "birch", "granite works").unit_count == 2


def test_update_keeps_id_and_created_utc(conn: psycopg.Connection[TupleRow]) -> None:
    case = question([passage(conn)])
    created = store.create(conn, case, now=lambda: NOW)
    updated = store.update(
        conn,
        created.id,
        case.model_copy(
            update={"note": "checked source", "payload": case.payload | {"answer": "3"}}
        ),
    )
    assert (updated.id, updated.created_utc) == (created.id, NOW)
    assert updated.note == "checked source"
    assert updated.payload["answer"] == "3"
    with pytest.raises(LabelValidationError):
        store.update(conn, created.id, question([passage(conn) | {"text": "wrong source"}]))
    assert store.get(conn, created.id) == updated


def test_delete_then_get_raises_not_found(conn: psycopg.Connection[TupleRow]) -> None:
    created = store.create(conn, question([passage(conn)]), now=lambda: NOW)
    store.delete(conn, created.id)
    with pytest.raises(NotFoundError):
        store.get(conn, created.id)
    with pytest.raises(NotFoundError):
        store.delete(conn, created.id)


def test_set_reviewed_toggles_flag(conn: psycopg.Connection[TupleRow]) -> None:
    created = store.create(conn, question([passage(conn)]), now=lambda: NOW)
    assert store.set_reviewed(conn, created.id, True).human_reviewed
    assert not store.set_reviewed(conn, created.id, False).human_reviewed
    assert store.get(conn, created.id).created_utc == NOW
    with pytest.raises(NotFoundError):
        store.set_reviewed(conn, "missing-label", True)


def test_sample_meetings_is_seeded_and_repeatable(conn: psycopg.Connection[TupleRow]) -> None:
    with pytest.raises(CorpusError):
        store.sample_meetings(conn, "birch")
    ids = [f"birch-label-sample-{number:02}" for number in range(12)]
    for meeting in ids:
        conn.execute(
            "INSERT INTO meetings (id, city_id, body, meeting_date, title, source_key) "
            "VALUES (%s, 'birch', 'City Council', '2024-01-01', 'Sample agenda', %s)",
            (meeting, meeting),
        )
        conn.execute(
            "INSERT INTO documents (id, meeting_id, city_id, kind, source_url, media_type, "
            "unit_kind, status, unit_count) VALUES (%s, %s, 'birch', 'agenda', "
            "'https://fixture.invalid/agenda', 'application/pdf', 'page', 'downloaded', 1)",
            (meeting + "-agenda", meeting),
        )
    candidates = sorted([*ids, "birch-201", "birch-202"])
    expected = random.Random(load_cities()["birch"].sample_seed).sample(candidates, 10)
    assert store.sample_meetings(conn, "birch") == expected
    assert store.sample_meetings(conn, "birch") == expected
    assert len(set(expected)) == 10


def test_sample_meetings_sorts_ids_as_python_strings(
    conn: psycopg.Connection[TupleRow], monkeypatch: pytest.MonkeyPatch
) -> None:
    city = "label-sample-city"
    config = load_cities()["birch"]
    seed = config.sample_seed
    monkeypatch.setattr(store, "load_cities", lambda: {city: config})
    conn.execute(
        "INSERT INTO cities (id, name, state, item_format) "
        "VALUES (%s, 'Sample City', 'WA', 'legistar')",
        (city,),
    )
    ids = [
        f"label-sample-{suffix}"
        for suffix in ("A", "a", "B", "b", "-", "_", ".", "!", "Z", "z", "0", "9")
    ]
    for meeting in ids:
        conn.execute(
            "INSERT INTO meetings (id, city_id, body, meeting_date, title, source_key) "
            "VALUES (%s, %s, 'City Council', '2024-01-01', 'Sample agenda', %s)",
            (meeting, city, meeting),
        )
        conn.execute(
            "INSERT INTO documents "
            "(id, meeting_id, city_id, kind, source_url, media_type, "
            "unit_kind, status, unit_count) "
            "VALUES (%s, %s, %s, 'agenda', 'https://fixture.invalid/agenda', "
            "'application/pdf', 'page', 'downloaded', 1)",
            (meeting + "-agenda", meeting, city),
        )
    assert store.sample_meetings(conn, city) == random.Random(seed).sample(sorted(ids), 10)


def test_payload_and_city_errors_are_collected(conn: psycopg.Connection[TupleRow]) -> None:
    case = question([passage(conn)]).model_copy(update={"city": "missing-city", "payload": {}})
    errors = messages(conn, case)
    assert errors[0].startswith("payload:")
    assert errors[-1] == "unknown city missing-city"


def test_absence_queries_must_be_distinct(conn: psycopg.Connection[TupleRow]) -> None:
    case = unanswerable()
    case.payload["absence_searches"] = [
        {"query": query, "hits": 0} for query in ("quarry", "QUARRY", "stone mine")
    ]
    assert messages(conn, case) == ["absence searches must have distinct lowercased queries"]


@pytest.mark.parametrize(
    "answer_type,answer", [("number", "four"), ("date", "March 12"), ("date", "2024-02-30")]
)
def test_agent_task_answer_must_match_answer_type(
    conn: psycopg.Connection[TupleRow], answer_type: str, answer: str
) -> None:
    case = LabelIn(
        type="agent_task",
        city="birch",
        author="human",
        payload={
            "question": "Which contracts were approved?",
            "answer": answer,
            "answer_type": answer_type,
            "document_ids": ["birch-agenda-201", "birch-minutes-201"],
        },
    )
    assert messages(conn, case) == [f"answer does not match answer_type {answer_type}"]
    assert store.list_labels(conn) == []


def test_agent_task_documents_must_be_distinct_and_local(
    conn: psycopg.Connection[TupleRow],
) -> None:
    case = LabelIn(
        type="agent_task",
        city="birch",
        author="human",
        payload={
            "question": "Which contracts were approved?",
            "answer": "Sidewalk repair",
            "answer_type": "free_text",
            "document_ids": ["birch-agenda-201", "birch-minutes-201"],
        },
    )
    assert store.create(conn, case, now=lambda: NOW)
    for documents, expected in (
        (["birch-agenda-201"] * 2, "agent_task: needs at least two distinct documents"),
        (["birch-agenda-201", "alder-agenda-101"], "agent_task: unknown document alder-agenda-101"),
        (["birch-agenda-201", "missing-document"], "agent_task: unknown document missing-document"),
    ):
        assert expected in messages(
            conn, case.model_copy(update={"payload": case.payload | {"document_ids": documents}})
        )
    conn.execute("UPDATE documents SET status = 'failed' WHERE id = 'birch-agenda-201'")
    assert messages(conn, case) == ["agent_task: unknown document birch-agenda-201"]


@pytest.mark.parametrize("text", ["Short?", "A sufficiently long question", "x" * 300 + "?"])
def test_question_text_bounds(conn: psycopg.Connection[TupleRow], text: str) -> None:
    error = "question must be 10 to 300 characters and end with a question mark"
    assert error in messages(conn, question([passage(conn)], question=text))
    absent = unanswerable()
    absent.payload["question"] = text
    assert error in messages(conn, absent)
    task = LabelIn(
        type="agent_task",
        city="birch",
        author="human",
        payload={
            "question": text,
            "answer": "Approved",
            "answer_type": "free_text",
            "document_ids": ["birch-agenda-201", "birch-minutes-201"],
        },
    )
    assert messages(conn, task) == [error]
