import hashlib
import json
import re
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import TupleRow

from minutes.api import routes_core, routes_labels
from minutes.api.app import create_app
from minutes.config import get_settings
from minutes.labels import export, store
from minutes.labels.rules import FIXTURE_RULES
from minutes.labels.schema import Label, LabelIn

pytestmark = pytest.mark.integration
Connection = psycopg.Connection[TupleRow]


@pytest.fixture(autouse=True)
def label_paths(fixture_corpus: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes_labels, "LABELS_DIR", tmp_path)
    monkeypatch.setattr(routes_labels, "RULES", FIXTURE_RULES)
    monkeypatch.setattr(get_settings(), "data_dir", tmp_path)


@pytest.fixture
def client(conn: Connection) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[routes_core.get_conn] = lambda: conn
    with TestClient(app) as client:
        yield client


def question(conn: Connection, city: str = "birch") -> LabelIn:
    document = "birch-minutes-201" if city == "birch" else "alder-minutes-101"
    row = conn.execute(
        "SELECT text FROM units WHERE document_id = %s AND unit_index = 1", (document,)
    ).fetchone()
    assert row is not None
    text: str = row[0][:50]
    passage = {
        "text": text,
        "end": len(text),
        "start": 0,
        "unit_index": 1,
        "document_id": document,
    }
    return LabelIn(
        type="question",
        city=city,
        author="human",
        note="source checked: caf\u00e9",
        payload={
            "passages": [passage],
            "answer_type": "number",
            "answer": "4",
            "question_type": "vote",
            "answerable": True,
            "question": "How many supported the proposal?",
        },
    )


def questions(conn: Connection, tmp_path: Path) -> list[Label]:
    labels = []
    for city in ("birch", "alder"):
        labels.append(store.create(conn, question(conn, city), labels_dir=tmp_path))
        labels.append(
            store.create(
                conn,
                LabelIn(
                    type="question",
                    city=city,
                    author="human",
                    payload={
                        "absence_searches": [
                            {"hits": 0, "query": query}
                            for query in ("quarry", "stone mine", "zeppelin")
                        ],
                        "reason": "entity_not_in_corpus",
                        "answerable": False,
                        "question": "Was the quarry approved?",
                    },
                ),
                labels_dir=tmp_path,
            )
        )
    return labels


def test_post_valid_question_returns_201_with_id(client: TestClient, conn: Connection) -> None:
    response = client.post("/api/labels", json=question(conn).model_dump())
    assert response.status_code == 201
    body = response.json()
    assert body["id"] == "q-birch-001"
    assert body["type"] == "question"
    assert body["human_reviewed"] is False
    assert store.get(conn, body["id"]).payload == question(conn).payload
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", body["created_utc"])


def test_post_invalid_passage_returns_422_label_invalid_with_messages(
    client: TestClient, conn: Connection
) -> None:
    label = question(conn)
    label.payload["passages"] = [
        {
            "document_id": "birch-minutes-201",
            "unit_index": 1,
            "start": 0,
            "end": 50,
            "text": "wrong source text",
        }
    ]
    response = client.post("/api/labels", json=label.model_dump())
    assert response.status_code == 422
    assert response.json() == {
        "error": "label_invalid",
        "detail": ["passage 1: text does not match stored unit text"],
    }
    assert store.list_labels(conn) == []


def test_put_updates_and_delete_returns_204(client: TestClient, conn: Connection) -> None:
    label = question(conn)
    created = client.post("/api/labels", json=label.model_dump()).json()
    label.note = "rechecked source"
    label.payload["answer"] = "3"
    updated = client.put(f"/api/labels/{created['id']}", json=label.model_dump())
    assert updated.status_code == 200
    assert updated.json()["created_utc"] == created["created_utc"]
    assert updated.json()["payload"]["answer"] == "3"
    assert updated.json()["note"] == "rechecked source"
    wrong_city = client.put(
        f"/api/labels/{created['id']}", json=question(conn, "alder").model_dump()
    )
    assert wrong_city.status_code == 422
    assert wrong_city.json()["detail"] == ["label type and city cannot change"]
    deleted = client.delete(f"/api/labels/{created['id']}")
    assert deleted.status_code == 204
    assert deleted.content == b""
    assert client.get(f"/api/labels/{created['id']}").status_code == 404


def test_get_unknown_label_returns_404(client: TestClient) -> None:
    response = client.get("/api/labels/missing-label")
    assert response.status_code == 404
    assert response.json() == {"error": "not_found", "detail": "unknown label missing-label"}


def test_review_endpoint_sets_flag(client: TestClient, conn: Connection) -> None:
    created = client.post("/api/labels", json=question(conn).model_dump()).json()
    for reviewed in (True, False):
        response = client.post(
            f"/api/labels/{created['id']}/review", json={"human_reviewed": reviewed}
        )
        assert response.status_code == 200
        assert response.json()["human_reviewed"] is reviewed
        assert store.get(conn, created["id"]).human_reviewed is reviewed


def test_list_filters_by_type_and_city_sorted_by_id(
    client: TestClient, conn: Connection, tmp_path: Path
) -> None:
    labels = questions(conn, tmp_path)
    store.create(
        conn,
        LabelIn(
            type="agent_task",
            city="birch",
            author="human",
            payload={
                "question": "Which contracts were approved?",
                "answer": "Sidewalk repair",
                "answer_type": "free_text",
                "document_ids": ["birch-agenda-201", "birch-minutes-201"],
            },
        ),
        labels_dir=tmp_path,
    )
    response = client.get("/api/labels", params={"type": "question", "city": "birch"})
    assert response.status_code == 200
    assert [label["id"] for label in response.json()["labels"]] == sorted(
        label.id for label in labels if label.city == "birch"
    )
    all_labels = client.get("/api/labels").json()["labels"]
    assert len(all_labels) == 5
    assert [label["id"] for label in all_labels] == sorted(label["id"] for label in all_labels)
    assert client.get("/api/labels", params={"city": "missing-city"}).status_code == 422
    assert client.get("/api/labels", params={"type": "missing-type"}).status_code == 422


def test_text_search_endpoint_returns_count_and_snippets(client: TestClient) -> None:
    response = client.get("/api/labels/text-search", params={"city": "birch", "q": "granite works"})
    assert response.status_code == 200
    body = response.json()
    assert body["unit_count"] == 3
    assert len(body["matches"]) == 3
    assert all("granite works" in match["snippet"].lower() for match in body["matches"])
    pairs = [(match["document_id"], match["unit_index"]) for match in body["matches"]]
    assert pairs == sorted(pairs)
    for params in ({"city": "birch", "q": "x"}, {"city": "missing", "q": "granite"}):
        assert client.get("/api/labels/text-search", params=params).status_code == 422


def test_sample_meetings_endpoint_marks_labelled(
    client: TestClient, conn: Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store, "sample_meetings", lambda *_: ["birch-202", "birch-201"])
    store.create(
        conn,
        LabelIn(
            type="agenda_count",
            city="birch",
            author="human",
            payload={
                "meeting_id": "birch-201",
                "count": 1,
                "items": [{"identifier": "9.B", "title": "Sidewalk repair", "start_page": 2}],
            },
        ),
        labels_dir=tmp_path,
    )
    response = client.get("/api/labels/sample-meetings", params={"city": "birch"})
    assert response.status_code == 200
    meetings = response.json()["meetings"]
    assert [meeting["meeting_id"] for meeting in meetings] == ["birch-202", "birch-201"]
    assert [meeting["labelled"] for meeting in meetings] == [False, True]
    assert meetings[1]["agenda_document_id"] == "birch-agenda-201"
    assert meetings[1]["meeting_date"] == "2024-03-12"


def test_progress_reports_counts_violations_and_reviewed(
    client: TestClient, conn: Connection, tmp_path: Path
) -> None:
    created = store.create(conn, question(conn), labels_dir=tmp_path)
    store.set_reviewed(conn, created.id, True, labels_dir=tmp_path)
    response = client.get("/api/labels/progress")
    assert response.status_code == 200
    body = response.json()
    assert body["frozen"] is False
    for label_type, rule in zip(export.LABEL_FILES, ("Q-1", "C-1", "X-1", "A-1"), strict=True):
        assert body["types"][label_type]["count"] == (1 if label_type == "question" else 0)
        assert body["types"][label_type]["violations"][0].startswith(rule)
        assert body["human_reviewed"][label_type] == (1 if label_type == "question" else 0)


def test_export_refuses_incomplete_set_with_409(
    client: TestClient, conn: Connection, tmp_path: Path
) -> None:
    response = client.post("/api/labels/export", json={"type": "question"})
    assert response.status_code == 409
    assert response.json()["error"] == "label_rules"
    assert response.json()["detail"][0].startswith("Q-1")
    assert not (tmp_path / "questions.jsonl").exists()
    questions(conn, tmp_path)
    assert client.post("/api/labels/export", json={"type": "question"}).status_code == 200
    before = (tmp_path / "questions.jsonl").read_bytes()
    store.create(conn, question(conn), labels_dir=tmp_path)
    response = client.post("/api/labels/export", json={"type": "question"})
    assert response.status_code == 409
    assert response.json()["detail"][0] == "Q-1: expected 4, found 5 (total)"
    assert (tmp_path / "questions.jsonl").read_bytes() == before


def test_export_is_byte_identical_for_same_labels(
    client: TestClient, conn: Connection, tmp_path: Path
) -> None:
    questions(conn, tmp_path)
    first = client.post("/api/labels/export", json={"type": "question"})
    assert first.status_code == 200
    path = Path(first.json()["path"])
    first_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    second = client.post("/api/labels/export", json={"type": "question"})
    assert second.status_code == 200
    assert second.json()["count"] == 4
    assert hashlib.sha256(path.read_bytes()).hexdigest() == first_sha


def test_export_file_format(client: TestClient, conn: Connection, tmp_path: Path) -> None:
    labels = questions(conn, tmp_path)
    response = client.post("/api/labels/export", json={"type": "question"})
    assert response.status_code == 200
    content = (tmp_path / "questions.jsonl").read_bytes()
    assert content.endswith(b"\n") and not content.endswith(b"\n\n")
    assert b"\r" not in content
    assert "caf\u00e9".encode() in content
    assert b"\\u00e9" not in content
    lines = content.decode("utf-8").splitlines()
    assert len(lines) == 4
    cases = [json.loads(line) for line in lines]
    assert [case["id"] for case in cases] == sorted(label.id for label in labels)
    for line, case in zip(lines, cases, strict=True):
        assert line == json.dumps(case, ensure_ascii=False, separators=(",", ":"))
        assert list(case) == [
            "id",
            "type",
            "city",
            "author",
            "created_utc",
            "human_reviewed",
            "note",
            "payload",
        ]
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", case["created_utc"])
        payload = case["payload"]
        if payload["answerable"]:
            assert list(payload) == [
                "question",
                "answerable",
                "question_type",
                "answer",
                "answer_type",
                "passages",
            ]
            assert list(payload["passages"][0]) == [
                "document_id",
                "unit_index",
                "start",
                "end",
                "text",
            ]
        else:
            assert list(payload) == ["question", "answerable", "reason", "absence_searches"]
            assert list(payload["absence_searches"][0]) == ["query", "hits"]


def test_create_after_freeze_is_rejected(
    client: TestClient, conn: Connection, tmp_path: Path
) -> None:
    label = question(conn)
    created = client.post("/api/labels", json=label.model_dump()).json()
    (tmp_path / "manifest.json").write_text('{"frozen": true}', encoding="utf-8")
    responses = [
        client.post("/api/labels", json=label.model_dump()),
        client.put(f"/api/labels/{created['id']}", json=label.model_dump()),
        client.delete(f"/api/labels/{created['id']}"),
        client.post(f"/api/labels/{created['id']}/review", json={"human_reviewed": True}),
        client.post("/api/labels/export", json={"type": "question"}),
    ]
    for response in responses:
        assert response.status_code == 409
        assert response.json() == {"error": "labels_frozen"}
    assert len(store.list_labels(conn)) == 1
    assert store.get(conn, created["id"]).human_reviewed is False
    assert client.get("/api/labels/progress").json()["frozen"] is True


def test_label_endpoints_do_not_touch_pipeline_tables(
    client: TestClient, conn: Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store, "sample_meetings", lambda *_: ["birch-201", "birch-202"])
    with conn.transaction(force_rollback=True):
        for table in ("items", "chunks", "facts"):
            conn.execute(f"ALTER TABLE {table} RENAME TO hidden_{table}")
        labels = questions(conn, tmp_path)
        label = question(conn)
        created = client.post("/api/labels", json=label.model_dump())
        assert created.status_code == 201
        id = created.json()["id"]
        assert client.get("/api/labels").status_code == 200
        assert client.get(f"/api/labels/{id}").status_code == 200
        assert client.put(f"/api/labels/{id}", json=label.model_dump()).status_code == 200
        assert (
            client.post(f"/api/labels/{id}/review", json={"human_reviewed": True}).status_code
            == 200
        )
        assert client.get("/api/labels/progress").status_code == 200
        assert (
            client.get(
                "/api/labels/text-search", params={"city": "birch", "q": "granite"}
            ).status_code
            == 200
        )
        assert (
            client.get("/api/labels/sample-meetings", params={"city": "birch"}).status_code == 200
        )
        assert client.delete(f"/api/labels/{id}").status_code == 204
        assert client.post("/api/labels/export", json={"type": "question"}).status_code == 200
        assert len(labels) == 4
