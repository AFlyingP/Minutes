import hashlib
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import TupleRow
from typer.testing import CliRunner

from minutes import cli, db
from minutes.api import routes_core, routes_labels
from minutes.api.app import create_app
from minutes.config import get_settings
from minutes.labels import export, store
from minutes.labels.rules import FIXTURE_RULES
from minutes.labels.schema import Label, LabelIn

pytestmark = pytest.mark.integration
Connection = psycopg.Connection[TupleRow]
runner = CliRunner()


@pytest.fixture(autouse=True)
def label_environment(
    fixture_corpus: str, conn: Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    @contextmanager
    def connect(url: str) -> Iterator[Connection]:
        assert url == fixture_corpus
        yield conn

    monkeypatch.setattr(db, "connect", connect)
    monkeypatch.setattr(cli, "LABELS_DIR", tmp_path)
    monkeypatch.setattr(cli, "LABEL_RULES", FIXTURE_RULES)
    monkeypatch.setattr(routes_labels, "LABELS_DIR", tmp_path)
    monkeypatch.setattr(routes_labels, "RULES", FIXTURE_RULES)
    monkeypatch.setattr(get_settings(), "data_dir", tmp_path)
    monkeypatch.setattr(
        store,
        "sample_meetings",
        lambda _conn, city: (
            ["birch-201", "birch-202"] if city == "birch" else ["alder-101", "alder-102"]
        ),
    )


def question(conn: Connection, city: str = "birch") -> LabelIn:
    document = "birch-minutes-201" if city == "birch" else "alder-minutes-101"
    row = conn.execute(
        "SELECT text FROM units WHERE document_id = %s AND unit_index = 1", (document,)
    ).fetchone()
    assert row is not None
    text: str = row[0][:50]
    return LabelIn(
        type="question",
        city=city,
        author="human",
        note="checked source",
        payload={
            "question": "How many supported the proposal?",
            "answerable": True,
            "question_type": "vote",
            "answer": "4",
            "answer_type": "number",
            "passages": [
                {
                    "document_id": document,
                    "unit_index": 1,
                    "start": 0,
                    "end": len(text),
                    "text": text,
                }
            ],
        },
    )


def complete_labels(conn: Connection, tmp_path: Path) -> list[Label]:
    labels = []
    for city, meetings in (("alder", ("101", "102")), ("birch", ("201", "202"))):
        labels.append(store.create(conn, question(conn, city), labels_dir=tmp_path))
        labels.append(
            store.create(
                conn,
                LabelIn(
                    type="question",
                    city=city,
                    author="human",
                    payload={
                        "question": "Was the quarry approved?",
                        "answerable": False,
                        "reason": "entity_not_in_corpus",
                        "absence_searches": [
                            {"query": query, "hits": 0}
                            for query in ("quarry", "stone mine", "zeppelin")
                        ],
                    },
                ),
                labels_dir=tmp_path,
            )
        )
        for meeting in meetings:
            labels.append(
                store.create(
                    conn,
                    LabelIn(
                        type="agenda_count",
                        city=city,
                        author="human",
                        payload={
                            "meeting_id": f"{city}-{meeting}",
                            "count": 1,
                            "items": [
                                {"identifier": "1.A", "title": "Business item", "start_page": 2}
                            ],
                        },
                    ),
                    labels_dir=tmp_path,
                )
            )
        row = conn.execute(
            "SELECT text FROM units WHERE document_id = %s AND unit_index = 1",
            (f"{city}-minutes-{meetings[0]}",),
        ).fetchone()
        assert row is not None
        text: str = row[0][:50]
        for _ in range(3):
            labels.append(
                store.create(
                    conn,
                    LabelIn(
                        type="extraction",
                        city=city,
                        author="human",
                        payload={
                            "kind": "statement",
                            "passage": {
                                "document_id": f"{city}-minutes-{meetings[0]}",
                                "unit_index": 1,
                                "start": 0,
                                "end": len(text),
                                "text": text,
                            },
                            "record": {
                                "speaker": "Minutes",
                                "role": None,
                                "summary": "Meeting heading",
                            },
                        },
                    ),
                    labels_dir=tmp_path,
                )
            )
        labels.append(
            store.create(
                conn,
                LabelIn(
                    type="agent_task",
                    city=city,
                    author="human",
                    payload={
                        "question": "Which contracts were approved?",
                        "answer": "Sidewalk repair",
                        "answer_type": "free_text",
                        "document_ids": [
                            f"{city}-agenda-{meetings[0]}",
                            f"{city}-minutes-{meetings[0]}",
                        ],
                    },
                ),
                labels_dir=tmp_path,
            )
        )
    return sorted(labels, key=lambda label: label.id)


def test_cli_add_and_api_share_validation(conn: Connection, tmp_path: Path) -> None:
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
    payload_file = tmp_path / "payload.json"
    payload_file.write_text(json.dumps(label.payload), encoding="utf-8")
    result = runner.invoke(
        cli.app,
        [
            "labels",
            "add",
            "--type",
            "question",
            "--city",
            "birch",
            "--payload-file",
            str(payload_file),
        ],
    )
    assert result.exit_code == 1
    assert result.stderr.splitlines() == ["error: passage 1: text does not match stored unit text"]
    app = create_app()
    app.dependency_overrides[routes_core.get_conn] = lambda: conn
    with TestClient(app) as client:
        response = client.post("/api/labels", json=label.model_dump())
    assert response.status_code == 422
    assert response.json() == {
        "error": "label_invalid",
        "detail": ["passage 1: text does not match stored unit text"],
    }
    assert store.list_labels(conn) == []


def test_cli_page_find_prints_offsets(conn: Connection) -> None:
    result = runner.invoke(
        cli.app,
        [
            "labels",
            "page",
            "birch-minutes-201",
            "1",
            "--find",
            "Granite Works Inc.",
        ],
    )
    assert result.exit_code == 0
    lines = result.stdout.splitlines()
    assert len(lines) == 1
    match = re.fullmatch(r"start=(\d+) end=(\d+) text=Granite Works Inc\.", lines[0])
    assert match is not None
    start, end = map(int, match.groups())
    assert end == start + len("Granite Works Inc.")
    row = conn.execute(
        "SELECT text FROM units WHERE document_id = 'birch-minutes-201' AND unit_index = 1"
    ).fetchone()
    assert row is not None
    assert row[0][start:end] == "Granite Works Inc."
    absent = runner.invoke(
        cli.app,
        [
            "labels",
            "page",
            "birch-minutes-201",
            "1",
            "--find",
            "absent text",
        ],
    )
    assert absent.exit_code == 1
    conn.execute(
        "UPDATE units SET text = 'banana banana' WHERE document_id = 'birch-minutes-201' "
        "AND unit_index = 1"
    )
    repeated = runner.invoke(
        cli.app,
        [
            "labels",
            "page",
            "birch-minutes-201",
            "1",
            "--find",
            "ana",
        ],
    )
    assert repeated.exit_code == 0
    assert repeated.stdout.splitlines() == ["start=1 end=4 text=ana", "start=8 end=11 text=ana"]


def test_backup_and_restore_round_trip(conn: Connection, tmp_path: Path) -> None:
    labels = complete_labels(conn, tmp_path)
    reviewed = store.set_reviewed(conn, labels[0].id, True, labels_dir=tmp_path)
    expected = [reviewed, *labels[1:]]
    result = runner.invoke(cli.app, ["labels", "backup"])
    assert result.exit_code == 0
    assert result.stdout == "backed up 16 labels\n"
    path = tmp_path / "labels_backup" / "labels.jsonl"
    assert path.is_file()
    assert len(path.read_text(encoding="utf-8").splitlines()) == 16
    conn.execute("DELETE FROM labels")
    store.create(conn, question(conn), labels_dir=tmp_path)
    restored = runner.invoke(cli.app, ["labels", "restore"])
    assert restored.exit_code == 0
    assert restored.stdout == "restored 16 labels\n"
    assert store.list_labels(conn) == expected
    conn.execute("DELETE FROM labels")
    assert runner.invoke(cli.app, ["labels", "backup"]).stdout == "backed up 0 labels\n"
    assert path.read_bytes() == b""
    assert runner.invoke(cli.app, ["labels", "restore"]).stdout == "restored 0 labels\n"


def test_export_exit_code_4_on_rule_violation(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["labels", "export", "--type", "question"])
    assert result.exit_code == 4
    lines = result.stderr.splitlines()
    assert lines[0] == "error: Q-1: expected 4, found 0 (total)"
    assert all(line.startswith("error: Q-") for line in lines)
    assert len(lines) > 1
    assert not (tmp_path / "questions.jsonl").exists()


def test_freeze_requires_four_complete_files(conn: Connection, tmp_path: Path) -> None:
    missing = runner.invoke(cli.app, ["labels", "freeze"])
    assert missing.exit_code == 4
    assert missing.stderr.splitlines() == [
        f"error: missing {name}" for name in export.LABEL_FILES.values()
    ]
    assert not (tmp_path / "manifest.json").exists()
    complete_labels(conn, tmp_path)
    assert runner.invoke(cli.app, ["labels", "export", "--all"]).exit_code == 0
    path = tmp_path / "questions.jsonl"
    content = path.read_bytes()
    path.write_bytes(b"\n".join(content.splitlines()[:-1]) + b"\n")
    incomplete = runner.invoke(cli.app, ["labels", "freeze"])
    assert incomplete.exit_code == 4
    assert "Q-1: expected 4, found 3" in incomplete.stderr
    assert not (tmp_path / "manifest.json").exists()
    path.write_bytes(content)
    frozen = runner.invoke(cli.app, ["labels", "freeze"])
    assert frozen.exit_code == 0
    assert frozen.stdout == "frozen\n"
    assert store.is_frozen(tmp_path)


def test_freeze_then_amend_updates_manifest(conn: Connection, tmp_path: Path) -> None:
    complete_labels(conn, tmp_path)
    assert runner.invoke(cli.app, ["labels", "export", "--all"]).exit_code == 0
    assert runner.invoke(cli.app, ["labels", "freeze"]).exit_code == 0
    manifest_path = tmp_path / "manifest.json"
    before = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert before["frozen"] is True
    assert before["amendments"] == []
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", before["frozen_utc"])
    for name, sha in before["files"].items():
        assert sha == hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    path = tmp_path / "questions.jsonl"
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    cases[0]["note"] = "rechecked against source"
    path.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False, separators=(",", ":")) for case in cases)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    reason = "Corrected the source note"
    result = runner.invoke(cli.app, ["labels", "amend", "--file", path.name, "--reason", reason])
    assert result.exit_code == 0
    assert result.stdout == "amended questions.jsonl\n"
    after = json.loads(manifest_path.read_text(encoding="utf-8"))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert after["files"][path.name] == sha
    assert after["frozen_utc"] == before["frozen_utc"]
    assert after["frozen"] is True
    assert len(after["amendments"]) == 1
    amendment = after["amendments"][0]
    assert amendment == {
        "utc": amendment["utc"],
        "file": path.name,
        "old_sha256": before["files"][path.name],
        "new_sha256": sha,
        "reason": reason,
    }
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", amendment["utc"])
    assert all(
        after["files"][name] == old for name, old in before["files"].items() if name != path.name
    )
    content = manifest_path.read_bytes()
    assert content.endswith(b"\n") and b"\r" not in content
    assert content.decode() == json.dumps(after, indent=2) + "\n"


def test_amend_unchanged_file_fails(conn: Connection, tmp_path: Path) -> None:
    complete_labels(conn, tmp_path)
    assert runner.invoke(cli.app, ["labels", "export", "--all"]).exit_code == 0
    assert runner.invoke(cli.app, ["labels", "freeze"]).exit_code == 0
    path = tmp_path / "manifest.json"
    before = path.read_bytes()
    result = runner.invoke(
        cli.app,
        [
            "labels",
            "amend",
            "--file",
            "questions.jsonl",
            "--reason",
            "Rechecked the source note",
        ],
    )
    assert result.exit_code == 4
    assert result.stderr == "error: questions.jsonl is unchanged\n"
    assert path.read_bytes() == before


def test_import_all_replaces_rows(conn: Connection, tmp_path: Path) -> None:
    labels = complete_labels(conn, tmp_path)
    assert runner.invoke(cli.app, ["labels", "export", "--all"]).exit_code == 0
    store.set_reviewed(conn, labels[0].id, True, labels_dir=tmp_path)
    added = store.create(conn, question(conn), labels_dir=tmp_path)
    assert len(store.list_labels(conn)) == 17
    assert export.import_all(conn, tmp_path) == 16
    assert store.list_labels(conn) == labels
    assert added.id not in {label.id for label in store.list_labels(conn)}
    (tmp_path / "agent_tasks.jsonl").unlink()
    assert export.import_all(conn, tmp_path) == 14
    assert store.list_labels(conn, "agent_task") == []
    for path in tmp_path.glob("*.jsonl"):
        path.unlink()
    assert export.import_all(conn, tmp_path) == 0
    assert store.list_labels(conn) == []
