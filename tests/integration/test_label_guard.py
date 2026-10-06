from pathlib import Path
from unittest.mock import Mock

import psycopg
import pytest
from psycopg.rows import TupleRow
from typer.testing import CliRunner

from minutes import cli, db
from minutes.config import get_settings
from minutes.errors import LabelsNotFrozenError, LabelValidationError
from minutes.ingest import pipeline
from minutes.labels import store
from minutes.labels.schema import LabelIn

pytestmark = pytest.mark.integration
Connection = psycopg.Connection[TupleRow]


@pytest.fixture(autouse=True)
def label_environment(fixture_corpus: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    require_frozen = store.require_frozen
    monkeypatch.setattr(
        store, "require_frozen", lambda labels_dir=tmp_path: require_frozen(tmp_path)
    )
    monkeypatch.setattr(cli, "LABELS_DIR", tmp_path)
    monkeypatch.setattr(get_settings(), "data_dir", tmp_path)


def test_quality_stages_refuse_before_label_freeze(
    conn: Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as patch:
        execute = Mock(side_effect=AssertionError("database accessed before freeze guard"))
        patch.setattr(conn, "execute", execute)
        for stage in pipeline.LABEL_GATED_STAGES:
            with pytest.raises(LabelsNotFrozenError, match="labels are not frozen"):
                pipeline.run_stage(conn, stage, "birch-minutes-201", "dev")
        execute.assert_not_called()
    monkeypatch.setattr(pipeline, "should_run", lambda *_: False)
    for stage in ("download", "parse", "ocr"):
        assert pipeline.run_stage(conn, stage, "birch-minutes-201", "dev").status == "skipped"
    for stage in pipeline.STAGES:
        assert pipeline.run_stage(conn, stage, "birch-minutes-201", "fixture").status == "skipped"
    (tmp_path / "manifest.json").write_text('{"frozen": true}', encoding="utf-8")
    for stage in pipeline.LABEL_GATED_STAGES:
        assert pipeline.run_stage(conn, stage, "birch-minutes-201", "dev").status == "skipped"


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
                {"query": query, "hits": 0} for query in ("quarry", "stone mine", "zeppelin")
            ],
        },
    )


def keep_one_output(conn: Connection, table: str) -> None:
    if table in ("chunks", "facts"):
        conn.execute(f"UPDATE {table} SET item_id = NULL")
    for other in ("facts", "chunks", "items"):
        if other != table:
            conn.execute(f"DELETE FROM {other}")
    conn.execute(f"DELETE FROM {table} WHERE id != (SELECT min(id) FROM {table})")
    for other in ("items", "chunks", "facts"):
        assert conn.execute(f"SELECT count(*) FROM {other}").fetchone() == (
            1 if other == table else 0,
        )


def test_labels_refused_when_derived_outputs_exist(
    conn: Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn.execute("DELETE FROM facts")
    conn.execute("DELETE FROM items")
    conn.execute("DELETE FROM chunks WHERE id != (SELECT min(id) FROM chunks)")
    row = conn.execute("SELECT count(*) FROM chunks").fetchone()
    assert row == (1,)
    monkeypatch.setattr(get_settings(), "corpus", "dev")
    with pytest.raises(LabelValidationError) as error:
        store.create(conn, unanswerable(), labels_dir=tmp_path)
    assert error.value.args[0] == [
        "derived outputs exist; labels must be written from source pages only"
    ]
    assert store.list_labels(conn) == []


@pytest.mark.parametrize("table", ["items", "facts"])
def test_labels_refused_when_items_or_facts_exist(
    conn: Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, table: str
) -> None:
    keep_one_output(conn, table)
    monkeypatch.setattr(get_settings(), "corpus", "dev")
    with pytest.raises(LabelValidationError) as error:
        store.create(conn, unanswerable(), labels_dir=tmp_path)
    assert error.value.args[0] == [
        "derived outputs exist; labels must be written from source pages only"
    ]
    assert store.list_labels(conn) == []


@pytest.mark.parametrize("table", ["items", "chunks", "facts"])
def test_update_refused_when_derived_outputs_exist(
    conn: Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, table: str
) -> None:
    label = unanswerable()
    created = store.create(conn, label, labels_dir=tmp_path)
    keep_one_output(conn, table)
    monkeypatch.setattr(get_settings(), "corpus", "dev")
    label.note = "rechecked source"
    with pytest.raises(LabelValidationError) as error:
        store.update(conn, created.id, label, labels_dir=tmp_path)
    assert error.value.args[0] == [
        "derived outputs exist; labels must be written from source pages only"
    ]
    assert store.get(conn, created.id) == created


def test_search_cli_refuses_before_freeze_on_dev_corpus(
    fixture_corpus: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings(), "corpus", "dev")
    monkeypatch.setattr(cli, "corpus_database_url", lambda _: fixture_corpus)
    connect = Mock(side_effect=AssertionError("database accessed before freeze guard"))
    monkeypatch.setattr(db, "connect", connect)
    for command in ("search", "ask"):
        result = CliRunner().invoke(cli.app, [command, "sidewalk repair"])
        assert result.exit_code == 1
        assert "labels are not frozen" in result.stderr
    connect.assert_not_called()


def test_ingest_run_refuses_before_enqueue_on_dev_corpus(
    fixture_corpus: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings(), "corpus", "dev")
    monkeypatch.setattr(cli, "corpus_database_url", lambda _: fixture_corpus)
    connect = Mock(side_effect=AssertionError("database accessed before freeze guard"))
    monkeypatch.setattr(db, "connect", connect)
    for stages in ("segment", "parse,ocr,segment"):
        result = CliRunner().invoke(cli.app, ["ingest", "run", "--stages", stages])
        assert result.exit_code == 1
        assert "labels are not frozen" in result.stderr
    connect.assert_not_called()
