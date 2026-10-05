from pathlib import Path

import psycopg
import pytest
from psycopg.rows import TupleRow
from typer.testing import CliRunner

from minutes import db
from minutes.cli import app
from minutes.errors import DatabaseError, MigrationError

pytestmark = pytest.mark.integration

TABLES = {
    "cities",
    "meetings",
    "documents",
    "units",
    "items",
    "chunks",
    "facts",
    "jobs",
    "stage_runs",
    "labels",
    "llm_calls",
    "answer_cache",
    "schema_migrations",
}
VIEWS = {
    "v_meetings",
    "v_documents",
    "v_units",
    "v_items",
    "v_facts",
    "v_motions",
    "v_votes",
    "v_vote_members",
    "v_ordinances",
    "v_amounts",
    "v_statements",
}


def test_migrate_creates_all_tables_and_views(conn: psycopg.Connection[TupleRow]) -> None:
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
    ).fetchall()
    views = conn.execute(
        "SELECT table_name FROM information_schema.views WHERE table_schema = 'public'"
    ).fetchall()
    assert {row[0] for row in tables} == TABLES
    assert {row[0] for row in views} == VIEWS


def test_migrate_is_idempotent(test_db_url: str) -> None:
    assert db.migrate(test_db_url) == []


def test_rollback_then_migrate_restores_schema(test_db_url: str) -> None:
    assert db.rollback(test_db_url, steps=1) == ["0001"]
    with db.connect(test_db_url) as conn:
        units = conn.execute("SELECT to_regclass('public.units')").fetchone()
    assert units == (None,)
    assert db.migrate(test_db_url) == ["0001"]


def test_vector_extension_and_hnsw_index_exist(conn: psycopg.Connection[TupleRow]) -> None:
    extension = conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone()
    index = conn.execute(
        "SELECT indexdef FROM pg_indexes WHERE indexname = 'chunks_embedding_idx'"
    ).fetchone()
    assert extension is not None
    assert index is not None
    assert "hnsw" in index[0]


def test_units_check_rejects_segment_without_times(conn: psycopg.Connection[TupleRow]) -> None:
    conn.execute("INSERT INTO cities VALUES ('birch', 'Birch', 'CA', 'granicus')")
    conn.execute(
        "INSERT INTO meetings (id, city_id, body, meeting_date, title, source_key) "
        "VALUES ('birch-201', 'birch', 'City Council', '2024-03-12', 'City Council', '201')"
    )
    conn.execute(
        "INSERT INTO documents (id, meeting_id, city_id, kind, source_url, media_type, unit_kind) "
        "VALUES ('birch-transcript-201', 'birch-201', 'birch', 'transcript', "
        "'file:birch-transcript-201.vtt', 'text/vtt', 'segment')"
    )
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute(
            "INSERT INTO units (document_id, unit_index, unit_kind, text, text_source, end_ms) "
            "VALUES ('birch-transcript-201', 1, 'segment', 'good evening', 'caption', 4000)"
        )


def test_reset_refuses_non_local_host() -> None:
    with pytest.raises(DatabaseError):
        db.reset("postgresql://u:p@db.internal:5432/x_test")


def test_failed_migration_rolls_back_only_that_file(
    test_db_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "0001_first.sql").write_text("CREATE TABLE first (id integer);")
    (tmp_path / "0002_second.sql").write_text("CREATE TABLE second (id nosuchtype);")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(db, "MIGRATIONS_DIR", tmp_path)
            with pytest.raises(MigrationError, match="0002"):
                db.reset(test_db_url)
        with psycopg.connect(test_db_url) as conn:
            versions = conn.execute("SELECT version FROM schema_migrations").fetchall()
            second = conn.execute("SELECT to_regclass('public.second')").fetchone()
        assert versions == [("0001",)]
        assert second == (None,)
    finally:
        # later tests share this database and need the real schema back
        db.reset(test_db_url)


def test_db_commands_print_what_they_did(test_db_url: str) -> None:
    runner = CliRunner()
    assert runner.invoke(app, ["db", "migrate"]).output == "up to date\n"
    assert runner.invoke(app, ["db", "rollback"]).output == "rolled back 0001\n"
    assert runner.invoke(app, ["db", "migrate"]).output == "applied 0001\n"
    assert runner.invoke(app, ["db", "reset", "--yes"]).output == "reset minutes_test\n"


def test_db_reset_refuses_without_yes(test_db_url: str) -> None:
    result = CliRunner().invoke(app, ["db", "reset"])
    assert result.exit_code == 2
    assert result.output == "refusing without --yes\n"
