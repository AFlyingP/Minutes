from pathlib import Path

import psycopg
import pytest
from psycopg.rows import TupleRow
from typer.testing import CliRunner

from minutes import db
from minutes.cli import app
from minutes.config import get_settings
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


@pytest.fixture(scope="module")
def scratch_url(test_db_url: str) -> str:
    # these tests drop and rebuild the schema, so they get a database of their own
    url = test_db_url.replace("/minutes_test", "/minutes_migrations_test")
    db.reset(url)
    return url


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


def test_migrate_is_idempotent(scratch_url: str) -> None:
    assert db.migrate(scratch_url) == []


def test_rollback_then_migrate_restores_schema(scratch_url: str) -> None:
    assert db.rollback(scratch_url, steps=1) == ["0001"]
    with db.connect(scratch_url) as conn:
        units = conn.execute("SELECT to_regclass('public.units')").fetchone()
    assert units == (None,)
    assert db.migrate(scratch_url) == ["0001"]


def test_rollback_without_a_down_file_is_a_migration_error(
    scratch_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(db, "MIGRATIONS_DIR", tmp_path)
    with pytest.raises(MigrationError, match="no down file for 0001"):
        db.rollback(scratch_url)


def test_lost_connection_is_a_database_error(scratch_url: str) -> None:
    with pytest.raises(DatabaseError, match="database unavailable"), db.connect(scratch_url):
        raise psycopg.OperationalError("server closed the connection")


def test_vector_extension_and_hnsw_index_exist(conn: psycopg.Connection[TupleRow]) -> None:
    extension = conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone()
    index = conn.execute(
        "SELECT indexdef FROM pg_indexes WHERE indexname = 'chunks_embedding_idx'"
    ).fetchone()
    assert extension is not None
    assert index is not None
    assert "hnsw" in index[0]


def test_units_check_rejects_segment_without_times(conn: psycopg.Connection[TupleRow]) -> None:
    conn.execute("INSERT INTO cities VALUES ('cedar', 'Cedar', 'CA', 'granicus')")
    conn.execute(
        "INSERT INTO meetings (id, city_id, body, meeting_date, title, source_key) "
        "VALUES ('cedar-301', 'cedar', 'City Council', '2024-03-12', 'City Council', '301')"
    )
    conn.execute(
        "INSERT INTO documents (id, meeting_id, city_id, kind, source_url, media_type, unit_kind) "
        "VALUES ('cedar-transcript-301', 'cedar-301', 'cedar', 'transcript', "
        "'file:cedar-transcript-301.vtt', 'text/vtt', 'segment')"
    )
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute(
            "INSERT INTO units (document_id, unit_index, unit_kind, text, text_source, end_ms) "
            "VALUES ('cedar-transcript-301', 1, 'segment', 'good evening', 'caption', 4000)"
        )


def test_reset_refuses_non_local_host() -> None:
    with pytest.raises(DatabaseError):
        db.reset("postgresql://u:p@db.internal:5432/x_test")


def test_failed_migration_rolls_back_only_that_file(
    scratch_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "0001_first.sql").write_text("CREATE TABLE first (id integer);")
    (tmp_path / "0002_second.sql").write_text("CREATE TABLE second (id nosuchtype);")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(db, "MIGRATIONS_DIR", tmp_path)
            with pytest.raises(MigrationError, match="0002"):
                db.reset(scratch_url)
        with psycopg.connect(scratch_url) as conn:
            versions = conn.execute("SELECT version FROM schema_migrations").fetchall()
            second = conn.execute("SELECT to_regclass('public.second')").fetchone()
        assert versions == [("0001",)]
        assert second == (None,)
    finally:
        # the tests below need the real schema back
        db.reset(scratch_url)


def test_db_commands_print_what_they_did(scratch_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINUTES_TEST_DATABASE_URL", scratch_url)
    get_settings.cache_clear()
    runner = CliRunner()
    assert runner.invoke(app, ["db", "migrate"]).output == "up to date\n"
    assert runner.invoke(app, ["db", "rollback"]).output == "rolled back 0001\n"
    assert runner.invoke(app, ["db", "migrate"]).output == "applied 0001\n"
    reset = runner.invoke(app, ["db", "reset", "--yes"])
    monkeypatch.undo()
    get_settings.cache_clear()
    assert reset.output == "reset minutes_migrations_test\n"


def test_db_reset_refuses_without_yes() -> None:
    result = CliRunner().invoke(app, ["db", "reset"])
    assert result.exit_code == 2
    assert result.output == "refusing without --yes\n"
