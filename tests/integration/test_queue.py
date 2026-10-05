from collections.abc import Iterator
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import psycopg
import pytest
from psycopg.rows import TupleRow
from typer.testing import CliRunner

from minutes import db, queue
from minutes.cli import app
from minutes.config import get_settings
from minutes.ingest import pipeline

pytestmark = pytest.mark.integration

Connection = psycopg.Connection[TupleRow]


def test_enqueue_is_idempotent_while_active(conn: Connection) -> None:
    document_id = f"queue-{uuid4().hex}"

    first = queue.enqueue(conn, "parse", document_id, "fixture")
    second = queue.enqueue(conn, "parse", document_id, "fixture")

    assert first is not None
    assert second is None


def test_two_claims_get_different_jobs(test_db_url: str) -> None:
    documents = [f"queue-{uuid4().hex}", f"queue-{uuid4().hex}"]
    try:
        with psycopg.connect(test_db_url) as setup:
            for document_id in documents:
                queue.enqueue(setup, "claim-test", document_id, "fixture")

        with (
            psycopg.connect(test_db_url) as first_conn,
            psycopg.connect(test_db_url) as second_conn,
        ):
            first = queue.claim(first_conn, "first-worker")
            second = queue.claim(second_conn, "second-worker")
            assert first is not None
            assert second is not None
            assert first.id != second.id
            assert queue.claim(first_conn, "first-worker") is None
            first_conn.commit()
            second_conn.commit()
    finally:
        with psycopg.connect(test_db_url) as cleanup:
            cleanup.execute("DELETE FROM jobs WHERE document_id = ANY(%s)", (documents,))


def test_retryable_failure_requeues_with_future_run_after(conn: Connection) -> None:
    document_id = f"queue-{uuid4().hex}"
    job_id = queue.enqueue(conn, "retry-test", document_id, "fixture")
    assert job_id is not None
    job = queue.claim(conn, "worker", stage="retry-test")
    assert job is not None

    assert queue.fail(conn, job.id, "temporary failure", permanent=False) == "queued"
    row = conn.execute(
        "SELECT status, attempts, run_after > now(), "
        "extract(epoch FROM run_after - now()) FROM jobs WHERE id = %s",
        (job.id,),
    ).fetchone()
    assert row is not None
    status, attempts, run_after, delay = row

    assert status == "queued"
    assert attempts == 1
    assert run_after
    assert float(delay) == pytest.approx(queue.backoff_seconds(job.id, 1), abs=0.01)


def test_fifth_failure_marks_dead(conn: Connection) -> None:
    document_id = f"queue-{uuid4().hex}"
    job_id = queue.enqueue(conn, "dead-test", document_id, "fixture")
    assert job_id is not None
    job = queue.claim(conn, "worker", stage="dead-test")
    assert job is not None
    conn.execute("UPDATE jobs SET attempts = 5 WHERE id = %s", (job.id,))

    assert queue.fail(conn, job.id, "failed five times", permanent=False) == "dead"
    row = conn.execute("SELECT status, finished_at FROM jobs WHERE id = %s", (job.id,)).fetchone()
    assert row is not None
    status, finished_at = row

    assert status == "dead"
    assert finished_at is not None


def test_permanent_failure_marks_failed_without_retry(conn: Connection) -> None:
    document_id = f"queue-{uuid4().hex}"
    job_id = queue.enqueue(conn, "permanent-test", document_id, "fixture")
    assert job_id is not None
    job = queue.claim(conn, "worker", stage="permanent-test")
    assert job is not None

    assert queue.fail(conn, job.id, "invalid document", permanent=True) == "failed"
    row = conn.execute(
        "SELECT status, attempts, finished_at, last_error FROM jobs WHERE id = %s", (job.id,)
    ).fetchone()
    assert row is not None
    status, attempts, finished_at, last_error = row

    assert status == "failed"
    assert attempts == 1
    assert finished_at is not None
    assert last_error == "invalid document"


def test_stale_running_job_is_requeued(conn: Connection) -> None:
    document_id = f"queue-{uuid4().hex}"
    job_id = queue.enqueue(conn, "stale-test", document_id, "fixture")
    assert job_id is not None
    job = queue.claim(conn, "stale-worker", stage="stale-test")
    assert job is not None
    conn.execute(
        "UPDATE jobs SET locked_at = now() - interval '901 seconds' WHERE id = %s", (job.id,)
    )

    assert queue.requeue_stale(conn) == 1
    row = conn.execute(
        "SELECT status, locked_by, locked_at FROM jobs WHERE id = %s", (job.id,)
    ).fetchone()
    assert row is not None
    status, locked_by, locked_at = row

    assert status == "queued"
    assert locked_by is None
    assert locked_at is None


def test_retry_dead_requeues(conn: Connection) -> None:
    document_id = f"queue-{uuid4().hex}"
    job_id = queue.enqueue(conn, "retry-dead-test", document_id, "fixture")
    assert job_id is not None
    conn.execute(
        "UPDATE jobs SET status = 'dead', attempts = 5, finished_at = now(), "
        "last_error = 'old failure' WHERE id = %s",
        (job_id,),
    )
    row = conn.execute("SELECT count(*) FROM jobs WHERE status = 'dead'").fetchone()
    assert row is not None
    before = row[0]

    assert queue.retry_dead(conn) == before
    row = conn.execute(
        "SELECT status, attempts, finished_at, last_error FROM jobs WHERE id = %s", (job_id,)
    ).fetchone()
    assert row is not None
    status, attempts, finished_at, last_error = row

    assert status == "queued"
    assert attempts == 0
    assert finished_at is None
    assert last_error == "old failure"


@pytest.mark.parametrize("active_status", ["queued", "running"])
def test_retry_dead_skips_pair_with_active_job(conn: Connection, active_status: str) -> None:
    document_id = f"queue-{uuid4().hex}"
    conn.execute("DELETE FROM jobs")
    dead_id = queue.enqueue(conn, "parse", document_id, "fixture")
    assert dead_id is not None
    conn.execute(
        "UPDATE jobs SET status = 'dead', attempts = 5, finished_at = now() WHERE id = %s",
        (dead_id,),
    )
    active_id = queue.enqueue(conn, "parse", document_id, "fixture")
    assert active_id is not None
    conn.execute("UPDATE jobs SET status = %s WHERE id = %s", (active_status, active_id))

    assert queue.retry_dead(conn) == 0
    assert conn.execute(
        "SELECT id, status, attempts, finished_at IS NOT NULL FROM jobs ORDER BY id"
    ).fetchall() == [(dead_id, "dead", 5, True), (active_id, active_status, 0, False)]


def test_retry_dead_requeues_one_job_per_pair(conn: Connection) -> None:
    documents = [f"queue-{uuid4().hex}", f"queue-{uuid4().hex}"]
    conn.execute("DELETE FROM jobs")
    pairs = [("parse", documents[0]), ("download", documents[0]), ("parse", documents[1])]
    expected = []
    for stage, document_id in pairs:
        for index in range(2):
            job_id = queue.enqueue(conn, stage, document_id, "fixture")
            assert job_id is not None
            conn.execute(
                "UPDATE jobs SET status = 'dead', attempts = 5, finished_at = now(), "
                "run_after = now() + interval '1 day', locked_by = 'old-worker', "
                "locked_at = now(), last_error = 'old failure' WHERE id = %s",
                (job_id,),
            )
            expected.append(
                (
                    job_id,
                    "dead" if index == 0 else "queued",
                    5 if index == 0 else 0,
                    index == 0,
                    index == 0,
                    index == 0,
                    index == 0,
                    "old failure",
                )
            )

    assert queue.retry_dead(conn) == len(pairs)
    assert (
        conn.execute(
            "SELECT id, status, attempts, finished_at IS NOT NULL, run_after > now(), "
            "locked_by IS NOT NULL, locked_at IS NOT NULL, last_error FROM jobs ORDER BY id"
        ).fetchall()
        == expected
    )
    assert queue.retry_dead(conn) == 0


@pytest.fixture
def fresh_fixture_db(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    settings = get_settings()
    url = urlsplit(settings.test_database_url)._replace(path="/minutes_migrations_test").geturl()
    db.reset(url)
    monkeypatch.setenv("MINUTES_TEST_DATABASE_URL", url)
    monkeypatch.setenv("MINUTES_CORPUS", "fixture")
    monkeypatch.setenv("MINUTES_LLM_MODE", "stub")
    monkeypatch.setenv("MINUTES_MODELS_MODE", "stub")
    monkeypatch.setenv("MINUTES_FAULT", "")
    get_settings.cache_clear()
    yield url
    get_settings.cache_clear()


def _run_ingest(url: str, stages: str, fault: str = "") -> Any:
    get_settings.cache_clear()
    result = CliRunner().invoke(
        app,
        ["ingest", "run", "--corpus", "fixture", "--stages", stages],
        env={
            "MINUTES_TEST_DATABASE_URL": url,
            "MINUTES_CORPUS": "fixture",
            "MINUTES_LLM_MODE": "stub",
            "MINUTES_MODELS_MODE": "stub",
            "MINUTES_FAULT": fault,
        },
    )
    get_settings.cache_clear()
    return result


def test_injected_parse_fault_recovers_on_third_attempt(fresh_fixture_db: str) -> None:
    with db.connect(fresh_fixture_db) as conn:
        pipeline.discover(conn, "fixture", None)

    result = _run_ingest(
        fresh_fixture_db,
        "download,parse",
        "stage:parse:birch-agenda-201:2",
    )

    assert result.exit_code == 0, result.output
    assert "done 24 skipped 0 failed 0 dead 0" in result.output
    assert '"event": "job_retry"' in result.output
    with db.connect(fresh_fixture_db) as conn:
        row = conn.execute(
            "SELECT status, attempts FROM jobs "
            "WHERE stage = 'parse' AND document_id = 'birch-agenda-201'"
        ).fetchone()
        assert row is not None
        status, attempts = row
    assert status == "done"
    assert attempts == 3


def test_quota_fault_releases_job_and_exits_75(fresh_fixture_db: str) -> None:
    with db.connect(fresh_fixture_db) as conn:
        pipeline.discover(conn, "fixture", None)
        documents = pipeline.documents_for(conn, "fixture", None)
    for document_id in documents:
        for stage in pipeline.STAGES[:-1]:
            with db.connect(fresh_fixture_db) as conn:
                pipeline.run_stage(conn, stage, document_id, "fixture")

    result = _run_ingest(fresh_fixture_db, "extract", "llm_quota:1")

    assert result.exit_code == 75
    assert "paused: quota exhausted, rerun the same command to resume" in result.output
    with db.connect(fresh_fixture_db) as conn:
        row = conn.execute(
            "SELECT status, attempts FROM jobs "
            "WHERE stage = 'extract' AND document_id = 'alder-agenda-101'"
        ).fetchone()
        assert row is not None
        status, attempts = row
    assert status == "queued"
    assert attempts == 0

    resumed = _run_ingest(fresh_fixture_db, "extract")
    assert resumed.exit_code == 0, resumed.output
