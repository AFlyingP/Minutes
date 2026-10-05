import random
from dataclasses import dataclass

import psycopg
from psycopg.rows import TupleRow

from minutes.config import get_settings

Connection = psycopg.Connection[TupleRow]


@dataclass(frozen=True)
class Job:
    id: int
    stage: str
    document_id: str
    corpus: str
    attempts: int
    max_attempts: int


def enqueue(conn: Connection, stage: str, document_id: str, corpus: str) -> int | None:
    row = conn.execute(
        "INSERT INTO jobs (stage, document_id, corpus) VALUES (%s, %s, %s) "
        "ON CONFLICT (stage, document_id) WHERE status IN ('queued', 'running') "
        "DO NOTHING RETURNING id",
        (stage, document_id, corpus),
    ).fetchone()
    if row is None:
        return None
    conn.execute("NOTIFY minutes_jobs")
    return int(row[0])


def claim(conn: Connection, worker_id: str, stage: str | None = None) -> Job | None:
    query = (
        "UPDATE jobs SET status = 'running', attempts = attempts + 1, locked_by = %s, "
        "locked_at = now() WHERE id = (SELECT id FROM jobs WHERE status = 'queued' "
        "AND run_after <= now()"
    )
    params: tuple[object, ...] = (worker_id,)
    if stage is not None:
        query += " AND stage = %s"
        params += (stage,)
    query += (
        " ORDER BY run_after, id FOR UPDATE SKIP LOCKED LIMIT 1) "
        "RETURNING id, stage, document_id, corpus, attempts, max_attempts"
    )
    row = conn.execute(query, params).fetchone()
    return Job(*row) if row is not None else None


def complete(conn: Connection, job_id: int) -> None:
    conn.execute(
        "UPDATE jobs SET status = 'done', finished_at = now(), locked_by = NULL, "
        "locked_at = NULL, last_error = NULL WHERE id = %s",
        (job_id,),
    )


def fail(conn: Connection, job_id: int, error: str, permanent: bool) -> str:
    message = error[:500]
    if permanent:
        row = conn.execute(
            "UPDATE jobs SET status = 'failed', finished_at = now(), last_error = %s, "
            "locked_by = NULL, locked_at = NULL WHERE id = %s RETURNING status",
            (message, job_id),
        ).fetchone()
        assert row is not None
        return str(row[0])

    row = conn.execute(
        "SELECT attempts, max_attempts FROM jobs WHERE id = %s", (job_id,)
    ).fetchone()
    assert row is not None
    attempts, max_attempts = row
    if attempts >= max_attempts:
        status = "dead"
        conn.execute(
            "UPDATE jobs SET status = 'dead', finished_at = now(), last_error = %s, "
            "locked_by = NULL, locked_at = NULL WHERE id = %s",
            (message, job_id),
        )
    else:
        status = "queued"
        delay = backoff_seconds(job_id, attempts)
        conn.execute(
            "UPDATE jobs SET status = 'queued', run_after = now() + %s * interval '1 second', "
            "last_error = %s, locked_by = NULL, locked_at = NULL WHERE id = %s",
            (delay, message, job_id),
        )
    return status


def release(conn: Connection, job_id: int) -> None:
    conn.execute(
        "UPDATE jobs SET status = 'queued', attempts = attempts - 1, run_after = now(), "
        "locked_by = NULL, locked_at = NULL WHERE id = %s",
        (job_id,),
    )


def requeue_stale(conn: Connection) -> int:
    result = conn.execute(
        "UPDATE jobs SET status = 'queued', locked_by = NULL, locked_at = NULL "
        "WHERE status = 'running' AND locked_at < now() - interval '900 seconds'"
    )
    return result.rowcount


def retry_dead(conn: Connection) -> int:
    result = conn.execute(
        "UPDATE jobs SET status = 'queued', attempts = 0, run_after = now(), "
        "finished_at = NULL, locked_by = NULL, locked_at = NULL WHERE id IN ("
        "SELECT max(dead.id) FROM jobs dead WHERE dead.status = 'dead' AND NOT EXISTS ("
        "SELECT 1 FROM jobs active WHERE active.stage = dead.stage "
        "AND active.document_id = dead.document_id AND active.status IN ('queued', 'running')) "
        "GROUP BY dead.stage, dead.document_id)"
    )
    if result.rowcount:
        conn.execute("NOTIFY minutes_jobs")
    return result.rowcount


def backoff_seconds(job_id: int, attempt: int) -> float:
    if get_settings().fault:
        return 0.0
    base = min(600, 5 * 2 ** (attempt - 1))
    jitter: float = random.Random(job_id * 1000 + attempt).random() * 0.25
    return float(base * (1 + jitter))
