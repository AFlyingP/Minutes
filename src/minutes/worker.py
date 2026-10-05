import hashlib
import os
import socket
from dataclasses import dataclass

from minutes import db, queue
from minutes.config import Corpus, corpus_database_url
from minutes.errors import LLMQuotaError, PermanentStageError, SourceError, StageError
from minutes.ingest import pipeline
from minutes.log import bind_correlation_id, get_logger

log = get_logger("worker")


@dataclass(frozen=True)
class WorkerSummary:
    done: int
    skipped: int
    failed: int
    dead: int


def run(
    corpus: Corpus,
    drain: bool,
    max_jobs: int | None = None,
    *,
    stage: str | None = None,
) -> WorkerSummary:
    """Run queued stage jobs until drained, limited, or stopped by a quota pause."""
    worker_id = f"{socket.gethostname()}-{os.getpid()}"
    done = skipped = failed = dead = claimed = 0
    url = corpus_database_url(corpus)
    with db.connect(url) as conn:
        conn.execute("LISTEN minutes_jobs")
        conn.commit()
        while max_jobs is None or claimed < max_jobs:
            with conn.transaction():
                queue.requeue_stale(conn)
                job = queue.claim(conn, worker_id, stage)
            if job is None:
                with conn.transaction():
                    where = " AND stage = %s" if stage is not None else ""
                    params = (stage,) if stage is not None else ()
                    pending = conn.execute(
                        "SELECT EXISTS (SELECT 1 FROM jobs WHERE status IN ('queued', 'running')"
                        + where
                        + ")",
                        params,
                    ).fetchone()
                    assert pending is not None
                    has_pending = pending[0]
                if drain and not has_pending:
                    break
                for _ in conn.notifies(timeout=5, stop_after=1):
                    pass
                continue

            claimed += 1
            correlation_id = hashlib.md5(f"job-{job.id}".encode()).hexdigest()
            bind_correlation_id(correlation_id)
            fields = {
                "job_id": job.id,
                "stage": job.stage,
                "document_id": job.document_id,
                "attempts": job.attempts,
            }
            try:
                with conn.transaction():
                    result = pipeline.run_stage(
                        conn, job.stage, job.document_id, job.corpus, attempt=job.attempts
                    )
                    queue.complete(conn, job.id)
            except LLMQuotaError:
                with conn.transaction():
                    queue.release(conn, job.id)
                log.warning(
                    "job released",
                    extra={"event": "job_released", "fields": fields},
                )
                raise
            except PermanentStageError as err:
                with conn.transaction():
                    queue.fail(conn, job.id, str(err), permanent=True)
                log.error(
                    "job failed",
                    extra={"event": "job_failed", "fields": {**fields, "error": str(err)[:200]}},
                )
                failed += 1
            except (StageError, SourceError) as err:
                with conn.transaction():
                    status = queue.fail(conn, job.id, str(err), permanent=False)
                if status == "dead":
                    log.error(
                        "job dead",
                        extra={"event": "job_dead", "fields": {**fields, "error": str(err)[:200]}},
                    )
                    dead += 1
                elif status == "queued":
                    log.warning(
                        "job retry",
                        extra={"event": "job_retry", "fields": {**fields, "error": str(err)[:200]}},
                    )
            else:
                if result.status == "skipped":
                    log.info(
                        "job skipped",
                        extra={"event": "job_skipped", "fields": fields},
                    )
                    skipped += 1
                else:
                    log.info(
                        "job done",
                        extra={"event": "job_done", "fields": fields},
                    )
                    done += 1

    return WorkerSummary(done, skipped, failed, dead)
