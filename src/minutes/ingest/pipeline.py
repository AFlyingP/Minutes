import hashlib
import json
import logging
import os
import socket
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import psycopg
from psycopg.rows import TupleRow
from psycopg.types.json import Jsonb

from minutes import db, queue
from minutes.config import CONFIG_DIR, Corpus, get_settings, load_cities
from minutes.errors import (
    LLMQuotaError,
    NotFoundError,
    PermanentStageError,
    SourceError,
    StageError,
)
from minutes.extract import extractor
from minutes.ingest import chunk, download, embed, ocr, parse, segment
from minutes.labels import store
from minutes.log import bind_correlation_id
from minutes.sources import get_source

Connection = psycopg.Connection[TupleRow]

STAGES: tuple[str, ...] = ("download", "parse", "ocr", "segment", "chunk", "embed", "extract")
EXTRACTION_WORKERS = 8
STAGE_VERSIONS: dict[str, int] = {stage: 1 for stage in STAGES} | {
    "parse": 5,
    "segment": 2,
    "chunk": 2,
    "extract": 2,
}
LABEL_GATED_STAGES = ("segment", "chunk", "embed", "extract")
_LOG = logging.getLogger("minutes.ingest")


@dataclass(frozen=True)
class StageResult:
    status: Literal["done", "skipped"]
    detail: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractPoolSummary:
    done: int
    skipped: int
    failed: int
    dead: int


@dataclass(frozen=True)
class _ExtractDocumentResult:
    document_id: str
    status: Literal["done", "skipped", "failed", "dead"]
    error: str | None = None


@dataclass(frozen=True)
class _ExtractWorkerResult:
    documents: list[_ExtractDocumentResult]
    quota_error: LLMQuotaError | None = None
    other_error: Exception | None = None


def stage_input_hash(content_sha256: str, stage: str) -> str:
    """Hash of the document content and the versions of this stage and every stage before it."""
    upstream = STAGES[: STAGES.index(stage) + 1]
    versions = ",".join(f"{s}:{STAGE_VERSIONS[s]}" for s in upstream)
    return hashlib.sha256(f"{content_sha256}|{versions}".encode()).hexdigest()


def should_run(conn: Connection, document_id: str, stage: str, input_hash: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM stage_runs WHERE document_id = %s AND stage = %s "
        "AND input_sha256 = %s AND status = 'done'",
        (document_id, stage, input_hash),
    ).fetchone()
    return row is None


STAGE_FUNCTIONS: dict[str, Callable[[Connection, str], dict[str, object]]] = {
    "download": download.run,
    "parse": parse.run,
    "ocr": ocr.run,
    "segment": segment.run,
    "chunk": chunk.run,
    "embed": embed.run,
    "extract": extractor.run,
}


def run_stage(
    conn: Connection,
    stage: str,
    document_id: str,
    corpus: str,
    *,
    attempt: int = 1,
    controls: bool = True,
) -> StageResult:
    """Run one stage for one document unless it already ran on the same input."""
    if stage in LABEL_GATED_STAGES and corpus != "fixture":
        store.require_frozen()
    row = conn.execute(
        "SELECT source_url, content_sha256 FROM documents WHERE id = %s", (document_id,)
    ).fetchone()
    if row is None:
        raise NotFoundError(f"unknown document {document_id}")
    source_url, content_sha256 = row
    if stage == "download":
        content_sha256 = hashlib.sha256(source_url.encode()).hexdigest()
    if content_sha256 is None:
        # nothing was downloaded for this document, so there is nothing to work on
        return StageResult("skipped")
    input_hash = stage_input_hash(content_sha256, stage)
    if stage == "extract" and not controls:
        input_hash = hashlib.sha256(f"{input_hash}|controls-off".encode()).hexdigest()
    if not should_run(conn, document_id, stage, input_hash):
        return StageResult("skipped")
    fault = get_settings().fault
    if fault.startswith("stage:"):
        _, fault_stage, fault_document, count = fault.split(":")
        if stage == fault_stage and document_id == fault_document and attempt <= int(count):
            raise StageError("injected fault")
    detail = (
        extractor.run(conn, document_id, controls=controls)
        if stage == "extract"
        else STAGE_FUNCTIONS[stage](conn, document_id)
    )
    conn.execute(
        "INSERT INTO stage_runs (document_id, stage, input_sha256, status, detail) "
        "VALUES (%s, %s, %s, 'done', %s) "
        "ON CONFLICT (document_id, stage) DO UPDATE SET input_sha256 = EXCLUDED.input_sha256, "
        "status = 'done', detail = EXCLUDED.detail, finished_at = now()",
        (document_id, stage, input_hash, Jsonb(detail)),
    )
    return StageResult("done", detail)


def _run_extract_worker(url: str, controls: bool, stop: threading.Event) -> _ExtractWorkerResult:
    worker_id = f"{socket.gethostname()}-{os.getpid()}-{threading.get_ident()}"
    documents: list[_ExtractDocumentResult] = []
    quota_error: LLMQuotaError | None = None
    other_error: Exception | None = None
    with db.connect(url) as conn:
        conn.execute("LISTEN minutes_jobs")
        conn.commit()
        while not stop.is_set():
            with conn.transaction():
                queue.requeue_stale(conn)
                job = queue.claim(conn, worker_id, "extract")
            if job is None:
                with conn.transaction():
                    pending = conn.execute(
                        "SELECT EXISTS (SELECT 1 FROM jobs WHERE status IN ('queued', 'running') "
                        "AND stage = 'extract')"
                    ).fetchone()
                    assert pending is not None
                if not pending[0] or stop.is_set():
                    break
                # Finishing a job sends no notification, so poll again soon.
                for _ in conn.notifies(timeout=0.5, stop_after=1):
                    pass
                continue

            if stop.is_set():
                with conn.transaction():
                    queue.release(conn, job.id)
                break

            bind_correlation_id(hashlib.md5(f"job-{job.id}".encode()).hexdigest())
            fields = {
                "job_id": job.id,
                "stage": job.stage,
                "document_id": job.document_id,
                "attempts": job.attempts,
            }
            try:
                with conn.transaction():
                    result = run_stage(
                        conn,
                        "extract",
                        job.document_id,
                        job.corpus,
                        attempt=job.attempts,
                        controls=controls,
                    )
                    queue.complete(conn, job.id)
            except LLMQuotaError as err:
                with conn.transaction():
                    queue.release(conn, job.id)
                _LOG.warning(
                    "job released",
                    extra={"event": "job_released", "fields": fields},
                )
                quota_error = err
                stop.set()
                break
            except PermanentStageError as err:
                with conn.transaction():
                    conn.execute(
                        "UPDATE documents SET status = 'failed', fail_reason = %s, "
                        "updated_at = now() WHERE id = %s",
                        (str(err), job.document_id),
                    )
                    queue.fail(conn, job.id, str(err), permanent=True)
                _LOG.error(
                    "job failed",
                    extra={"event": "job_failed", "fields": {**fields, "error": str(err)[:200]}},
                )
                documents.append(_ExtractDocumentResult(job.document_id, "failed", str(err)))
            except (StageError, SourceError) as err:
                with conn.transaction():
                    status = queue.fail(conn, job.id, str(err), permanent=False)
                if status == "dead":
                    _LOG.error(
                        "job dead",
                        extra={"event": "job_dead", "fields": {**fields, "error": str(err)[:200]}},
                    )
                    documents.append(_ExtractDocumentResult(job.document_id, "dead", str(err)))
                elif status == "queued":
                    _LOG.warning(
                        "job retry",
                        extra={"event": "job_retry", "fields": {**fields, "error": str(err)[:200]}},
                    )
            except Exception as err:
                with conn.transaction():
                    queue.release(conn, job.id)
                other_error = err
                stop.set()
                break
            else:
                event = "job skipped" if result.status == "skipped" else "job done"
                _LOG.info(
                    event,
                    extra={"event": event.replace(" ", "_"), "fields": fields},
                )
                documents.append(_ExtractDocumentResult(job.document_id, result.status))
    return _ExtractWorkerResult(documents, quota_error, other_error)


def run_extract_pool(
    url: str, corpus: str, document_ids: list[str], *, controls: bool = True
) -> ExtractPoolSummary:
    """Run queued extraction jobs in independent connections and aggregate by document ID."""
    with db.connect(url) as conn:
        for document_id in sorted(set(document_ids)):
            queue.enqueue(conn, "extract", document_id, corpus)

    stop = threading.Event()
    worker_results: list[_ExtractWorkerResult] = []
    quota_error: LLMQuotaError | None = None
    other_error: Exception | None = None
    with ThreadPoolExecutor(max_workers=EXTRACTION_WORKERS) as pool:
        futures: list[Future[_ExtractWorkerResult]] = [
            pool.submit(_run_extract_worker, url, controls, stop) for _ in range(EXTRACTION_WORKERS)
        ]
        for future in as_completed(futures):
            try:
                worker_results.append(future.result())
            except Exception as err:
                stop.set()
                other_error = other_error or err
    results = [document for result in worker_results for document in result.documents]
    results.sort(key=lambda result: result.document_id)
    for worker_result in worker_results:
        quota_error = quota_error or worker_result.quota_error
        other_error = other_error or worker_result.other_error
    for document_result in results:
        if document_result.error is not None:
            _LOG.error(
                "extract stage failed",
                extra={
                    "event": "stage_failed",
                    "fields": {
                        "stage": "extract",
                        "document_id": document_result.document_id,
                        "error": document_result.error[:200],
                    },
                },
            )
    if quota_error is not None:
        raise quota_error
    if other_error is not None:
        raise other_error
    counts = {status: 0 for status in ("done", "skipped", "failed", "dead")}
    for document_result in results:
        counts[document_result.status] += 1
    return ExtractPoolSummary(**counts)


def documents_for(conn: Connection, corpus: str, city: str | None) -> list[str]:
    """IDs of the documents a stage run covers, in ascending order."""
    if corpus == "dev":
        subset = json.loads((CONFIG_DIR / "dev_subset.json").read_text(encoding="utf-8"))
        rows = conn.execute(
            "SELECT id FROM documents WHERE id = ANY(%s) ORDER BY id", (subset["documents"],)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id FROM documents WHERE status IN ('discovered', 'downloaded') "
            "AND (%s::text IS NULL OR city_id = %s) ORDER BY id",
            (city, city),
        ).fetchall()
    return [row[0] for row in rows]


def discover(conn: Connection, corpus: Corpus, city: str | None) -> int:
    """List meetings from each city's source and add the documents not seen before."""
    added = 0
    for city_id, config in load_cities(corpus).items():
        if city is not None and city != city_id:
            continue
        conn.execute(
            "INSERT INTO cities (id, name, state, item_format) VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (id) DO NOTHING",
            (city_id, config.name, config.state, config.item_format),
        )
        source = get_source(city_id, config)
        date_from = date.fromisoformat(config.date_from)
        date_to = date.fromisoformat(config.date_to)
        listed = source.list_meetings(date_from, date_to)
        for meeting, documents in listed:
            conn.execute(
                "INSERT INTO meetings (id, city_id, body, meeting_date, title, source_key, "
                "recording_url, recording_seek) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (id) DO NOTHING",
                (
                    meeting.id,
                    meeting.city_id,
                    meeting.body,
                    meeting.meeting_date,
                    meeting.title,
                    meeting.source_key,
                    meeting.recording_url,
                    meeting.recording_seek,
                ),
            )
            for doc in documents:
                unit_kind = "page" if doc.media_type == "application/pdf" else "segment"
                inserted = conn.execute(
                    "INSERT INTO documents (id, meeting_id, city_id, kind, source_url, "
                    "media_type, unit_kind) VALUES (%s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (id) DO NOTHING",
                    (
                        doc.id,
                        doc.meeting_id,
                        doc.city_id,
                        doc.kind,
                        doc.source_url,
                        doc.media_type,
                        unit_kind,
                    ),
                )
                added += inserted.rowcount
    return added


def load_fixture_corpus(url: str) -> int:
    """Reset the database at url and run the test corpus through every stage."""
    db.reset(url)
    with db.connect(url) as conn:
        discover(conn, "fixture", None)
        documents = documents_for(conn, "fixture", None)
    for document_id in documents:
        with db.connect(url) as conn:
            for stage in STAGES:
                run_stage(conn, stage, document_id, "fixture")
    return len(documents)
