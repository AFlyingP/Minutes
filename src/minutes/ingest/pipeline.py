import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import psycopg
from psycopg.rows import TupleRow
from psycopg.types.json import Jsonb

from minutes import db
from minutes.config import CONFIG_DIR, Corpus, get_settings, load_cities
from minutes.errors import NotFoundError, StageError
from minutes.extract import extractor
from minutes.ingest import chunk, download, embed, ocr, parse, segment
from minutes.sources import get_source

Connection = psycopg.Connection[TupleRow]

STAGES: tuple[str, ...] = ("download", "parse", "ocr", "segment", "chunk", "embed", "extract")
STAGE_VERSIONS: dict[str, int] = {stage: 1 for stage in STAGES} | {"parse": 5}
LABEL_GATED_STAGES = ("segment", "chunk", "embed", "extract")


@dataclass(frozen=True)
class StageResult:
    status: Literal["done", "skipped"]
    detail: dict[str, object] = field(default_factory=dict)


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
    conn: Connection, stage: str, document_id: str, corpus: str, *, attempt: int = 1
) -> StageResult:
    """Run one stage for one document unless it already ran on the same input."""
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
    if not should_run(conn, document_id, stage, input_hash):
        return StageResult("skipped")
    fault = get_settings().fault
    if fault.startswith("stage:"):
        _, fault_stage, fault_document, count = fault.split(":")
        if stage == fault_stage and document_id == fault_document and attempt <= int(count):
            raise StageError("injected fault")
    detail = STAGE_FUNCTIONS[stage](conn, document_id)
    conn.execute(
        "INSERT INTO stage_runs (document_id, stage, input_sha256, status, detail) "
        "VALUES (%s, %s, %s, 'done', %s) "
        "ON CONFLICT (document_id, stage) DO UPDATE SET input_sha256 = EXCLUDED.input_sha256, "
        "status = 'done', detail = EXCLUDED.detail, finished_at = now()",
        (document_id, stage, input_hash, Jsonb(detail)),
    )
    return StageResult("done", detail)


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
