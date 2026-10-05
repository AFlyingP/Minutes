from collections.abc import Iterator
from dataclasses import asdict
from datetime import date
from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends, Query
from psycopg.rows import TupleRow

from minutes import db
from minutes.answer.pipeline import answer
from minutes.api.schemas import (
    DECLINE_MESSAGE,
    AskIn,
    AskOut,
    Chunker,
    CitationOut,
    CitiesOut,
    CityOut,
    DocKind,
    HealthOut,
    HitOut,
    SearchMode,
    SearchOut,
    SentenceOut,
    SpanOut,
)
from minutes.config import corpus_database_url, get_settings
from minutes.errors import ValidationError
from minutes.retrieval.search import Filters, city_names, hit_heading, search, span_label

router = APIRouter(prefix="/api")


def get_conn() -> Iterator[psycopg.Connection[TupleRow]]:
    with db.connect(corpus_database_url(get_settings().corpus)) as conn:
        yield conn


Conn = Annotated[psycopg.Connection[TupleRow], Depends(get_conn)]


def checked_filters(
    conn: psycopg.Connection[TupleRow],
    city: str | None,
    body: str | None,
    date_from: date | None,
    date_to: date | None,
    kind: str | None,
) -> Filters:
    if city is not None and city not in city_names(conn):
        raise ValidationError(f"city: unknown city {city}")
    return Filters(city=city, date_from=date_from, date_to=date_to, body=body, kind=kind)


@router.get("/health")
def health(conn: Conn) -> HealthOut:
    conn.execute("SELECT 1")
    return HealthOut(status="ok", corpus=get_settings().corpus, db="ok")


@router.get("/cities")
def cities(conn: Conn) -> CitiesOut:
    rows = conn.execute(
        "SELECT c.id, c.name, c.state, "
        "coalesce((SELECT array_agg(DISTINCT m.body ORDER BY m.body) FROM meetings m "
        "          WHERE m.city_id = c.id), '{}'), "
        "(SELECT min(m.meeting_date) FROM meetings m WHERE m.city_id = c.id), "
        "(SELECT max(m.meeting_date) FROM meetings m WHERE m.city_id = c.id), "
        "(SELECT count(*) FROM documents d WHERE d.city_id = c.id AND d.status = 'downloaded') "
        "FROM cities c ORDER BY c.id"
    ).fetchall()
    return CitiesOut(
        cities=[
            CityOut(
                id=row[0],
                name=row[1],
                state=row[2],
                bodies=row[3],
                date_min=row[4],
                date_max=row[5],
                documents=row[6],
            )
            for row in rows
        ]
    )


@router.get("/search")
def search_endpoint(
    conn: Conn,
    q: str,
    city: str | None = None,
    body: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    kind: DocKind | None = None,
    mode: SearchMode = "keyword",
    k: Annotated[int, Query(ge=1, le=50)] = 10,
    chunker: Chunker = "fixed",
) -> SearchOut:
    filters = checked_filters(conn, city, body, date_from, date_to, kind)
    hits = search(conn, q.strip(), filters, mode=mode, chunker=chunker, k=k)
    names = city_names(conn)
    meetings: dict[str, str] = dict(
        conn.execute(
            "SELECT id, meeting_id FROM documents WHERE id = ANY(%s)",
            ([hit.document_id for hit in hits],),
        ).fetchall()
    )
    return SearchOut(
        query=q.strip(),
        mode=mode,
        chunker=chunker,
        hits=[
            HitOut(
                rank=hit.rank,
                score=hit.score,
                document_id=hit.document_id,
                meeting_id=meetings[hit.document_id],
                city_id=hit.city_id,
                city_name=names[hit.city_id],
                body=hit.body,
                meeting_date=hit.meeting_date,
                doc_kind=hit.doc_kind,
                item_identifier=hit.item_identifier,
                item_title=hit.item_title,
                heading=hit_heading(names[hit.city_id], hit),
                snippet=hit.spans[0].text.replace("\n", " ")[:240],
                spans=[
                    SpanOut(
                        document_id=span.document_id,
                        unit_index=span.unit_index,
                        unit_kind=span.unit_kind,
                        start_offset=span.start_offset,
                        end_offset=span.end_offset,
                        start_ms=hit.start_ms,
                        label=span_label(span.unit_kind, span.unit_index, hit.start_ms),
                    )
                    for span in hit.spans
                ],
            )
            for hit in hits
        ],
    )


@router.post("/ask")
def ask(conn: Conn, request: AskIn) -> AskOut:
    filters = checked_filters(
        conn, request.city, request.body, request.date_from, request.date_to, request.kind
    )
    result = answer(conn, request.question, filters, pipeline=request.pipeline)
    return AskOut(
        declined=result.declined,
        decline_reason=result.decline_reason,
        decline_message=DECLINE_MESSAGE if result.declined else None,
        sentences=[SentenceOut(text=s.text, citations=s.citations) for s in result.sentences],
        citations=[
            CitationOut(**asdict(c), label=span_label(c.unit_kind, c.unit_index, c.start_ms))
            for c in result.citations
        ],
        pipeline=result.pipeline,
        model=result.model,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
        cached=result.cached,
        dropped_sentences=result.dropped_sentences,
    )
