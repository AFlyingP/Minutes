from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from datetime import date
from typing import TYPE_CHECKING, cast

import psycopg
from psycopg.rows import TupleRow
from psycopg.types.json import Jsonb

from minutes.retrieval.search import Filters

if TYPE_CHECKING:
    from minutes.answer.pipeline import AnswerResult


def corpus_version(conn: psycopg.Connection[TupleRow]) -> str:
    row = conn.execute(
        "SELECT count(*), coalesce(max(id), 0) FROM chunks WHERE chunker = 'item'"
    ).fetchone()
    assert row is not None
    count, maximum = row
    return hashlib.sha256(f"{count}:{maximum}".encode()).hexdigest()


def cache_key(question: str, filters: Filters, pipeline: str, version: str) -> str:
    normalized = " ".join(question.lower().split()).rstrip("?.!")
    filter_values = asdict(filters)
    canonical_filters = {
        key: value.isoformat() if isinstance(value, date) else value
        for key, value in filter_values.items()
    }
    encoded_filters = json.dumps(canonical_filters, sort_keys=True, separators=(",", ":"))
    value = f"{pipeline}\n{normalized}\n{encoded_filters}\n{version}"
    return hashlib.sha256(value.encode()).hexdigest()


def get(conn: psycopg.Connection[TupleRow], key: str) -> AnswerResult | None:
    started = time.perf_counter()
    row = conn.execute("SELECT response FROM answer_cache WHERE key = %s", (key,)).fetchone()
    elapsed = int((time.perf_counter() - started) * 1000)
    if row is None:
        return None
    from minutes.answer.pipeline import AnswerResult, AnswerSentence, Citation

    value = cast(dict[str, object], row[0])
    citations = []
    for citation in cast(list[dict[str, object]], value["citations"]):
        citations.append(
            Citation(
                number=cast(int, citation["number"]),
                document_id=cast(str, citation["document_id"]),
                unit_index=cast(int, citation["unit_index"]),
                unit_kind=cast(str, citation["unit_kind"]),
                start_offset=cast(int, citation["start_offset"]),
                end_offset=cast(int, citation["end_offset"]),
                quote=cast(str, citation["quote"]),
                city_id=cast(str, citation["city_id"]),
                body=cast(str, citation["body"]),
                meeting_date=date.fromisoformat(cast(str, citation["meeting_date"])),
                doc_kind=cast(str, citation["doc_kind"]),
                start_ms=cast(int | None, citation["start_ms"]),
            )
        )
    return AnswerResult(
        declined=bool(value["declined"]),
        decline_reason=cast(str | None, value["decline_reason"]),
        sentences=[
            AnswerSentence(
                text=cast(str, sentence["text"]),
                citations=cast(list[int], sentence["citations"]),
            )
            for sentence in cast(list[dict[str, object]], value["sentences"])
        ],
        citations=citations,
        pipeline=str(value["pipeline"]),
        model=cast(str | None, value["model"]),
        cost_usd=0.0,
        latency_ms=elapsed,
        cached=True,
        dropped_sentences=cast(int, value["dropped_sentences"]),
    )


def put(conn: psycopg.Connection[TupleRow], key: str, result: AnswerResult) -> None:
    response = json.loads(json.dumps(asdict(result), default=lambda value: value.isoformat()))
    response["cached"] = False
    response["cost_usd"] = result.cost_usd
    conn.execute(
        "INSERT INTO answer_cache (key, response) VALUES (%s, %s) "
        "ON CONFLICT (key) DO UPDATE SET response = EXCLUDED.response",
        (key, Jsonb(response)),
    )
