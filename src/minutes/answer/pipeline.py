import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import psycopg
from psycopg.rows import TupleRow

from minutes.errors import LLMOutputError, ValidationError
from minutes.llm import get_client
from minutes.llm.prompts import SYNTH_SCHEMA, SYNTH_SYSTEM, SourceSpan, synth_user
from minutes.retrieval.search import Filters, Hit, Span, city_names, search, span_label

K = 8


@dataclass(frozen=True)
class AnswerSentence:
    text: str
    citations: list[int]


@dataclass(frozen=True)
class Citation:
    number: int
    document_id: str
    unit_index: int
    unit_kind: str
    start_offset: int
    end_offset: int
    quote: str
    city_id: str
    body: str
    meeting_date: date
    doc_kind: str
    start_ms: int | None


@dataclass(frozen=True)
class AnswerResult:
    declined: bool
    decline_reason: str | None
    sentences: list[AnswerSentence]
    citations: list[Citation]
    pipeline: str
    model: str | None
    cost_usd: float
    latency_ms: int
    cached: bool
    dropped_sentences: int


def _declined(reason: str, pipeline: str, started: float, dropped: int = 0) -> AnswerResult:
    elapsed = int((time.perf_counter() - started) * 1000)
    return AnswerResult(True, reason, [], [], pipeline, None, 0.0, elapsed, False, dropped)


def _citation(number: int, hit: Hit, span: Span) -> Citation:
    return Citation(
        number=number,
        document_id=span.document_id,
        unit_index=span.unit_index,
        unit_kind=span.unit_kind,
        start_offset=span.start_offset,
        end_offset=span.end_offset,
        quote=span.text,
        city_id=hit.city_id,
        body=hit.body,
        meeting_date=hit.meeting_date,
        doc_kind=hit.doc_kind,
        start_ms=hit.start_ms,
    )


def answer(
    conn: psycopg.Connection[TupleRow],
    question: str,
    filters: Filters,
    *,
    pipeline: Literal["baseline", "final", "langgraph"] = "final",
    controls: bool = True,
    sample_index: int = 0,
) -> AnswerResult:
    """Answer from retrieved text only; every sentence kept cites the spans it rests on."""
    if not question.strip() or len(question) > 500:
        raise ValidationError("question: must be 1 to 500 characters")
    if pipeline != "baseline":
        raise ValidationError(f"pipeline {pipeline} is not available")
    started = time.perf_counter()
    hits = search(conn, question, filters, mode="keyword", chunker="fixed", k=K)
    if not hits:
        return _declined("no_retrieval", pipeline, started)
    names = city_names(conn)
    sources = [(hit, span) for hit in hits for span in hit.spans]
    spans = [
        SourceSpan(
            number,
            names[hit.city_id],
            hit.body,
            hit.meeting_date.isoformat(),
            hit.doc_kind,
            span_label(span.unit_kind, span.unit_index, hit.start_ms),
            span.text,
        )
        for number, (hit, span) in enumerate(sources, start=1)
    ]
    try:
        result = get_client().chat(
            purpose="synth",
            messages=[
                {"role": "system", "content": SYNTH_SYSTEM},
                {"role": "user", "content": synth_user(question, spans)},
            ],
            schema=SYNTH_SCHEMA,
            schema_name="synth",
            sample_index=sample_index,
        )
    except LLMOutputError:
        return _declined("model_error", pipeline, started)
    output: dict[str, Any] = result.parsed or {}
    if output["decline"]:
        return _declined("model_declined", pipeline, started)
    kept = [
        s
        for s in output["sentences"]
        if s["citations"] and all(1 <= n <= len(sources) for n in s["citations"])
    ]
    dropped = len(output["sentences"]) - len(kept)
    if not kept:
        return _declined("unsupported", pipeline, started, dropped)
    # sources are renumbered in the order the answer first uses them
    numbers: dict[int, int] = {}
    for sentence in kept:
        for cited in sentence["citations"]:
            numbers.setdefault(cited, len(numbers) + 1)
    return AnswerResult(
        declined=False,
        decline_reason=None,
        sentences=[
            AnswerSentence(s["text"], [numbers[cited] for cited in s["citations"]]) for s in kept
        ],
        citations=[_citation(new, *sources[old - 1]) for old, new in numbers.items()],
        pipeline=pipeline,
        model=result.model,
        cost_usd=result.cost_usd,
        latency_ms=int((time.perf_counter() - started) * 1000),
        cached=False,
        dropped_sentences=dropped,
    )
