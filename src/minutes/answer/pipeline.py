import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, cast

import psycopg
from psycopg.rows import TupleRow

from minutes import llm
from minutes.answer import cache
from minutes.answer.faithfulness import check_answer
from minutes.config import get_settings, load_models
from minutes.errors import BudgetExceeded, LLMOutputError, ValidationError
from minutes.llm import Budget, ChatResult
from minutes.llm.prompts import (
    JUDGE_SYSTEM,
    SUPPORTED_SCHEMA,
    SYNTH_SCHEMA,
    SYNTH_SYSTEM,
    SourceSpan,
    judge_user,
    synth_user,
)
from minutes.llm.routing import is_easy_question
from minutes.retrieval.search import Filters, Hit, Span, city_names, search, span_label

K = 8
MIN_RERANK_SCORE = 0.20
MAX_DROPPED_SHARE = 0.5


def get_client() -> llm.LLMClient:
    """Get the current LLM client, allowing callers to replace the factory."""
    return llm.get_client()


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


def _latency_ms(started: float, results: list[ChatResult]) -> int:
    elapsed = int((time.perf_counter() - started) * 1000)
    # a reply from the development cache took no time now, so its original latency is added
    return elapsed + sum(result.latency_ms for result in results if result.cached)


def _declined(
    reason: str,
    pipeline: str,
    started: float,
    dropped: int = 0,
    results: list[ChatResult] | None = None,
) -> AnswerResult:
    calls = results or []
    model = calls[0].model if calls else None
    cost = sum(result.cost_usd for result in calls)
    latency = _latency_ms(started, calls)
    return AnswerResult(True, reason, [], [], pipeline, model, cost, latency, False, dropped)


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
    if pipeline not in ("baseline", "final"):
        raise ValidationError(f"pipeline {pipeline} is not available")
    started = time.perf_counter()
    is_final = pipeline == "final"
    response_key = None
    if is_final and controls:
        response_key = cache.cache_key(question, filters, pipeline, cache.corpus_version(conn))
        cached = cache.get(conn, response_key)
        if cached is not None:
            return cached

    def finish(result: AnswerResult) -> AnswerResult:
        if response_key is not None and not (
            result.declined and result.decline_reason in ("budget_exceeded", "model_error")
        ):
            cache.put(conn, response_key, result)
        return result

    hits = search(
        conn,
        question,
        filters,
        mode="hybrid_rerank" if is_final else "vector",
        chunker="item" if is_final else "fixed",
        k=K,
    )
    if not hits or (is_final and hits[0].score < MIN_RERANK_SCORE):
        return finish(_declined("no_retrieval", pipeline, started))
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
    calls: list[ChatResult] = []
    budget = Budget(get_settings().query_budget_usd) if is_final else None
    models = load_models()
    model = None
    if is_final:
        model = (
            models.cheap
            if controls and is_easy_question(question, hits[0].score)
            else models.strong
        )
    try:
        result = get_client().chat(
            purpose="synth",
            messages=[
                {"role": "system", "content": SYNTH_SYSTEM},
                {"role": "user", "content": synth_user(question, spans)},
            ],
            schema=SYNTH_SCHEMA,
            schema_name="synth",
            model=model,
            sample_index=sample_index,
            budget=budget,
        )
        calls.append(result)
    except BudgetExceeded:
        return finish(_declined("budget_exceeded", pipeline, started, results=calls))
    except LLMOutputError:
        return finish(_declined("model_error", pipeline, started, results=calls))
    output: dict[str, Any] = result.parsed or {}
    if output["decline"]:
        return finish(_declined("model_declined", pipeline, started, results=calls))
    valid = [
        s
        for s in output["sentences"]
        if s["citations"] and all(1 <= n <= len(sources) for n in s["citations"])
    ]
    dropped = len(output["sentences"]) - len(valid)
    kept = valid
    if is_final and valid:
        judge_inputs = [
            (sentence["text"], [sources[number - 1][1].text for number in sentence["citations"]])
            for sentence in valid
        ]

        def judge(sentences: list[tuple[str, list[str]]]) -> list[bool]:
            judged = get_client().chat(
                purpose="judge",
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": judge_user(sentences)},
                ],
                schema=SUPPORTED_SCHEMA,
                schema_name="supported",
                sample_index=sample_index,
                budget=budget,
            )
            calls.append(judged)
            verdicts = [False] * len(sentences)
            verdicts_raw = cast(list[dict[str, Any]], (judged.parsed or {}).get("verdicts", []))
            for verdict in verdicts_raw:
                number = verdict["sentence"]
                if 1 <= number <= len(verdicts):
                    verdicts[number - 1] = verdict["supported"]
            return verdicts

        try:
            supported = check_answer(judge_inputs, judge)
        except BudgetExceeded:
            return finish(_declined("budget_exceeded", pipeline, started, dropped, calls))
        except LLMOutputError:
            return finish(_declined("model_error", pipeline, started, dropped, calls))
        kept = [
            sentence
            for sentence, is_supported in zip(valid, supported, strict=True)
            if is_supported
        ]
        dropped += len(valid) - len(kept)
        if not kept or dropped / len(output["sentences"]) > MAX_DROPPED_SHARE:
            return finish(_declined("unsupported", pipeline, started, dropped, calls))
    if not kept:
        return finish(_declined("unsupported", pipeline, started, dropped, calls))
    # sources are renumbered in the order the answer first uses them
    numbers: dict[int, int] = {}
    for sentence in kept:
        for cited in sentence["citations"]:
            numbers.setdefault(cited, len(numbers) + 1)
    return finish(
        AnswerResult(
            declined=False,
            decline_reason=None,
            sentences=[
                AnswerSentence(s["text"], [numbers[cited] for cited in s["citations"]])
                for s in kept
            ],
            citations=[_citation(new, *sources[old - 1]) for old, new in numbers.items()],
            pipeline=pipeline,
            model=result.model,
            cost_usd=sum(call.cost_usd for call in calls),
            latency_ms=_latency_ms(started, calls),
            cached=False,
            dropped_sentences=dropped,
        )
    )
