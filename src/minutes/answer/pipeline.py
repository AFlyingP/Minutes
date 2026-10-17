from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, NotRequired, TypedDict, cast

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


class AnswerState(TypedDict):
    question: str
    filters: Filters
    controls: bool
    sample_index: int
    budget: Budget | None
    hits: list[Hit]
    spans: list[SourceSpan]
    model: str | None
    draft: dict[str, Any] | None
    sentences: list[dict[str, Any]]
    dropped: int
    result: AnswerResult | None
    started: NotRequired[float]
    sources: NotRequired[list[tuple[Hit, Span]]]
    span_metadata: NotRequired[dict[tuple[str, int], tuple[str, int | None]]]
    calls: NotRequired[list[ChatResult]]
    synth_result: NotRequired[ChatResult]
    conn: NotRequired[psycopg.Connection[TupleRow]]


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


def _citation(number: int, hit: Hit, span: Span, doc_kind: str, start_ms: int | None) -> Citation:
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
        doc_kind=doc_kind,
        start_ms=start_ms,
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
    if pipeline == "langgraph":
        from minutes.answer.graph_variant import answer as graph_answer

        return graph_answer(conn, question, filters, controls=controls, sample_index=sample_index)
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

    state: AnswerState = {
        "question": question,
        "filters": filters,
        "controls": controls,
        "sample_index": sample_index,
        "budget": None,
        "hits": [],
        "spans": [],
        "model": None,
        "draft": None,
        "sentences": [],
        "dropped": 0,
        "result": None,
        "started": started,
        "calls": [],
    }
    retrieve_step(conn, state, pipeline=pipeline)
    if state["result"] is None:
        synthesize_step(state, pipeline=pipeline)
    if state["result"] is None:
        verify_step(state, pipeline=pipeline)
    if state["result"] is None:
        finalize_step(state, pipeline=pipeline)
    assert state["result"] is not None
    return finish(state["result"])


def retrieve_step(
    conn: psycopg.Connection[TupleRow],
    state: AnswerState,
    *,
    pipeline: Literal["baseline", "final", "langgraph"] = "final",
) -> AnswerState:
    started = state.get("started", time.perf_counter())
    is_final = pipeline != "baseline"
    hits = search(
        conn,
        state["question"],
        state["filters"],
        mode="hybrid_rerank" if is_final else "vector",
        chunker="item" if is_final else "fixed",
        k=K,
    )
    state["hits"] = hits
    if not hits or (is_final and hits[0].score < MIN_RERANK_SCORE):
        state["result"] = _declined("no_retrieval", pipeline, started)
        return state
    names = city_names(conn)
    sources = [(hit, span) for hit in hits for span in hit.spans]
    state["sources"] = sources
    span_documents = [span.document_id for _, span in sources]
    span_units = [span.unit_index for _, span in sources]
    state["span_metadata"] = {
        (document_id, unit_index): (doc_kind, start_ms)
        for document_id, unit_index, doc_kind, start_ms in conn.execute(
            "SELECT refs.document_id, refs.unit_index, d.kind, u.start_ms "
            "FROM unnest(%s::text[], %s::int[]) AS refs(document_id, unit_index) "
            "JOIN documents d ON d.id = refs.document_id "
            "JOIN units u ON u.document_id = refs.document_id AND u.unit_index = refs.unit_index",
            (span_documents, span_units),
        ).fetchall()
    }
    state["spans"] = [
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
    state["budget"] = Budget(get_settings().query_budget_usd) if is_final else None
    if is_final:
        models = load_models()
        state["model"] = (
            models.cheap
            if state["controls"] and is_easy_question(state["question"], hits[0].score)
            else models.strong
        )
    return state


def synthesize_step(
    state: AnswerState,
    *,
    pipeline: Literal["baseline", "final", "langgraph"] = "final",
    chat_model: Any | None = None,
) -> AnswerState:
    started = state.get("started", time.perf_counter())
    calls = state.setdefault("calls", [])
    try:
        if chat_model is None:
            result = get_client().chat(
                purpose="synth",
                messages=[
                    {"role": "system", "content": SYNTH_SYSTEM},
                    {"role": "user", "content": synth_user(state["question"], state["spans"])},
                ],
                schema=SYNTH_SCHEMA,
                schema_name="synth",
                model=state["model"],
                sample_index=state["sample_index"],
                budget=state["budget"],
            )
        else:
            from langchain_core.messages import HumanMessage, SystemMessage

            response = chat_model.invoke(
                [
                    SystemMessage(content=SYNTH_SYSTEM),
                    HumanMessage(content=synth_user(state["question"], state["spans"])),
                ]
            )
            result = _chat_result_from_message(response)
        calls.append(result)
    except BudgetExceeded:
        state["result"] = _declined("budget_exceeded", pipeline, started, results=calls)
        return state
    except LLMOutputError:
        state["result"] = _declined("model_error", pipeline, started, results=calls)
        return state
    state["synth_result"] = result
    if result.parsed is None:
        state["result"] = _declined("model_error", pipeline, started, results=calls)
        return state
    output: dict[str, Any] = result.parsed
    state["draft"] = output
    if output["decline"]:
        state["result"] = _declined("model_declined", pipeline, started, results=calls)
        return state
    sources = state["sources"]
    state["sentences"] = [
        sentence
        for sentence in output["sentences"]
        if sentence["citations"]
        and all(1 <= number <= len(sources) for number in sentence["citations"])
    ]
    state["dropped"] = len(output["sentences"]) - len(state["sentences"])
    return state


def _chat_result_from_message(message: Any) -> ChatResult:
    metadata = message.response_metadata
    return ChatResult(
        text=message.content,
        parsed=metadata["parsed"],
        model=metadata["model"],
        prompt_tokens=0,
        completion_tokens=0,
        cost_usd=metadata["cost_usd"],
        latency_ms=metadata["latency_ms"],
        cached=metadata["cached"],
    )


def verify_step(
    state: AnswerState,
    *,
    pipeline: Literal["baseline", "final", "langgraph"] = "final",
    chat_model: Any | None = None,
) -> AnswerState:
    if pipeline == "baseline" or not state["sentences"]:
        return state
    started = state.get("started", time.perf_counter())
    calls = state.setdefault("calls", [])
    sources = state["sources"]
    sentences = state["sentences"]
    judge_inputs = [
        (sentence["text"], [sources[number - 1][1].text for number in sentence["citations"]])
        for sentence in sentences
    ]

    def judge(inputs: list[tuple[str, list[str]]]) -> list[bool]:
        if chat_model is None:
            judged = get_client().chat(
                purpose="judge",
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": judge_user(inputs)},
                ],
                schema=SUPPORTED_SCHEMA,
                schema_name="supported",
                sample_index=state["sample_index"],
                budget=state["budget"],
            )
        else:
            from langchain_core.messages import HumanMessage, SystemMessage

            response = chat_model.invoke(
                [SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=judge_user(inputs))]
            )
            judged = _chat_result_from_message(response)
        calls.append(judged)
        verdicts = [False] * len(inputs)
        verdicts_raw = cast(list[dict[str, Any]], (judged.parsed or {}).get("verdicts", []))
        for verdict in verdicts_raw:
            number = verdict["sentence"]
            if 1 <= number <= len(verdicts):
                verdicts[number - 1] = verdict["supported"]
        return verdicts

    try:
        supported = check_answer(judge_inputs, judge)
    except BudgetExceeded:
        state["result"] = _declined("budget_exceeded", pipeline, started, state["dropped"], calls)
        return state
    except LLMOutputError:
        state["result"] = _declined("model_error", pipeline, started, state["dropped"], calls)
        return state
    kept = [
        sentence
        for sentence, is_supported in zip(sentences, supported, strict=True)
        if is_supported
    ]
    state["dropped"] += len(sentences) - len(kept)
    state["sentences"] = kept
    draft = state["draft"]
    assert draft is not None
    if not kept or state["dropped"] / len(draft["sentences"]) > MAX_DROPPED_SHARE:
        state["result"] = _declined("unsupported", pipeline, started, state["dropped"], calls)
    return state


def finalize_step(
    state: AnswerState,
    *,
    pipeline: Literal["baseline", "final", "langgraph"] = "final",
) -> AnswerState:
    if state["result"] is not None:
        return state
    started = state.get("started", time.perf_counter())
    calls = state.setdefault("calls", [])
    kept = state["sentences"]
    if not kept:
        state["result"] = _declined("unsupported", pipeline, started, state["dropped"], calls)
        return state
    numbers: dict[int, int] = {}
    for sentence in kept:
        for cited in sentence["citations"]:
            numbers.setdefault(cited, len(numbers) + 1)
    sources = state["sources"]
    metadata = state["span_metadata"]
    result = state["synth_result"]
    state["result"] = AnswerResult(
        declined=False,
        decline_reason=None,
        sentences=[
            AnswerSentence(sentence["text"], [numbers[cited] for cited in sentence["citations"]])
            for sentence in kept
        ],
        citations=[
            _citation(
                new,
                *sources[old - 1],
                *metadata[(sources[old - 1][1].document_id, sources[old - 1][1].unit_index)],
            )
            for old, new in numbers.items()
        ],
        pipeline=pipeline,
        model=result.model if result else None,
        cost_usd=sum(call.cost_usd for call in calls),
        latency_ms=_latency_ms(started, calls),
        cached=False,
        dropped_sentences=state["dropped"],
    )
    return state
