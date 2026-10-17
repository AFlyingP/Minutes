from __future__ import annotations

import time
from typing import Any, cast

import psycopg
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs import ChatResult as LangChainChatResult
from langgraph.graph import END, START, StateGraph
from psycopg.rows import TupleRow

from minutes import llm
from minutes.answer import cache
from minutes.answer.pipeline import (
    AnswerResult,
    AnswerState,
    finalize_step,
    retrieve_step,
    synthesize_step,
    verify_step,
)
from minutes.llm import Budget
from minutes.llm.prompts import SUPPORTED_SCHEMA, SYNTH_SCHEMA
from minutes.retrieval.search import Filters


def get_client() -> llm.LLMClient:
    return llm.get_client()


class ThinChatModel(BaseChatModel):
    purpose: str
    response_schema: dict[str, Any] | None = None
    schema_name: str = "output"
    model_id: str | None = None
    sample_index: int = 0
    budget: Budget | None = None

    def __init__(self, **data: Any) -> None:
        if "schema" in data:
            data["response_schema"] = data.pop("schema")
        super().__init__(**data)

    @property
    def _llm_type(self) -> str:
        return "thin-client"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> LangChainChatResult:
        del stop, run_manager, kwargs
        client_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                raise TypeError(f"unsupported chat message type: {type(message).__name__}")
            client_messages.append({"role": role, "content": str(message.content)})
        result = get_client().chat(
            purpose=self.purpose,
            messages=client_messages,
            schema=self.response_schema,
            schema_name=self.schema_name,
            model=self.model_id,
            sample_index=self.sample_index,
            budget=self.budget,
        )
        message = AIMessage(
            content=result.text,
            response_metadata={
                "model": result.model,
                "cost_usd": result.cost_usd,
                "latency_ms": result.latency_ms,
                "cached": result.cached,
                "parsed": result.parsed,
            },
        )
        return LangChainChatResult(generations=[ChatGeneration(message=message)])


def _route_after_retrieve(state: AnswerState) -> str:
    return "finalize" if state["result"] is not None else "synthesize"


def _route_after_synthesize(state: AnswerState) -> str:
    return "finalize" if state["result"] is not None else "verify"


def _retrieve_node(state: AnswerState) -> AnswerState:
    return retrieve_step(state["conn"], state, pipeline="langgraph")


def _synthesize_node(state: AnswerState) -> AnswerState:
    model = ThinChatModel(
        purpose="synth",
        response_schema=SYNTH_SCHEMA,
        schema_name="synth",
        model_id=state["model"],
        sample_index=state["sample_index"],
        budget=state["budget"],
    )
    return synthesize_step(state, pipeline="langgraph", chat_model=model)


def _verify_node(state: AnswerState) -> AnswerState:
    model = ThinChatModel(
        purpose="judge",
        response_schema=SUPPORTED_SCHEMA,
        schema_name="supported",
        model_id=None,
        sample_index=state["sample_index"],
        budget=state["budget"],
    )
    return verify_step(state, pipeline="langgraph", chat_model=model)


def _finalize_node(state: AnswerState) -> AnswerState:
    return finalize_step(state, pipeline="langgraph")


def build_graph() -> Any:
    builder = StateGraph(AnswerState)
    builder.add_node("retrieve", _retrieve_node)
    builder.add_node("synthesize", _synthesize_node)
    builder.add_node("verify", _verify_node)
    builder.add_node("finalize", _finalize_node)
    builder.add_edge(START, "retrieve")
    builder.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {"finalize": "finalize", "synthesize": "synthesize"},
    )
    builder.add_conditional_edges(
        "synthesize",
        _route_after_synthesize,
        {"finalize": "finalize", "verify": "verify"},
    )
    builder.add_edge("verify", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()


graph = build_graph()


def answer(
    conn: psycopg.Connection[TupleRow],
    question: str,
    filters: Filters,
    *,
    controls: bool = True,
    sample_index: int = 0,
) -> AnswerResult:
    started = time.perf_counter()
    response_key = None
    if controls:
        response_key = cache.cache_key(question, filters, "langgraph", cache.corpus_version(conn))
        cached = cache.get(conn, response_key)
        if cached is not None:
            return cached
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
        "conn": conn,
    }
    final_state = cast(AnswerState, graph.invoke(state))
    result = final_state["result"]
    assert result is not None
    if result.pipeline != "langgraph":
        from dataclasses import replace

        result = replace(result, pipeline="langgraph")
    if response_key is not None and not (
        result.declined and result.decline_reason in ("budget_exceeded", "model_error")
    ):
        cache.put(conn, response_key, result)
    return result
