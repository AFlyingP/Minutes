from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from minutes import db
from minutes.answer.graph_variant import ThinChatModel, answer, graph
from minutes.answer.pipeline import answer as answer_final
from minutes.api.app import create_app
from minutes.errors import BudgetExceeded
from minutes.llm import Budget, ChatResult
from minutes.llm.prompts import SUPPORTED_SCHEMA
from minutes.retrieval.search import Filters


class Client:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error = error

    def chat(self, **kwargs: Any) -> ChatResult:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        purpose = cast(str, kwargs["purpose"])
        parsed: dict[str, object] = (
            {"decline": False, "sentences": [{"text": "Answer.", "citations": [1]}]}
            if purpose == "synth"
            else {"verdicts": [{"sentence": 1, "supported": True}]}
        )
        return ChatResult("client text", parsed, "stub", 1, 1, 0.0, 0, False)


def test_graph_has_four_nodes_and_expected_edges() -> None:
    graph_view = graph.get_graph()
    nodes = set(graph_view.nodes)
    assert {"retrieve", "synthesize", "verify", "finalize"} <= nodes
    edges = {(edge.source, edge.target) for edge in graph_view.edges}
    assert {
        ("__start__", "retrieve"),
        ("retrieve", "synthesize"),
        ("retrieve", "finalize"),
        ("synthesize", "verify"),
        ("synthesize", "finalize"),
        ("verify", "finalize"),
        ("finalize", "__end__"),
    } <= edges


def test_thin_chat_model_maps_roles_and_returns_text(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Client()
    monkeypatch.setattr("minutes.answer.graph_variant.get_client", lambda: client)
    model = ThinChatModel(purpose="synth")

    message = model.invoke(
        [SystemMessage(content="system text"), HumanMessage(content="user text")]
    )

    assert client.calls[0]["messages"] == [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "user text"},
    ]
    assert isinstance(message, AIMessage)
    assert message.content == "client text"


def test_thin_chat_model_passes_purpose_schema_sample_and_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Client()
    monkeypatch.setattr("minutes.answer.graph_variant.get_client", lambda: client)
    budget = Budget(2.0)
    model = ThinChatModel(
        purpose="judge",
        schema=SUPPORTED_SCHEMA,
        schema_name="supported",
        model_id="model-x",
        sample_index=3,
        budget=budget,
    )

    model.invoke([HumanMessage(content="user text")])

    call = client.calls[0]
    assert call["purpose"] == "judge"
    assert call["schema"] == SUPPORTED_SCHEMA
    assert call["schema_name"] == "supported"
    assert call["model"] == "model-x"
    assert call["sample_index"] == 3
    assert call["budget"] is budget


@pytest.mark.integration
def test_variant_matches_final_pipeline_on_demo_question(fixture_corpus: str) -> None:
    question = "How did the council vote on the Granite Works contract in Resolution 2024-07?"
    with db.connect(fixture_corpus) as conn:
        conn.execute("DELETE FROM answer_cache")
        final = answer_final(conn, question, Filters(city="birch"), controls=False)
        variant = answer(conn, question, Filters(city="birch"), controls=False)

    assert [item.text for item in variant.sentences] == [item.text for item in final.sentences]
    assert variant.citations == final.citations
    assert variant.pipeline == "langgraph"


@pytest.mark.integration
def test_variant_declines_like_final_on_unanswerable_question(
    fixture_corpus: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = Client()
    monkeypatch.setattr("minutes.answer.graph_variant.get_client", lambda: client)
    question = "What happened at the zeppelin moorings in 2030?"

    with db.connect(fixture_corpus) as conn:
        result = answer(conn, question, Filters(city="birch"), controls=False)

    assert result.declined
    assert result.decline_reason == "no_retrieval"
    assert not client.calls


@pytest.mark.integration
def test_variant_uses_same_cache_keys_as_final(fixture_corpus: str) -> None:
    question = "How did the council vote on the Granite Works contract in Resolution 2024-07?"
    with db.connect(fixture_corpus) as conn:
        conn.execute("DELETE FROM answer_cache")
        answer_final(conn, question, Filters(city="birch"), sample_index=4)
        final_keys = conn.execute(
            "SELECT purpose, cache_key FROM llm_calls WHERE purpose IN ('synth', 'judge') "
            "ORDER BY id DESC LIMIT 2"
        ).fetchall()
        answer(conn, question, Filters(city="birch"), sample_index=4)
        variant_keys = conn.execute(
            "SELECT purpose, cache_key FROM llm_calls WHERE purpose IN ('synth', 'judge') "
            "ORDER BY id DESC LIMIT 2"
        ).fetchall()

    assert final_keys
    assert variant_keys == final_keys


@pytest.mark.integration
def test_budget_exceeded_in_node_becomes_decline(
    fixture_corpus: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = Client(BudgetExceeded("over budget"))
    monkeypatch.setattr("minutes.answer.graph_variant.get_client", lambda: client)

    with db.connect(fixture_corpus) as conn:
        result = answer(
            conn,
            "How did the council vote on the Granite Works contract in Resolution 2024-07?",
            Filters(city="birch"),
            controls=False,
        )

    assert result.declined
    assert result.decline_reason == "budget_exceeded"
    assert result.pipeline == "langgraph"


@pytest.mark.integration
def test_ask_endpoint_accepts_langgraph_pipeline(fixture_corpus: str) -> None:
    response = TestClient(create_app()).post(
        "/api/ask",
        json={
            "question": (
                "How did the council vote on the Granite Works contract in Resolution 2024-07?"
            ),
            "city": "birch",
            "pipeline": "langgraph",
        },
    )

    assert response.status_code == 200
    assert response.json()["pipeline"] == "langgraph"
