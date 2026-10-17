from collections.abc import Callable
from typing import Any, cast

import pytest

from minutes.agent import agent
from minutes.errors import BudgetExceeded, LLMOutputError, ToolError
from minutes.llm import ChatResult


def _action(
    name: str,
    *,
    search: dict[str, object] | None = None,
    sql: str | None = None,
    answer: str = "done",
    citations: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "thought": "test",
        "action": name,
        "search": search,
        "sql": sql,
        "final": {"answer": answer, "citations": citations or []} if name == "final" else None,
    }


class ScriptedClient:
    def __init__(
        self, values: list[object], inspect: Callable[[list[dict[str, str]]], None] | None = None
    ):
        self.values = values
        self.messages: list[list[dict[str, str]]] = []
        self.inspect = inspect

    def chat(
        self, *, messages: list[dict[str, str]], budget: Any = None, **kwargs: Any
    ) -> ChatResult:
        self.messages.append(messages)
        if self.inspect:
            self.inspect(messages)
        value = self.values.pop(0)
        if isinstance(value, BaseException):
            raise value
        if budget is not None:
            budget.check(0.01)
            budget.add(0.01)
        return ChatResult("{}", cast(dict[str, object], value), "fake", 1, 1, 0.01, 0, False)


@pytest.fixture
def fake_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "minutes.agent.agent.tools.search_tool",
        lambda *args, **kwargs: [{"document_id": "d1", "unit_index": 2, "text": "record"}],
    )
    monkeypatch.setattr(
        "minutes.agent.agent.tools.sql_tool", lambda query: {"columns": ["n"], "rows": [[1]]}
    )


def _client(monkeypatch: pytest.MonkeyPatch, client: ScriptedClient) -> None:
    monkeypatch.setattr(agent, "get_client", lambda: client)


def test_final_on_first_call_completes_with_zero_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    client = ScriptedClient([_action("final", answer="answer")])
    _client(monkeypatch, client)
    result = agent.run_agent(object(), "question")  # type: ignore[arg-type]
    assert (result.status, result.answer, result.steps) == ("completed", "answer", [])


def test_search_then_sql_then_final_records_two_steps(
    fake_tools: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = ScriptedClient(
        [
            _action(
                "search",
                search={
                    "query": "q",
                    "city": None,
                    "date_from": None,
                    "date_to": None,
                    "body": None,
                },
            ),
            _action("sql", sql="SELECT 1"),
            _action("final", citations=[{"document_id": "d1", "unit_index": 2}]),
        ]
    )
    _client(monkeypatch, client)
    result = agent.run_agent(object(), "question")  # type: ignore[arg-type]
    assert result.status == "completed"
    assert [step.tool for step in result.steps] == ["search", "sql"]
    assert [step.result_summary for step in result.steps] == ["1 results", "1 rows"]
    assert len(result.citations) == 1


def test_step_limit_forces_final_then_stops(
    fake_tools: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = ScriptedClient([_action("search") for _ in range(9)])
    _client(monkeypatch, client)
    result = agent.run_agent(object(), "question")  # type: ignore[arg-type]
    assert (result.status, result.limit, len(result.steps)) == ("limit_exceeded", "steps", 8)
    assert (
        'You have no tool calls left. Reply with action "final".'
        in client.messages[-1][-1]["content"]
    )


def test_cost_limit_ends_run(monkeypatch: pytest.MonkeyPatch) -> None:
    client = ScriptedClient([BudgetExceeded("budget")])
    _client(monkeypatch, client)
    result = agent.run_agent(object(), "question")  # type: ignore[arg-type]
    assert (result.status, result.limit, result.answer) == ("limit_exceeded", "cost", "")


def test_time_limit_ends_run(fake_tools: None, monkeypatch: pytest.MonkeyPatch) -> None:
    client = ScriptedClient([_action("search")])
    _client(monkeypatch, client)
    ticks = iter([0.0, 0.0, 0.0, 121.0])
    result = agent.run_agent(object(), "question", monotonic=lambda: next(ticks))  # type: ignore[arg-type]
    assert (result.status, result.limit, len(result.steps)) == ("limit_exceeded", "time", 1)


def test_disallowed_tool_is_refused_and_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    client = ScriptedClient([_action("write"), _action("final")])
    _client(monkeypatch, client)
    result = agent.run_agent(object(), "question")  # type: ignore[arg-type]
    assert result.steps[0].error == "tool_not_allowed"
    assert result.steps[0].result_summary == "tool_not_allowed"


def test_tool_error_becomes_observation_and_run_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "minutes.agent.agent.tools.search_tool",
        lambda *args, **kwargs: (_ for _ in ()).throw(ToolError("timeout", "late")),
    )
    client = ScriptedClient([_action("search"), _action("final")])
    _client(monkeypatch, client)
    result = agent.run_agent(object(), "question")  # type: ignore[arg-type]
    assert result.status == "completed"
    assert result.steps[0].error == "timeout"
    assert "error: timeout: late" in client.messages[1][-1]["content"]


def test_observation_truncated_to_6000_characters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("minutes.agent.agent.tools.search_tool", lambda *args, **kwargs: "x" * 7000)
    observed = []

    def inspect(messages: list[dict[str, str]]) -> None:
        if any(item["content"].startswith("OBSERVATION") for item in messages):
            observed.append(messages[-1]["content"])

    client = ScriptedClient([_action("search"), _action("final")], inspect)
    _client(monkeypatch, client)
    agent.run_agent(object(), "question")  # type: ignore[arg-type]
    assert len(observed[0].split(": ", 1)[1]) == 6000


def test_uncited_or_unknown_citations_are_dropped(
    fake_tools: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = ScriptedClient(
        [
            _action("search"),
            _action(
                "final",
                citations=[
                    {"document_id": "d1", "unit_index": 2},
                    {"document_id": "unknown", "unit_index": 8},
                ],
            ),
        ]
    )
    _client(monkeypatch, client)
    result = agent.run_agent(object(), "question")  # type: ignore[arg-type]
    assert result.citations == [{"document_id": "d1", "unit_index": 2}]


def test_invalid_model_output_ends_with_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _client(monkeypatch, ScriptedClient([LLMOutputError("invalid")]))
    result = agent.run_agent(object(), "question")  # type: ignore[arg-type]
    assert (result.status, result.answer, result.citations) == ("error", "", [])
