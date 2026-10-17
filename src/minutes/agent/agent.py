from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

import psycopg
from psycopg.rows import TupleRow

from minutes import llm
from minutes.agent import tools
from minutes.config import get_settings
from minutes.errors import BudgetExceeded, LLMOutputError, ToolError
from minutes.labels.store import require_frozen
from minutes.llm import Budget
from minutes.llm.prompts import AGENT_ACTION_SCHEMA, AGENT_SYSTEM

MAX_STEPS: Literal[8] = 8
MAX_COST_USD = 0.50
MAX_SECONDS = 120
ALLOWED_TOOLS = ("search", "sql")


def get_client() -> llm.LLMClient:
    """Get the current LLM client so evaluation can replace the factory."""
    return llm.get_client()


@dataclass(frozen=True)
class AgentStep:
    index: int
    tool: str
    arguments: dict[str, object]
    result_summary: str
    error: str | None


@dataclass(frozen=True)
class AgentResult:
    status: Literal["completed", "limit_exceeded", "error"]
    answer: str
    citations: list[dict[str, object]]
    steps: list[AgentStep]
    cost_usd: float
    limit: str | None


def _failure(
    status: Literal["limit_exceeded", "error"],
    steps: list[AgentStep],
    budget: Budget,
    limit: str | None = None,
) -> AgentResult:
    return AgentResult(status, "", [], steps, budget.spent, limit)


def run_agent(
    conn: psycopg.Connection[TupleRow],
    question: str,
    *,
    sample_index: int = 0,
    monotonic: Callable[[], float] = time.monotonic,
) -> AgentResult:
    settings = get_settings()
    if settings.corpus != "fixture":
        require_frozen()
    budget = Budget(MAX_COST_USD)
    started = monotonic()
    messages = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": question},
    ]
    steps: list[AgentStep] = []
    observations: list[object] = []
    force_final = False
    while True:
        if monotonic() - started > MAX_SECONDS:
            return _failure("limit_exceeded", steps, budget, "time")
        if force_final:
            messages.append(
                {
                    "role": "user",
                    "content": 'You have no tool calls left. Reply with action "final".',
                }
            )
        try:
            result = get_client().chat(
                purpose="agent",
                messages=messages,
                schema=AGENT_ACTION_SCHEMA,
                schema_name="agent_action",
                sample_index=sample_index,
                budget=budget,
            )
        except BudgetExceeded:
            return _failure("limit_exceeded", steps, budget, "cost")
        except LLMOutputError:
            return _failure("error", steps, budget)
        if monotonic() - started > MAX_SECONDS:
            return _failure("limit_exceeded", steps, budget, "time")
        action = cast(dict[str, Any] | None, result.parsed)
        if action is None:
            return _failure("error", steps, budget)
        if action.get("action") == "final":
            final = action.get("final")
            if not isinstance(final, dict):
                return _failure("error", steps, budget)
            seen = {
                (str(row.get("document_id")), row.get("unit_index"))
                for observation in observations
                if isinstance(observation, list)
                for row in observation
                if isinstance(row, dict)
            }
            citations = [
                citation
                for citation in cast(list[dict[str, Any]], final.get("citations", []))
                if (citation.get("document_id"), citation.get("unit_index")) in seen
            ]
            return AgentResult(
                "completed", str(final.get("answer", "")), citations, steps, budget.spent, None
            )
        if force_final:
            return _failure("limit_exceeded", steps, budget, "steps")
        tool_name = action.get("action")
        arguments: dict[str, object]
        observation: object
        error_code: str | None = None
        if tool_name not in ALLOWED_TOOLS:
            tool_name = str(tool_name)
            arguments = {}
            observation = "error: tool not allowed"
            error_code = "tool_not_allowed"
        elif tool_name == "search":
            raw = action.get("search")
            arguments = cast(dict[str, object], raw if isinstance(raw, dict) else {})
            try:
                observation = tools.search_tool(
                    conn,
                    str(arguments.get("query", "")),
                    cast(str | None, arguments.get("city")),
                    cast(str | None, arguments.get("date_from")),
                    cast(str | None, arguments.get("date_to")),
                    cast(str | None, arguments.get("body")),
                )
            except ToolError as err:
                observation = f"error: {err.code}: {err.message}"
                error_code = err.code
        else:
            query = action.get("sql")
            arguments = {"query": query} if isinstance(query, str) else {}
            try:
                observation = tools.sql_tool(query if isinstance(query, str) else "")
            except ToolError as err:
                observation = f"error: {err.code}: {err.message}"
                error_code = err.code
        observations.append(observation)
        index = len(steps) + 1
        if error_code:
            summary = error_code
        elif tool_name == "search":
            summary = (
                f"{len(observation)} results" if isinstance(observation, list) else "0 results"
            )
        else:
            rows = observation.get("rows", []) if isinstance(observation, dict) else []
            summary = f"{len(rows)} rows" if isinstance(rows, list) else "0 rows"
        steps.append(AgentStep(index, str(tool_name), arguments, summary, error_code))
        serialized = json.dumps(observation, default=str)[:6000]
        messages.append({"role": "user", "content": f"OBSERVATION {index}: {serialized}"})
        if len(steps) == MAX_STEPS:
            force_final = True
