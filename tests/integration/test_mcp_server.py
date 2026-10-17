from __future__ import annotations

import json
from typing import Any

import anyio
from mcp.client import ClientSession
from mcp.client._memory import InMemoryTransport

from minutes.agent.mcp_server import build_server


async def _call(name: str, arguments: dict[str, object] | None = None) -> Any:
    async with (
        InMemoryTransport(build_server()) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()
        return await session.call_tool(name, arguments or {})


def call(name: str, arguments: dict[str, object] | None = None) -> Any:
    return anyio.run(_call, name, arguments)


def text(result: Any) -> str:
    return str(result.content[0].text)


def test_lists_exactly_search_and_sql_tools(fixture_corpus: str) -> None:
    async def list_tools() -> list[str]:
        async with (
            InMemoryTransport(build_server()) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            result = await session.list_tools()
            return sorted(tool.name for tool in result.tools)

    assert anyio.run(list_tools) == ["search", "sql"]


def test_search_tool_returns_passages_as_json(fixture_corpus: str) -> None:
    result = call("search", {"query": "sidewalk repair contract", "city": "birch"})
    assert not result.is_error
    passages = json.loads(text(result))
    assert passages[0]["document_id"].startswith("birch-")


def test_sql_tool_returns_columns_and_rows(fixture_corpus: str) -> None:
    result = call("sql", {"query": "SELECT count(*) AS n FROM v_votes WHERE city_id = 'birch'"})
    assert not result.is_error
    assert json.loads(text(result)) == {"columns": ["n"], "rows": [[4]]}


def test_sql_write_is_rejected_through_mcp(fixture_corpus: str, conn: Any) -> None:
    before = conn.execute("SELECT count(*) FROM facts").fetchone()[0]
    result = call("sql", {"query": "DELETE FROM facts"})
    after = conn.execute("SELECT count(*) FROM facts").fetchone()[0]
    assert result.is_error
    assert text(result).startswith(("permission_denied", "read_only_violation"))
    assert after == before


def test_sql_row_limit_applies_through_mcp(fixture_corpus: str) -> None:
    result = call("sql", {"query": "SELECT generate_series(1, 201)"})
    assert result.is_error
    assert text(result).startswith("row_limit_exceeded")


def test_sql_timeout_applies_through_mcp(fixture_corpus: str) -> None:
    result = call("sql", {"query": "SELECT pg_sleep(4)"})
    assert result.is_error
    assert text(result).startswith("timeout")


def test_invalid_search_arguments_return_tool_error(fixture_corpus: str) -> None:
    result = call("search", {"query": "sidewalk repair", "date_from": "2024-13-01"})
    assert result.is_error
    assert text(result).startswith("invalid_arguments")
