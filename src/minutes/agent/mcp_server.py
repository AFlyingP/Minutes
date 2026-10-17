from __future__ import annotations

import json

from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, TextContent

from minutes import db
from minutes.agent import tools
from minutes.config import corpus_database_url, get_settings
from minutes.errors import ToolError


def _text(text: str, *, is_error: bool = False) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=is_error)


def _tool_error(err: ToolError) -> CallToolResult:
    return _text(f"{err.code}: {err.message}", is_error=True)


def build_server() -> MCPServer:
    server = MCPServer("minutes")
    database_url = corpus_database_url(get_settings().corpus)

    @server.tool(
        description=(
            "Search city council agendas, minutes, and meeting captions. Returns up to 5 passages "
            "with document_id and unit_index."
        )
    )
    def search(
        query: str,
        city: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        body: str | None = None,
    ) -> CallToolResult:
        try:
            with db.connect(database_url) as conn:
                result = tools.search_tool(conn, query, city, date_from, date_to, body)
        except ToolError as err:
            return _tool_error(err)
        return _text(json.dumps(result, ensure_ascii=False))

    @server.tool(
        description=(
            "Run one read-only PostgreSQL SELECT over the v_* views. Fails above 200 rows or "
            "3 seconds."
        )
    )
    def sql(query: str) -> CallToolResult:
        try:
            result = tools.sql_tool(query)
        except ToolError as err:
            return _tool_error(err)
        return _text(json.dumps(result, ensure_ascii=False))

    return server


def main() -> None:
    from minutes.cli import LABELS_DIR, use_corpus
    from minutes.labels.store import require_frozen
    from minutes.log import configure_logging

    corpus = use_corpus(None)
    if corpus != "fixture":
        require_frozen(LABELS_DIR)
    configure_logging(get_settings().log_level)
    build_server().run("stdio")
