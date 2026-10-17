from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

import psycopg
from psycopg.rows import TupleRow

from minutes.config import get_settings
from minutes.errors import ToolError, ValidationError
from minutes.retrieval.search import Filters, city_names, search


def search_tool(
    conn: psycopg.Connection[TupleRow],
    query: str,
    city: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    body: str | None = None,
) -> list[dict[str, object]]:
    try:
        start = date.fromisoformat(date_from) if date_from is not None else None
        end = date.fromisoformat(date_to) if date_to is not None else None
    except ValueError as err:
        raise ToolError("invalid_arguments", "dates must use YYYY-MM-DD") from err
    names = city_names(conn)
    if city is not None and city not in names:
        raise ToolError("invalid_arguments", f"unknown city {city}")
    try:
        hits = search(
            conn,
            query,
            Filters(city=city, date_from=start, date_to=end, body=body),
            mode="hybrid_rerank",
            chunker="item",
            k=5,
        )
    except ValidationError as err:
        raise ToolError("invalid_arguments", str(err)) from err
    return [
        {
            "document_id": span.document_id,
            "unit_index": span.unit_index,
            "city": names[hit.city_id],
            "body": hit.body,
            "date": hit.meeting_date.isoformat(),
            "kind": hit.doc_kind,
            "item": hit.item_identifier,
            "text": "\n".join(span.text for span in hit.spans)[:1500],
        }
        for hit in hits
        for span in hit.spans[:1]
    ]


def _json_value(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def sql_tool(query: str) -> dict[str, object]:
    if not query.strip() or len(query.strip()) > 4000:
        raise ToolError("invalid_arguments", "query must be 1 to 4000 characters")
    settings = get_settings()
    url = (
        settings.sql_tool_test_database_url
        if settings.corpus == "fixture"
        else settings.sql_tool_database_url
    )
    conn: psycopg.Connection[TupleRow] | None = None
    try:
        conn = psycopg.connect(url, connect_timeout=10)
        conn.execute("BEGIN READ ONLY")
        conn.execute("SET LOCAL statement_timeout = 3000")
        cursor = conn.execute(query, prepare=True)
        if cursor.description is None:
            return {"columns": [], "rows": []}
        rows = cursor.fetchmany(201)
        if len(rows) > 200:
            raise ToolError("row_limit_exceeded", "query returned more than 200 rows")
        columns = [description.name for description in cursor.description]
        return {
            "columns": columns,
            "rows": [[_json_value(value) for value in row] for row in rows],
        }
    except ToolError:
        raise
    except psycopg.Error as err:
        state = err.sqlstate
        message = str(err).strip()
        if state == "57014":
            code = "timeout"
        elif state == "42501":
            code = "permission_denied"
        elif state == "25006":
            code = "read_only_violation"
        elif state == "42601" and "multiple commands" in message.lower():
            code = "multiple_statements"
        else:
            code = "sql_error"
        raise ToolError(code, message) from err
    finally:
        if conn is not None:
            try:
                conn.rollback()
            finally:
                conn.close()
