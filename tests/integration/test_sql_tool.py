import time

import pytest

from minutes import db
from minutes.agent.tools import sql_tool
from minutes.config import get_settings
from minutes.errors import ToolError

pytestmark = pytest.mark.integration


def _assert_tool_error(query: str, *codes: str) -> None:
    with pytest.raises(ToolError) as caught:
        sql_tool(query)
    assert caught.value.code in codes


def test_select_on_allowed_view_returns_columns_and_rows(fixture_corpus: str) -> None:
    result = sql_tool("SELECT subject_identifier, ayes FROM v_votes ORDER BY id LIMIT 2")
    assert result["columns"] == ["subject_identifier", "ayes"]
    rows = result["rows"]
    assert isinstance(rows, list) and len(rows) == 2


def test_insert_is_rejected(fixture_corpus: str) -> None:
    _assert_tool_error(
        "INSERT INTO cities VALUES ('x', 'X', 'XX', 'legistar')",
        "permission_denied",
        "read_only_violation",
    )
    with db.connect(fixture_corpus) as conn:
        assert conn.execute("SELECT count(*) FROM cities WHERE id = 'x'").fetchone() == (0,)


def test_update_is_rejected(fixture_corpus: str) -> None:
    _assert_tool_error(
        "UPDATE cities SET name = 'X' WHERE id = 'alder'",
        "permission_denied",
        "read_only_violation",
    )
    with db.connect(fixture_corpus) as conn:
        assert conn.execute("SELECT name FROM cities WHERE id = 'alder'").fetchone() == ("Alder",)


def test_delete_is_rejected(fixture_corpus: str) -> None:
    _assert_tool_error(
        "DELETE FROM cities WHERE id = 'alder'", "permission_denied", "read_only_violation"
    )
    with db.connect(fixture_corpus) as conn:
        assert conn.execute("SELECT count(*) FROM cities WHERE id = 'alder'").fetchone() == (1,)


def test_drop_table_is_rejected(fixture_corpus: str) -> None:
    _assert_tool_error("DROP TABLE cities", "permission_denied", "read_only_violation")
    with db.connect(fixture_corpus) as conn:
        assert conn.execute("SELECT to_regclass('public.cities')").fetchone() == ("cities",)


@pytest.mark.parametrize("table", ["labels", "jobs", "units"])
def test_select_on_table_outside_allowlist_is_rejected(fixture_corpus: str, table: str) -> None:
    _assert_tool_error(f"SELECT * FROM {table}", "permission_denied")


def test_query_exceeding_timeout_is_rejected() -> None:
    started = time.monotonic()
    _assert_tool_error("SELECT pg_sleep(4)", "timeout")
    assert time.monotonic() - started < 5


def test_two_statements_in_one_call_are_rejected() -> None:
    _assert_tool_error("SELECT 1; SELECT 2", "multiple_statements")


def test_more_than_200_rows_is_rejected() -> None:
    _assert_tool_error("SELECT generate_series(1, 201)", "row_limit_exceeded")
    rows = sql_tool("SELECT generate_series(1, 200)")["rows"]
    assert isinstance(rows, list) and len(rows) == 200


def test_set_role_and_set_timeout_do_not_persist() -> None:
    sql_tool("SET statement_timeout = 0")
    started = time.monotonic()
    _assert_tool_error("SELECT pg_sleep(4)", "timeout")
    assert time.monotonic() - started < 5


@pytest.mark.parametrize(
    "query",
    [
        "COPY cities TO STDOUT",
        "CREATE FUNCTION public.bad() RETURNS int LANGUAGE sql AS 'SELECT 1'",
    ],
)
def test_copy_and_create_function_are_rejected(query: str) -> None:
    with pytest.raises(ToolError):
        sql_tool(query)


def test_tool_uses_readonly_url_not_application_url(fixture_corpus: str) -> None:
    settings = get_settings()
    assert settings.sql_tool_test_database_url != settings.test_database_url
    assert sql_tool("SELECT current_user AS who")["rows"] == [["minutes_ro"]]


@pytest.mark.parametrize("query", ["", " " * 2, "x" * 4001])
def test_empty_and_overlong_query_are_invalid_arguments(query: str) -> None:
    _assert_tool_error(query, "invalid_arguments")
