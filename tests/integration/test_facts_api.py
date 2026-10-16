from collections.abc import Iterator

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import TupleRow

from minutes.api import routes_core
from minutes.api.app import create_app

pytestmark = pytest.mark.integration
Connection = psycopg.Connection[TupleRow]


@pytest.fixture
def client(fixture_corpus: str, conn: Connection) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[routes_core.get_conn] = lambda: conn
    with TestClient(app) as client:
        yield client


def test_votes_for_birch_are_four_rows_newest_first(client: TestClient) -> None:
    response = client.get("/api/facts/votes", params={"city": "birch"})
    assert response.status_code == 200
    result = response.json()
    assert result["kind"] == "votes"
    assert result["total"] == 4
    rows = result["rows"]
    assert rows[0]["document_id"] == "birch-minutes-202"
    dates = [row["meeting_date"] for row in rows]
    assert dates == sorted(dates, reverse=True)

    vote = next(
        row for row in rows if row["data"].get("subject_identifier") == "RESOLUTION 2024-07"
    )
    assert vote["data"]["ayes"] == 4
    assert vote["data"]["noes"] == 1
    assert vote["data"]["outcome"] == "passed"
    assert vote["label"] == "p. 2"


def test_unknown_kind_is_404(client: TestClient) -> None:
    response = client.get("/api/facts/unknown")

    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


def test_q_filters_on_quote(client: TestClient) -> None:
    response = client.get("/api/facts/votes", params={"city": "birch", "q": "gray, hale"})

    assert response.status_code == 200
    result = response.json()
    assert result["total"] == 2
    assert all("gray, hale" in row["quote"].casefold() for row in result["rows"])

    all_birch_votes = client.get("/api/facts/votes", params={"city": "birch"}).json()["rows"]
    table_row_vote = next(
        row
        for row in all_birch_votes
        if row["document_id"] == "birch-minutes-201" and "gray | aye" in row["quote"].casefold()
    )
    assert table_row_vote["id"] not in {row["id"] for row in result["rows"]}

    no_match = client.get("/api/facts/votes", params={"city": "birch", "q": "no matching quote"})
    assert no_match.status_code == 200
    assert no_match.json()["total"] == 0


def test_limit_offset_paging(client: TestClient) -> None:
    all_rows = client.get("/api/facts/votes", params={"city": "birch"}).json()
    first = client.get("/api/facts/votes", params={"city": "birch", "limit": 2, "offset": 0}).json()
    second = client.get(
        "/api/facts/votes", params={"city": "birch", "limit": 2, "offset": 2}
    ).json()

    assert first["total"] == second["total"] == 4
    assert [row["id"] for row in first["rows"] + second["rows"]] == [
        row["id"] for row in all_rows["rows"]
    ]


def test_unverified_facts_are_not_returned(client: TestClient, conn: Connection) -> None:
    before = client.get("/api/facts/votes", params={"city": "birch"}).json()
    hidden_id = before["rows"][0]["id"]
    conn.execute("UPDATE facts SET verified = false WHERE id = %s", (hidden_id,))

    result = client.get("/api/facts/votes", params={"city": "birch"}).json()

    assert result["total"] == 3
    assert hidden_id not in [row["id"] for row in result["rows"]]


def test_amounts_expose_numeric_amount(client: TestClient) -> None:
    response = client.get("/api/facts/amounts", params={"city": "birch"})

    assert response.status_code == 200
    amounts = [row["data"]["amount_usd"] for row in response.json()["rows"]]
    assert 184500 in amounts
    assert all(isinstance(amount, (int, float)) for amount in amounts)
