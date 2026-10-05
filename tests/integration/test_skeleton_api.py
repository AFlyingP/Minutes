import re

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import TupleRow
from typer.testing import CliRunner

from minutes.api.app import create_app
from minutes.cli import app

pytestmark = pytest.mark.integration


@pytest.fixture
def client(fixture_corpus: str) -> TestClient:
    return TestClient(create_app())


def test_health_ok(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "corpus": "fixture", "db": "ok"}


def test_cities_lists_alder_and_birch_with_bodies(client: TestClient) -> None:
    cities = client.get("/api/cities").json()["cities"]
    assert [city["id"] for city in cities] == ["alder", "birch"]
    assert cities[0]["bodies"] == ["City Council", "Land Use Committee"]
    assert cities[1] == {
        "id": "birch",
        "name": "Birch",
        "state": "CA",
        "bodies": ["City Council", "Planning Commission"],
        "date_min": "2024-03-12",
        "date_max": "2024-04-09",
        "documents": 6,
    }


def test_search_filters_by_city_and_date(client: TestClient) -> None:
    params = {"q": "sidewalk", "city": "birch"}
    late = client.get("/api/search", params={**params, "date_from": "2024-04-01"}).json()
    hits = client.get("/api/search", params=params).json()["hits"]
    assert late["hits"] == []
    assert hits
    assert {hit["city_id"] for hit in hits} == {"birch"}
    assert hits[0]["heading"].startswith("Birch City Council, 2024-03-12, ")
    assert hits[0]["spans"][0]["label"].startswith("p. ")


def test_search_rejects_empty_query_and_unknown_city(client: TestClient) -> None:
    empty = client.get("/api/search", params={"q": " "})
    unknown = client.get("/api/search", params={"q": "sidewalk", "city": "cedar"})
    bad_date = client.get("/api/search", params={"q": "sidewalk", "date_from": "March 1"})
    for response in (empty, unknown, bad_date):
        assert response.status_code == 422
        assert response.json()["error"] == "validation_error"
    assert unknown.json()["detail"] == "city: unknown city cedar"
    assert bad_date.json()["detail"].startswith("query.date_from: ")


def test_response_has_correlation_id_header(client: TestClient) -> None:
    generated = client.get("/api/health").headers["X-Correlation-ID"]
    supplied = "0123456789abcdef0123456789abcdef"
    echoed = client.get("/api/health", headers={"X-Correlation-ID": supplied})
    replaced = client.get("/api/health", headers={"X-Correlation-ID": "not-an-id"})
    assert re.fullmatch(r"[0-9a-f]{32}", generated)
    assert echoed.headers["X-Correlation-ID"] == supplied
    assert re.fullmatch(r"[0-9a-f]{32}", replaced.headers["X-Correlation-ID"])


def test_extract_stage_writes_unverified_facts(
    fixture_corpus: str, conn: psycopg.Connection[TupleRow]
) -> None:
    row = conn.execute(
        "SELECT count(*), count(*) FILTER (WHERE verified) FROM facts WHERE kind = 'vote'"
    ).fetchone()
    assert row is not None
    assert row[0] >= 5
    assert row[1] == 0


def test_ask_declines_when_nothing_is_retrieved(client: TestClient) -> None:
    body = client.post("/api/ask", json={"question": "zeppelin moorings"}).json()
    assert body["declined"] is True
    assert body["decline_reason"] == "no_retrieval"
    assert body["decline_message"] == "I can't answer that from the indexed records."
    assert body["sentences"] == []


def test_ask_rejects_a_pipeline_that_is_not_built_yet(client: TestClient) -> None:
    response = client.post("/api/ask", json={"question": "sidewalk", "pipeline": "final"})
    assert response.status_code == 422
    assert response.json()["detail"] == "pipeline final is not available"


def test_search_and_ask_commands_print_hits_and_citations(fixture_corpus: str) -> None:
    runner = CliRunner()
    found = runner.invoke(
        app,
        ["search", "sidewalk repair", "--city", "birch", "--mode", "keyword", "--chunker", "fixed"],
    )
    answered = runner.invoke(
        app, ["ask", "Granite Works contract vote", "--city", "birch", "--pipeline", "baseline"]
    )
    declined = runner.invoke(app, ["ask", "zeppelin moorings", "--pipeline", "baseline"])
    assert re.match(r"1 \d\.\d{3} birch-\w+-201 p\. \d Birch City Council, ", found.output)
    assert answered.output.splitlines() == [
        "The roll call vote was 4 ayes and 1 noes. [1]",
        "[1] birch-minutes-201 p. 2",
    ]
    assert declined.output == "declined: no_retrieval\n"
