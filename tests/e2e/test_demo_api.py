import pytest
from fastapi.testclient import TestClient

from minutes.api.app import create_app

pytestmark = pytest.mark.e2e


def test_demo_step_3_search(fixture_corpus: str) -> None:
    client = TestClient(create_app())
    response = client.get("/api/search", params={"q": "sidewalk repair contract", "city": "birch"})
    assert response.status_code == 200
    result = response.json()
    assert result["mode"] == "hybrid_rerank"
    assert result["chunker"] == "item"
    hit = result["hits"][0]
    assert hit["rank"] == 1
    assert hit["heading"] == "Birch City Council, 2024-03-12, minutes, item 9.B"
    assert "Granite Works Inc." in hit["snippet"]
    assert hit["spans"][0]["label"] == "p. 1"
    assert {(span["document_id"], span["unit_index"]) for span in hit["spans"]} >= {
        ("birch-minutes-201", 1),
        ("birch-minutes-201", 2),
        ("birch-agenda-201", 2),
    }


def test_search_then_ask_returns_cited_answer(fixture_corpus: str) -> None:
    client = TestClient(create_app())

    hits = client.get("/api/search", params={"q": "sidewalk repair"}).json()["hits"]
    assert "birch" in [hit["city_id"] for hit in hits[:5]]

    request = {"question": "Granite Works contract vote", "city": "birch", "pipeline": "baseline"}
    answer = client.post("/api/ask", json=request).json()
    assert answer["declined"] is False
    assert len(answer["sentences"]) >= 1
    documents = {c["number"]: c["document_id"] for c in answer["citations"]}
    for sentence in answer["sentences"]:
        assert sentence["citations"]
        assert any(documents[n].startswith("birch-") for n in sentence["citations"])


def test_demo_step_5_facts(fixture_corpus: str) -> None:
    client = TestClient(create_app())
    response = client.get("/api/facts/votes", params={"city": "birch"})

    assert response.status_code == 200
    result = response.json()
    assert result["total"] == 4
    assert result["rows"][0]["document_id"] == "birch-minutes-202"
    vote = next(
        row
        for row in result["rows"]
        if row["data"].get("subject_identifier") == "RESOLUTION 2024-07"
    )
    assert vote["data"]["ayes"] == 4
    assert vote["label"] == "p. 2"


def test_demo_step_6_ask(fixture_corpus: str) -> None:
    client = TestClient(create_app())
    response = client.post(
        "/api/ask",
        json={
            "question": "How did the council vote on the Granite Works contract in "
            "Resolution 2024-07?",
            "city": "birch",
        },
    )

    assert response.status_code == 200
    answer = response.json()
    assert answer["declined"] is False
    assert answer["sentences"] == [
        {"text": "The roll call vote was 4 ayes and 1 noes.", "citations": [1]}
    ]
    assert answer["citations"][0]["document_id"] == "birch-minutes-201"
    assert answer["citations"][0]["unit_index"] == 2


def test_demo_step_7_decline(fixture_corpus: str) -> None:
    client = TestClient(create_app())
    response = client.post(
        "/api/ask", json={"question": "What was decided about the airport runway extension?"}
    )

    assert response.status_code == 200
    answer = response.json()
    assert answer["declined"] is True
    assert answer["decline_reason"] == "no_retrieval"
    assert answer["decline_message"] == "I can't answer that from the indexed records."
    assert answer["sentences"] == []


def test_demo_step_8_agent(fixture_corpus: str) -> None:
    response = TestClient(create_app()).post(
        "/api/agent",
        json={"question": "Which sidewalk repair contract items passed, and by what vote?"},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "completed"
    assert len(result["steps"]) == 2
    assert "RESOLUTION 2024-07 passed 4-1" in result["answer"]
