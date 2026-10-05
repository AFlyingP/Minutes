import pytest
from fastapi.testclient import TestClient

from minutes.api.app import create_app

pytestmark = pytest.mark.e2e


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
