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


def test_unit_endpoint_returns_text_boxes_and_label(client: TestClient, conn: Connection) -> None:
    response = client.get("/api/documents/birch-minutes-201/units/1")
    assert response.status_code == 200
    unit = response.json()
    assert unit["unit_kind"] == "page"
    assert unit["text"]
    assert unit["boxes"] and all(len(box) == 6 for box in unit["boxes"])
    assert unit["label"] == "p. 1"
    assert unit["recording_link"] is None
    assert unit["width_pt"] > 0 and unit["height_pt"] > 0
    document = client.get("/api/documents/birch-minutes-201")
    assert document.status_code == 200
    assert document.json()["unit_count"] == 3
    assert document.json()["meeting"]["id"] == "birch-201"
    assert document.json()["source_url"]
    for path in ("/api/documents/missing", "/api/documents/birch-minutes-201/units/99"):
        assert client.get(path).json()["error"] == "not_found"
    conn.execute("UPDATE documents SET status = 'failed' WHERE id = 'birch-minutes-201'")
    assert client.get("/api/documents/birch-minutes-201").status_code == 404
    assert client.get("/api/documents/birch-minutes-201/units/1").status_code == 404


def test_segment_unit_has_times_and_entrytime_link(client: TestClient) -> None:
    response = client.get("/api/documents/birch-transcript-201/units/1")
    assert response.status_code == 200
    unit = response.json()
    assert unit["unit_kind"] == "segment"
    assert unit["label"] == "00:00:00"
    assert unit["start_ms"] == 0 and unit["end_ms"] > 0
    assert unit["boxes"] is None
    assert unit["text_source"] == "caption"
    assert unit["recording_link"] == "https://video.invalid/birch/201&entrytime=0"


def test_segment_unit_without_seek_links_to_recording(client: TestClient) -> None:
    response = client.get("/api/documents/alder-transcript-101/units/2")
    assert response.status_code == 200
    assert response.json()["recording_link"] == "https://video.invalid/alder/101"
    assert response.json()["label"] == "00:01:10"


def test_segment_without_recording_has_null_link(client: TestClient) -> None:
    response = client.get("/api/documents/alder-transcript-102/units/1")
    assert response.status_code == 200
    assert response.json()["recording_link"] is None


def test_page_image_is_png(client: TestClient) -> None:
    response = client.get("/api/documents/birch-minutes-201/units/1/image")
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "image/png"
    assert response.headers["Cache-Control"] == "max-age=3600"
    assert response.headers["X-Correlation-ID"]
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_page_image_404_for_segment_and_unknown_unit(client: TestClient, conn: Connection) -> None:
    for path in (
        "/api/documents/birch-transcript-201/units/1/image",
        "/api/documents/birch-minutes-201/units/99/image",
        "/api/documents/missing/units/1/image",
    ):
        response = client.get(path)
        assert response.status_code == 404
        assert response.json()["error"] == "not_found"
    conn.execute(
        "UPDATE documents SET file_path = 'tests/fixtures/corpus/missing.pdf' "
        "WHERE id = 'birch-minutes-201'"
    )
    response = client.get("/api/documents/birch-minutes-201/units/1/image")
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


def test_image_dpi_out_of_range_is_422(client: TestClient) -> None:
    for dpi in (71, 201):
        response = client.get("/api/documents/birch-minutes-201/units/1/image", params={"dpi": dpi})
        assert response.status_code == 422
        assert response.json()["error"] == "validation_error"


def test_documents_list_filters_and_orders(client: TestClient, conn: Connection) -> None:
    for document, meeting, meeting_date in (
        ("birch-list-a", "birch-list-late", "2024-04-09"),
        ("birch-list-z", "birch-list-early", "2024-03-12"),
    ):
        conn.execute(
            "INSERT INTO meetings (id, city_id, body, meeting_date, title, source_key) "
            "VALUES (%s, 'birch', 'City Council', %s, 'List order meeting', %s)",
            (meeting, meeting_date, meeting),
        )
        conn.execute(
            "INSERT INTO documents (id, meeting_id, city_id, kind, source_url, media_type, "
            "unit_kind, status, unit_count) VALUES (%s, %s, 'birch', 'minutes', "
            "'https://fixture.invalid/minutes', 'application/pdf', 'page', 'downloaded', 1)",
            (document, meeting),
        )
    params = {"city": "birch", "kind": "minutes"}
    response = client.get("/api/documents", params=params)
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 4
    assert [doc["id"] for doc in body["documents"]] == [
        "birch-list-z",
        "birch-minutes-201",
        "birch-list-a",
        "birch-minutes-202",
    ]
    assert [doc["meeting_date"] for doc in body["documents"]] == [
        "2024-03-12",
        "2024-03-12",
        "2024-04-09",
        "2024-04-09",
    ]
    page = client.get("/api/documents", params={**params, "limit": 1, "offset": 1}).json()
    assert page["total"] == 4
    assert [doc["id"] for doc in page["documents"]] == ["birch-minutes-201"]
    assert client.get("/api/documents").json()["total"] == 14
    for invalid in ({"city": "missing"}, {"kind": "missing"}, {"limit": 501}, {"offset": -1}):
        assert client.get("/api/documents", params=invalid).status_code == 422


def test_meeting_endpoint_lists_documents(client: TestClient) -> None:
    response = client.get("/api/meetings/birch-201")
    assert response.status_code == 200
    meeting = response.json()
    assert meeting["city_id"] == "birch"
    assert meeting["body"] == "City Council"
    assert meeting["meeting_date"] == "2024-03-12"
    assert meeting["recording_url"] == "https://video.invalid/birch/201"
    assert meeting["documents"] == [
        {"id": "birch-agenda-201", "kind": "agenda", "status": "downloaded", "unit_count": 2},
        {"id": "birch-minutes-201", "kind": "minutes", "status": "downloaded", "unit_count": 3},
        {
            "id": "birch-transcript-201",
            "kind": "transcript",
            "status": "downloaded",
            "unit_count": 2,
        },
    ]
    missing = client.get("/api/meetings/missing")
    assert missing.status_code == 404
    assert missing.json()["error"] == "not_found"
