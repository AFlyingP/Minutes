from datetime import date

import httpx
import pytest

from minutes.config import load_cities
from minutes.errors import PermanentStageError
from minutes.sources import base
from minutes.sources.base import DocRef
from minutes.sources.granicus import GranicusSource

CITY = load_cities("full")["lincoln"]
DATE_FROM = date.fromisoformat(CITY.date_from)
DATE_TO = date.fromisoformat(CITY.date_to)


@pytest.fixture(autouse=True)
def no_throttle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base.HOST_THROTTLE, "wait", lambda url: None)


def test_rows_parsed_with_body_mapping() -> None:
    page = """
    <table>
      <tr><td>REGULAR MEETING - City Council<br>Mar 5, 2024</td>
        <td><a href="//lincoln.granicus.com/AgendaViewer.php?clip_id=41">Agenda</a>
        <a href="//lincoln.granicus.com/MinutesViewer.php?clip_id=41">Meeting Minutes</a>
        <a href="MediaPlayer.php?view_id=1&amp;clip_id=41">Watch</a></td></tr>
      <tr><td>Planning Commission Regular Meeting<br>Apr 8, 2024</td>
        <td><a href="//lincoln.granicus.com/AgendaViewer.php?clip_id=42">Agenda</a></td></tr>
      <tr><td>Airport Committee<br>Apr 9, 2024</td><td>clip_id=43</td></tr>
      <tr><td>Canceled City Council Meeting<br>Apr 10, 2024</td><td>clip_id=44</td></tr>
    </table>
    """

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=page)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        listed = GranicusSource("lincoln", CITY, DATE_FROM, DATE_TO, client).list_meetings(
            DATE_FROM, DATE_TO
        )

    assert [meeting.id for meeting, _ in listed] == ["lincoln-41", "lincoln-42"]
    assert [meeting.body for meeting, _ in listed] == ["City Council", "Planning Commission"]
    assert listed[0][0].recording_seek == "granicus_entrytime"
    assert [document.kind for document in listed[0][1]] == ["agenda", "minutes", "transcript"]
    assert listed[0][1][2].source_url == "https://lincoln.granicus.com/videos/41/captions.vtt"


def test_minutes_link_requires_minutes_label() -> None:
    page = """
    <tr><td>City Council Regular Meeting Mar 5, 2024</td>
      <td><a href="//lincoln.granicus.com/MinutesViewer.php?clip_id=51">
        Meeting Presentations</a></td></tr>
    <tr><td>City Council Regular Meeting Mar 6, 2024</td>
      <td><a href="//lincoln.granicus.com/MinutesViewer.php?clip_id=52">Minutes</a></td></tr>
    """

    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text=page))
    ) as client:
        listed = GranicusSource("lincoln", CITY, DATE_FROM, DATE_TO, client).list_meetings(
            DATE_FROM, DATE_TO
        )

    assert [
        document.id
        for _, documents in listed
        for document in documents
        if document.kind == "minutes"
    ] == ["lincoln-minutes-52"]


def test_agenda_redirect_to_s3_uses_path_style_url() -> None:
    requests: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        if len(requests) == 1:
            return httpx.Response(
                302,
                headers={
                    "Location": "https://granicus_production_attachments.s3.amazonaws.com/"
                    "lincoln/abc.pdf"
                },
            )
        return httpx.Response(200, content=b"%PDF-1.7 body")

    document = DocRef(
        "lincoln-agenda-1",
        "lincoln-1",
        "lincoln",
        "agenda",
        "https://lincoln.granicus.com/AgendaViewer.php?clip_id=1",
        "application/pdf",
    )
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        body = GranicusSource("lincoln", CITY, DATE_FROM, DATE_TO, client).fetch(document)

    assert body.startswith(b"%PDF")
    assert requests == [
        "https://lincoln.granicus.com/AgendaViewer.php?clip_id=1",
        "https://s3.amazonaws.com/granicus_production_attachments/lincoln/abc.pdf",
    ]


def test_minutes_redirect_to_document_viewer_fetches_file_url() -> None:
    requests: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        if len(requests) == 1:
            return httpx.Response(
                302,
                headers={
                    "Location": "https://lincoln.granicus.com/DocumentViewer.php?file=minutes.pdf"
                },
            )
        return httpx.Response(200, content=b"%PDF-1.7 body")

    document = DocRef(
        "lincoln-minutes-1",
        "lincoln-1",
        "lincoln",
        "minutes",
        "https://lincoln.granicus.com/MinutesViewer.php?clip_id=1",
        "application/pdf",
    )
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        GranicusSource("lincoln", CITY, DATE_FROM, DATE_TO, client).fetch(document)

    assert requests == [
        "https://lincoln.granicus.com/MinutesViewer.php?clip_id=1",
        "https://lincoln.granicus.com/DocumentViewer.php?file=minutes.pdf",
    ]


def test_non_pdf_body_is_permanent_error() -> None:
    document = DocRef(
        "lincoln-agenda-1",
        "lincoln-1",
        "lincoln",
        "agenda",
        "https://lincoln.granicus.com/AgendaViewer.php?clip_id=1",
        "application/pdf",
    )
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, text="not a pdf"))
        ) as client,
        pytest.raises(PermanentStageError, match="not a pdf"),
    ):
        GranicusSource("lincoln", CITY, DATE_FROM, DATE_TO, client).fetch(document)


def test_rows_outside_date_range_are_skipped() -> None:
    page = """
    <tr><td>City Council Regular Meeting Jan 1, 2022</td><td>clip_id=61</td></tr>
    <tr><td>City Council Regular Meeting Jan 1, 2026</td><td>clip_id=62</td></tr>
    """
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text=page))
    ) as client:
        listed = GranicusSource("lincoln", CITY, DATE_FROM, DATE_TO, client).list_meetings(
            DATE_FROM, DATE_TO
        )

    assert listed == []
