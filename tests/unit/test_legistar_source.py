import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from minutes import corpus
from minutes.config import load_cities
from minutes.errors import PermanentStageError, SourceError
from minutes.sources import base
from minutes.sources.base import DocRef
from minutes.sources.legistar import LegistarSource

CITY = load_cities("full")["seattle"]
DATE_FROM = date.fromisoformat(CITY.date_from)
DATE_TO = date.fromisoformat(CITY.date_to)
CAPTION_URL = (
    "https://www.seattlechannel.org/documents/seattlechannel/closedcaption/"
    "2024/council_030524_2022419.srt"
)
VIDEO_PAGE = """
playerInstance.setup({
  sources: [{ file: "//video.seattle.gov/media/council_030524.mp4" }],
  tracks: [{ file: "documents/seattlechannel/closedcaption/2024/council_030524_2022419.srt",
             label: "English" }]
});
"""
EVENTS = [
    {
        "EventId": 5773,
        "EventBodyName": "City Council",
        "EventDate": "2024-03-05T00:00:00",
        "EventAgendaFile": "https://legistar2.granicus.com/seattle/meetings/2024/3/5773_A.pdf",
        "EventMinutesFile": "https://legistar2.granicus.com/seattle/meetings/2024/3/5773_M.pdf",
        "EventMedia": "https://www.seattlechannel.org/FullCouncil?videoid=x5773",
    },
    {
        "EventId": 5774,
        "EventBodyName": "Land Use Committee",
        "EventDate": "2024-03-06T00:00:00",
        "EventAgendaFile": "https://legistar2.granicus.com/seattle/meetings/2024/3/5774_A.pdf",
        "EventMinutesFile": None,
        "EventMedia": None,
    },
    {
        "EventId": 5775,
        "EventBodyName": "City Council",
        "EventDate": "2024-03-07T00:00:00",
        "EventAgendaFile": None,
        "EventMinutesFile": None,
        "EventMedia": "https://www.seattlechannel.org/FullCouncil?videoid=x5775",
    },
]


@pytest.fixture(autouse=True)
def no_throttle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base.HOST_THROTTLE, "wait", lambda url: None)


def test_list_meetings_builds_ids_and_documents() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.host == "webapi.legistar.com":
            return httpx.Response(200, json=EVENTS)
        if request.url.host == "www.seattlechannel.org":
            return httpx.Response(200, text=VIDEO_PAGE)
        raise AssertionError(request.url)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        listed = LegistarSource("seattle", CITY, DATE_FROM, DATE_TO, client).list_meetings(
            DATE_FROM, DATE_TO
        )

    meeting, documents = listed[0]
    assert meeting.id == "seattle-5773"
    assert meeting.body == "City Council"
    assert meeting.meeting_date == date(2024, 3, 5)
    assert meeting.title == "City Council 2024-03-05"
    assert meeting.source_key == "5773"
    assert meeting.recording_url == "https://www.seattlechannel.org/FullCouncil?videoid=x5773"
    assert [document.id for document in documents] == [
        "seattle-agenda-5773",
        "seattle-minutes-5773",
        "seattle-transcript-5773",
    ]
    assert documents[2].source_url == CAPTION_URL
    assert documents[2].media_type == "application/x-subrip"


def test_event_without_minutes_or_media_yields_only_agenda() -> None:
    event = {**EVENTS[1], "EventId": 5881}

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[event])

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        listed = LegistarSource("seattle", CITY, DATE_FROM, DATE_TO, client).list_meetings(
            DATE_FROM, DATE_TO
        )

    assert [document.id for document in listed[0][1]] == ["seattle-agenda-5881"]


def test_pagination_requests_until_short_page() -> None:
    skips: list[str] = []

    def event(event_id: int) -> dict[str, Any]:
        return {
            "EventId": event_id,
            "EventBodyName": "City Council",
            "EventDate": "2024-03-05T00:00:00",
            "EventAgendaFile": None,
            "EventMinutesFile": None,
            "EventMedia": None,
        }

    def respond(request: httpx.Request) -> httpx.Response:
        skips.append(request.url.params["$skip"])
        if request.url.params["$skip"] == "0":
            return httpx.Response(200, json=[event(i) for i in range(1000)])
        return httpx.Response(200, json=[event(1000)])

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        listed = LegistarSource("seattle", CITY, DATE_FROM, DATE_TO, client).list_meetings(
            DATE_FROM, DATE_TO
        )

    assert skips == ["0", "1000"]
    assert len(listed) == 1001


def test_video_page_without_srt_gives_no_transcript() -> None:
    event = EVENTS[0]

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.host == "webapi.legistar.com":
            return httpx.Response(200, json=[event])
        return httpx.Response(200, text="no caption track here")

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        listed = LegistarSource("seattle", CITY, DATE_FROM, DATE_TO, client).list_meetings(
            DATE_FROM, DATE_TO
        )

    assert [document.kind for document in listed[0][1]] == ["agenda", "minutes"]


@pytest.mark.parametrize(
    ("status", "error"),
    [(503, SourceError), (404, PermanentStageError)],
)
def test_fetch_maps_http_errors(status: int, error: type[Exception]) -> None:
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(status))
    ) as client:
        source = LegistarSource("seattle", CITY, DATE_FROM, DATE_TO, client)
        document = DocRef(
            "seattle-agenda-1",
            "seattle-1",
            "seattle",
            "agenda",
            "https://legistar2.granicus.com/seattle/1.pdf",
            "application/pdf",
        )
        with pytest.raises(error):
            source.fetch(document)


def test_closed_connection_is_source_error() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("server disconnected", request=request)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        source = LegistarSource("seattle", CITY, DATE_FROM, DATE_TO, client)
        with pytest.raises(SourceError, match="server disconnected") as error:
            source.list_meetings(DATE_FROM, DATE_TO)

    assert isinstance(error.value.__cause__, httpx.RemoteProtocolError)


def test_non_json_listing_is_source_error() -> None:
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text="not json"))
    ) as client:
        source = LegistarSource("seattle", CITY, DATE_FROM, DATE_TO, client)
        with pytest.raises(SourceError, match=r"^invalid json$") as error:
            source.list_meetings(DATE_FROM, DATE_TO)

    assert isinstance(error.value.__cause__, json.JSONDecodeError)


class FakeCursor:
    def __init__(self, rows: list[tuple[str, str, str]]) -> None:
        self.rows = rows

    def fetchall(self) -> list[tuple[str, str, str]]:
        return self.rows


class FakeConnection:
    def __init__(self, rows: list[tuple[str, str, str]]) -> None:
        self.rows = rows

    def execute(self, query: str, params: tuple[list[str]]) -> FakeCursor:
        return FakeCursor(self.rows)


class FakeOutputFile:
    content = ""

    def write_text(self, content: str, encoding: str) -> int:
        self.content = content
        return len(content)


class FakeConfigDir:
    def __init__(self, output: FakeOutputFile) -> None:
        self.output = output

    def __truediv__(self, name: str) -> FakeOutputFile:
        assert name == "dev_subset.json"
        return self.output


def test_dev_subset_is_seeded_and_sorted(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        (f"{city}-{kind}-{index:02d}", city, kind)
        for city in ("seattle", "lincoln")
        for kind in ("agenda", "minutes", "transcript")
        for index in range(10)
    ]
    conn = FakeConnection(rows)
    output = FakeOutputFile()
    monkeypatch.setattr(corpus, "CONFIG_DIR", cast(Path, FakeConfigDir(output)))

    first = corpus.make_dev_subset(conn, seed=7)  # type: ignore[arg-type]
    second = corpus.make_dev_subset(conn, seed=7)  # type: ignore[arg-type]
    counts = Counter((document.split("-")[0], document.split("-")[1]) for document in first)

    assert len(first) == 20
    assert first == sorted(first)
    assert first == second
    assert '"seed": 7' in output.content
    assert len(json.loads(output.content)["documents"]) == 20
    assert counts == Counter(
        {(city, "agenda"): 4 for city in ("seattle", "lincoln")}
        | {(city, "minutes"): 4 for city in ("seattle", "lincoln")}
        | {(city, "transcript"): 2 for city in ("seattle", "lincoln")}
    )
