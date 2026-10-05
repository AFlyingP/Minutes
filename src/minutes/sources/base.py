from dataclasses import dataclass
from datetime import date
from threading import Event
from time import monotonic
from typing import Literal, Protocol
from urllib.parse import urljoin, urlsplit

import httpx

from minutes.errors import PermanentStageError, SourceError

DocKind = Literal["agenda", "minutes", "transcript"]
MediaType = Literal["application/pdf", "application/x-subrip", "text/vtt"]


@dataclass(frozen=True)
class MeetingRef:
    id: str
    city_id: str
    body: str
    meeting_date: date
    title: str
    source_key: str
    recording_url: str | None
    recording_seek: str | None


@dataclass(frozen=True)
class DocRef:
    id: str
    meeting_id: str
    city_id: str
    kind: DocKind
    source_url: str
    media_type: MediaType


class Source(Protocol):
    city_id: str

    def list_meetings(
        self, date_from: date, date_to: date
    ) -> list[tuple[MeetingRef, list[DocRef]]]: ...

    def fetch(self, ref: DocRef) -> bytes: ...


class HostThrottle:
    """Keep request starts at least half a second apart for each host."""

    def __init__(self) -> None:
        self._last: dict[str, float] = {}

    def wait(self, url: str) -> None:
        host = urlsplit(url).hostname or ""
        now = monotonic()
        previous = self._last.get(host)
        start = max(now, previous + 0.5) if previous is not None else now
        self._last[host] = start
        remaining = start - now
        if remaining > 0:
            Event().wait(remaining)


HOST_THROTTLE = HostThrottle()
USER_AGENT = "minutes-research/0.1"


def new_client() -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(60, connect=10),
        headers={"User-Agent": USER_AGENT},
        max_redirects=5,
    )


def get(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, str] | None = None,
    follow_redirects: bool = False,
    max_redirects: int = 5,
) -> httpx.Response:
    current_url = url
    current_params = params
    for redirect_count in range(max_redirects + 1):
        HOST_THROTTLE.wait(current_url)
        try:
            response = client.get(current_url, params=current_params, follow_redirects=False)
        except (httpx.TimeoutException, httpx.NetworkError) as err:
            raise SourceError(str(err) or type(err).__name__) from err
        if not follow_redirects or not response.is_redirect:
            return response
        location = response.headers.get("Location")
        if location is None:
            return response
        if redirect_count == max_redirects:
            raise SourceError("too many redirects")
        current_url = urljoin(str(response.url), location)
        current_params = None
    raise SourceError("too many redirects")


def check_status(response: httpx.Response) -> None:
    status = response.status_code
    if status in (404, 410):
        raise PermanentStageError(f"http {status}")
    if status >= 400:
        raise SourceError(f"http {status}")
