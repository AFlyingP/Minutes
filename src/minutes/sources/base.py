from dataclasses import dataclass
from datetime import date
from typing import Literal, Protocol

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
