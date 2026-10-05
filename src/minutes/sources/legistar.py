import re
from datetime import date
from typing import Any

import httpx

from minutes.config import CityConfig
from minutes.errors import PermanentStageError
from minutes.sources.base import (
    USER_AGENT,
    DocRef,
    MediaType,
    MeetingRef,
    check_status,
    get,
    new_client,
)

CAPTION_TRACK = re.compile(
    r'tracks:\s*\[\s*\{\s*file:\s*"'
    r'(documents/seattlechannel/closedcaption/\d{4}/[A-Za-z0-9_]+\.(?:srt|vtt))"',
    re.I,
)


class LegistarSource:
    def __init__(
        self,
        city_id: str,
        city: CityConfig,
        date_from: date,
        date_to: date,
        client: httpx.Client | None = None,
    ) -> None:
        self.city = city
        self.city_id = city_id
        self.date_from = date_from
        self.date_to = date_to
        self.client = client or new_client()
        self.client.headers.setdefault("User-Agent", USER_AGENT)

    def list_meetings(
        self, date_from: date, date_to: date
    ) -> list[tuple[MeetingRef, list[DocRef]]]:
        date_from = max(date_from, self.date_from)
        date_to = min(date_to, self.date_to)
        bodies = " or ".join(f"EventBodyName eq '{body}'" for body in self.city.bodies)
        params = {
            "$filter": f"EventDate ge datetime'{date_from.isoformat()}' and "
            f"EventDate le datetime'{date_to.isoformat()}' and ({bodies})",
            "$orderby": "EventDate,EventId",
            "$top": "1000",
        }
        url = f"https://webapi.legistar.com/v1/{self.city.legistar_client}/events"
        events: list[dict[str, Any]] = []
        while True:
            response = get(
                self.client,
                url,
                params={**params, "$skip": str(len(events))},
            )
            check_status(response)
            page = response.json()
            events.extend(page)
            if len(page) < 1000:
                break
        return [self._event(event) for event in events]

    def _event(self, event: dict[str, Any]) -> tuple[MeetingRef, list[DocRef]]:
        event_id = str(event["EventId"])
        body = event["EventBodyName"]
        meeting_date = date.fromisoformat(event["EventDate"][:10])
        recording_url = event["EventMedia"]
        if not isinstance(recording_url, str) or not recording_url.startswith("http"):
            recording_url = None
        meeting_id = f"{self.city_id}-{event_id}"
        meeting = MeetingRef(
            id=meeting_id,
            city_id=self.city_id,
            body=body,
            meeting_date=meeting_date,
            title=f"{body} {meeting_date.isoformat()}",
            source_key=event_id,
            recording_url=recording_url,
            recording_seek=None,
        )
        documents: list[DocRef] = []
        agenda_url = event["EventAgendaFile"]
        if agenda_url is not None:
            documents.append(
                DocRef(
                    f"{self.city_id}-agenda-{event_id}",
                    meeting_id,
                    self.city_id,
                    "agenda",
                    agenda_url,
                    "application/pdf",
                )
            )
        minutes_url = event["EventMinutesFile"]
        if minutes_url is not None:
            documents.append(
                DocRef(
                    f"{self.city_id}-minutes-{event_id}",
                    meeting_id,
                    self.city_id,
                    "minutes",
                    minutes_url,
                    "application/pdf",
                )
            )
        if recording_url is not None:
            response = get(self.client, recording_url, follow_redirects=True)
            if 400 <= response.status_code < 500:
                return meeting, documents
            check_status(response)
            track = CAPTION_TRACK.search(response.text)
            if track:
                path = track[1]
                media_type: MediaType = (
                    "application/x-subrip" if path.lower().endswith(".srt") else "text/vtt"
                )
                documents.append(
                    DocRef(
                        f"{self.city_id}-transcript-{event_id}",
                        meeting_id,
                        self.city_id,
                        "transcript",
                        f"https://www.seattlechannel.org/{path}",
                        media_type,
                    )
                )
        return meeting, documents

    def fetch(self, ref: DocRef) -> bytes:
        response = get(self.client, ref.source_url, follow_redirects=True)
        check_status(response)
        if ref.media_type == "application/pdf" and not response.content.startswith(b"%PDF"):
            raise PermanentStageError("not a pdf")
        return response.content
