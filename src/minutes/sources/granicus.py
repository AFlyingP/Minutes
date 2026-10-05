import html
import re
from datetime import date
from urllib.parse import urljoin, urlsplit

import httpx

from minutes.config import CityConfig
from minutes.errors import PermanentStageError
from minutes.sources.base import (
    USER_AGENT,
    DocRef,
    MeetingRef,
    check_status,
    get,
    new_client,
)

MONTHS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)
ROW_DATE = re.compile(rf"({'|'.join(MONTHS)})[a-z]* +(\d{{1,2}}), +(\d{{4}})")


def _text(fragment: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", fragment)).split())


class GranicusSource:
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
        url = f"https://{self.city.granicus_host}/ViewPublisher.php"
        response = get(self.client, url, params={"view_id": str(self.city.granicus_view_id)})
        check_status(response)
        return self._rows(response.text, date_from, date_to)

    def _rows(
        self, page: str, date_from: date, date_to: date
    ) -> list[tuple[MeetingRef, list[DocRef]]]:
        meetings: list[tuple[MeetingRef, list[DocRef]]] = []
        seen: set[str] = set()
        for row in re.findall(r"<tr[^>]*>(.*?)</tr>", page, re.S):
            clip = re.search(r"clip_id=(\d+)", row)
            if not clip or clip[1] in seen:
                continue
            first_cell = re.search(r"<td[^>]*>(.*?)</td>", row, re.S)
            row_date = ROW_DATE.search(_text(row))
            if not first_cell or not row_date:
                continue
            name = _text(first_cell[1])
            try:
                meeting_date = date(
                    int(row_date[3]), MONTHS.index(row_date[1]) + 1, int(row_date[2])
                )
            except ValueError:
                continue
            if not date_from <= meeting_date <= date_to or re.search(r"cancel", name, re.I):
                continue
            if re.search(r"planning commission", name, re.I):
                body = "Planning Commission"
            elif re.search(r"council", name, re.I):
                body = "City Council"
            else:
                continue
            seen.add(clip[1])
            clip_id = clip[1]
            meeting_id = f"{self.city_id}-{clip_id}"
            has_recording = "MediaPlayer.php" in row
            recording_url = (
                f"https://{self.city.granicus_host}/MediaPlayer.php?view_id="
                f"{self.city.granicus_view_id}&clip_id={clip_id}"
                if has_recording
                else None
            )
            meeting = MeetingRef(
                id=meeting_id,
                city_id=self.city_id,
                body=body,
                meeting_date=meeting_date,
                title=name,
                source_key=clip_id,
                recording_url=recording_url,
                recording_seek="granicus_entrytime" if recording_url else None,
            )
            documents: list[DocRef] = []
            agenda = re.search(r'href="(//[^"]*AgendaViewer\.php\?[^\"]+)"', row)
            if agenda:
                documents.append(
                    DocRef(
                        f"{self.city_id}-agenda-{clip_id}",
                        meeting_id,
                        self.city_id,
                        "agenda",
                        "https:" + html.unescape(agenda[1]),
                        "application/pdf",
                    )
                )
            minutes = re.search(
                r'<a href="(//[^"]*MinutesViewer\.php\?[^\"]+)"[^>]*>'
                r"\s*(Minutes|Meeting Minutes)\s*</a>",
                row,
            )
            if minutes:
                documents.append(
                    DocRef(
                        f"{self.city_id}-minutes-{clip_id}",
                        meeting_id,
                        self.city_id,
                        "minutes",
                        "https:" + html.unescape(minutes[1]),
                        "application/pdf",
                    )
                )
            if recording_url:
                documents.append(
                    DocRef(
                        f"{self.city_id}-transcript-{clip_id}",
                        meeting_id,
                        self.city_id,
                        "transcript",
                        f"https://{self.city.granicus_host}/videos/{clip_id}/captions.vtt",
                        "text/vtt",
                    )
                )
            meetings.append((meeting, documents))
        return meetings

    def fetch(self, ref: DocRef) -> bytes:
        if ref.media_type == "application/pdf":
            response = get(self.client, ref.source_url)
            if response.is_redirect:
                location = response.headers.get("Location")
                if location is not None:
                    target = self._redirect_target(ref.source_url, location)
                    response = get(self.client, target, follow_redirects=True)
            check_status(response)
        else:
            response = get(self.client, ref.source_url, follow_redirects=True)
            check_status(response)
        if ref.media_type == "application/pdf" and not response.content.startswith(b"%PDF"):
            raise PermanentStageError("not a pdf")
        return response.content

    def _redirect_target(self, source_url: str, location: str) -> str:
        target = urljoin(source_url, location)
        parts = urlsplit(target)
        if parts.hostname == "granicus_production_attachments.s3.amazonaws.com":
            return "https://s3.amazonaws.com/granicus_production_attachments" + parts.path
        viewer = re.search(r"DocumentViewer\.php\?file=([^&]+\.pdf)", target)
        if viewer:
            return f"https://{self.city.granicus_host}/DocumentViewer.php?file={viewer[1]}"
        return target
