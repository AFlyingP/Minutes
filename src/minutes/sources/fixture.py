import json
from datetime import date

from minutes.config import FIXTURE_MANIFEST
from minutes.errors import PermanentStageError
from minutes.sources.base import DocRef, MeetingRef


class FixtureSource:
    """The tracked test corpus, served from its manifest instead of a city website."""

    def __init__(self, city_id: str) -> None:
        self.city_id = city_id
        self._manifest = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))

    def list_meetings(
        self, date_from: date, date_to: date
    ) -> list[tuple[MeetingRef, list[DocRef]]]:
        listed = []
        for row in self._manifest["meetings"]:
            meeting_date = date.fromisoformat(row["meeting_date"])
            if row["city_id"] != self.city_id or not date_from <= meeting_date <= date_to:
                continue
            meeting = MeetingRef(**{**row, "meeting_date": meeting_date})
            documents = [
                DocRef(
                    id=doc["id"],
                    meeting_id=meeting.id,
                    city_id=self.city_id,
                    kind=doc["kind"],
                    source_url=f"fixture:{doc['file']}",
                    media_type=doc["media_type"],
                )
                for doc in self._manifest["documents"]
                if doc["meeting_id"] == meeting.id
            ]
            listed.append((meeting, documents))
        return listed

    def fetch(self, ref: DocRef) -> bytes:
        path = FIXTURE_MANIFEST.parent / ref.source_url.removeprefix("fixture:")
        if not path.is_file():
            raise PermanentStageError(f"fixture file missing: {path.name}")
        return path.read_bytes()
