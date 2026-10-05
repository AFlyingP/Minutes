import html
import re
from dataclasses import dataclass
from pathlib import Path

import psycopg
from psycopg.rows import TupleRow

from minutes.errors import NotFoundError, PermanentStageError
from minutes.ingest.layout import normalize_text

TIMING = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})"
)
SPEAKER = re.compile(r"^(?:>>+\s*|--\s*)?((?:SPEAKER \d+ \([A-Z ]+\))|(?:[A-Z][A-Z .'-]{1,39})):\s")
MARKER = re.compile(r"^(>>+|--)\s")


@dataclass(frozen=True)
class Cue:
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class Segment:
    index: int
    start_ms: int
    end_ms: int
    text: str
    speaker: str | None


def _time_ms(hours: str, minutes: str, seconds: str, milliseconds: str) -> int:
    return ((int(hours) * 60 + int(minutes)) * 60 + int(seconds)) * 1000 + int(milliseconds)


def _parse(data: bytes, *, webvtt: bool) -> list[Cue]:
    lines = data.decode("utf-8", errors="replace").splitlines()
    cues = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if webvtt and re.match(r"^(?:NOTE|STYLE)(?:\s|$)", line):
            while index < len(lines) and lines[index].strip():
                index += 1
            continue
        timing = TIMING.match(line)
        index += 1
        if timing is None:
            continue
        text_lines = []
        while index < len(lines):
            line = lines[index].strip()
            if not line or "-->" in line:
                break
            if (
                not webvtt
                and line.isdigit()
                and index + 1 < len(lines)
                and TIMING.match(lines[index + 1].strip())
            ):
                break
            text_lines.append(line)
            index += 1
        text = normalize_text(re.sub(r"<[^>]+>", "", html.unescape(" ".join(text_lines)))).strip()
        if text:
            cues.append(
                Cue(
                    _time_ms(timing[1], timing[2], timing[3], timing[4]),
                    _time_ms(timing[5], timing[6], timing[7], timing[8]),
                    text,
                )
            )
    if not cues:
        raise PermanentStageError("no cues")
    return cues


def parse_srt(data: bytes) -> list[Cue]:
    return _parse(data, webvtt=False)


def parse_vtt(data: bytes) -> list[Cue]:
    return _parse(data, webvtt=True)


def segment_cues(cues: list[Cue]) -> list[Segment]:
    """Group cues in file order by duration, text length, speaker markers, and gaps."""
    segments: list[Segment] = []
    for cue in cues:
        for offset in range(0, len(cue.text), 1200):
            text = cue.text[offset : offset + 1200]
            speaker = SPEAKER.match(text)
            marker = speaker is not None or MARKER.match(text) is not None
            if segments:
                previous = segments[-1]
                close = (
                    cue.end_ms - previous.start_ms > 60000
                    or len(previous.text) + 1 + len(text) > 1200
                    or (marker and len(previous.text) >= 300)
                    or cue.start_ms - previous.end_ms > 5000
                )
                if not close:
                    segments[-1] = Segment(
                        previous.index,
                        previous.start_ms,
                        cue.end_ms,
                        previous.text + " " + text,
                        previous.speaker,
                    )
                    continue
            segments.append(
                Segment(
                    len(segments) + 1,
                    cue.start_ms,
                    cue.end_ms,
                    text,
                    speaker[1] if speaker else None,
                )
            )
    return segments


def build_units(conn: psycopg.Connection[TupleRow], document_id: str) -> int:
    """Read a caption document and insert its timestamped segment units."""
    row = conn.execute(
        "SELECT file_path, media_type FROM documents WHERE id = %s", (document_id,)
    ).fetchone()
    if row is None:
        raise NotFoundError(f"unknown document {document_id}")
    file_path, media_type = row
    parser = {"application/x-subrip": parse_srt, "text/vtt": parse_vtt}[media_type]
    segments = segment_cues(parser(Path(file_path).read_bytes()))
    for segment in segments:
        conn.execute(
            "INSERT INTO units (document_id, unit_index, unit_kind, text, text_source, "
            "start_ms, end_ms, speaker, boxes, needs_ocr) "
            "VALUES (%s, %s, 'segment', %s, 'caption', %s, %s, %s, NULL, false)",
            (
                document_id,
                segment.index,
                segment.text,
                segment.start_ms,
                segment.end_ms,
                segment.speaker,
            ),
        )
    return len(segments)
