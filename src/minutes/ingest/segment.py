import re
from collections.abc import Iterable
from dataclasses import astuple, dataclass
from typing import Literal

import psycopg
from psycopg.rows import TupleRow


@dataclass(frozen=True)
class ItemSpan:
    kind: Literal["business", "section"]
    identifier: str
    title: str
    ordinal: int
    start_unit: int
    start_offset: int
    end_unit: int
    end_offset: int


@dataclass(frozen=True)
class Line:
    unit: int
    offset: int
    text: str


def _lines(units: list[tuple[int, str]]) -> list[Line]:
    lines = []
    for unit, text in units:
        offset = 0
        for line in text.split("\n"):
            lines.append(Line(unit, offset, line))
            offset += len(line) + 1
    return lines


def _title(parts: Iterable[str], stops: tuple[re.Pattern[str], ...]) -> str:
    value = ""
    for part in parts:
        if any(pattern.search(part) for pattern in stops):
            break
        value = f"{value} {part}".strip()
        if len(value) >= 300:
            return value[:300]
    return value[:300]


def _span(
    kind: Literal["business", "section"],
    identifier: str,
    title: str,
    start: Line,
    ordinal: int,
    end: Line,
) -> ItemSpan:
    return ItemSpan(
        kind, identifier, title, ordinal, start.unit, start.offset, end.unit, end.offset
    )


def _make_spans(
    entries: list[tuple[Line, Literal["business", "section"], str, str]],
    units: list[tuple[int, str]],
) -> list[ItemSpan]:
    if not entries or not units:
        return []
    entries.sort(key=lambda entry: (entry[0].unit, entry[0].offset))
    spans = []
    for index, (start, kind, identifier, title) in enumerate(entries):
        if index + 1 < len(entries):
            end_line = entries[index + 1][0]
            if end_line.offset == 0 and end_line.unit != start.unit:
                previous_index = (
                    next(
                        position
                        for position, (unit, _) in enumerate(units)
                        if unit == end_line.unit
                    )
                    - 1
                )
                previous_unit, previous_text = units[previous_index]
                end = Line(previous_unit, len(previous_text), "")
            else:
                end = Line(end_line.unit, end_line.offset, end_line.text)
        else:
            unit, text = units[-1]
            end = Line(unit, len(text), "")
        spans.append(_span(kind, identifier, title, start, index + 1, end))
    return spans


LEGISTAR_HEADING = re.compile(r"^([A-L])\. ([A-Z][A-Z ,&/'-]{3,}):?$")
LEGISTAR_ANCHOR = re.compile(r"^(?:(\d{1,3})\.\s+)?((?:CB|Res|CF|Appt|Min|IRC|Inf) \d+)\b(.*)$")
LEGISTAR_TITLE_START = re.compile(
    r"^(AN ORDINANCE|A RESOLUTION|Appointment|Reappointment|Council Bill|Clerk File)\b"
)
LEGISTAR_TITLE_STOP = tuple(
    re.compile(pattern)
    for pattern in (
        r"^The Committee recommends",
        r"^In Favor:",
        r"^Opposed:",
        r"^Supporting",
        r"^Attachments:",
        r"^The .* was ",
    )
)
GRANICUS_SUBITEM = re.compile(r"^(\d{1,2}\.[A-Z]{1,2})\.?(?:\s+|$)(.*)$")
GRANICUS_TOP = re.compile(r"^(\d{1,2})\.\s+([A-Z][A-Z0-9 ,&/()'-]{2,}.*)$")
GRANICUS_TITLE_STOP = tuple(
    re.compile(pattern)
    for pattern in (
        r"^Ayes:",
        r"^Noes:",
        r"^Seeing no",
        r"^[A-Z][a-z]+ [A-Z][a-z]+(?:,.*)? presented\b",
        r"(?i)\bmotioned\b",
    )
)


def _segment_legistar(units: list[tuple[int, str]], lines: list[Line]) -> list[ItemSpan]:
    headings: list[tuple[int, re.Match[str]]] = []
    anchors: list[tuple[int, re.Match[str]]] = []
    anchor_lines: list[int] = []
    seen: set[str] = set()
    for index, line in enumerate(lines):
        heading = LEGISTAR_HEADING.fullmatch(line.text)
        if heading:
            headings.append((index, heading))
        anchor = LEGISTAR_ANCHOR.fullmatch(line.text)
        if anchor:
            anchor_lines.append(index)
            if anchor.group(2) not in seen:
                seen.add(anchor.group(2))
                anchors.append((index, anchor))

    entries: list[tuple[Line, Literal["business", "section"], str, str]] = []
    for index, heading in headings:
        entries.append((lines[index], "section", heading.group(1), heading.group(2)))
    for anchor_index, anchor in anchors:
        boundary = max(
            (position for position, _ in headings if position < anchor_index), default=-1
        )
        boundary = max(
            boundary,
            max((position for position in anchor_lines if position < anchor_index), default=-1),
        )
        title_start = anchor_index
        has_title_start = False
        if not anchor.group(3).strip():
            candidates = [
                index
                for index in range(boundary + 1, anchor_index)
                if LEGISTAR_TITLE_START.match(lines[index].text)
            ]
            if candidates:
                title_start = candidates[0]
                has_title_start = True
            if has_title_start:
                title = _title(
                    (lines[index].text for index in range(title_start, anchor_index)), ()
                )
            else:
                title = lines[anchor_index].text
        else:
            title_parts = [anchor.group(3).strip()]
            next_heading = min(
                [position for position, _ in headings if position > anchor_index]
                + [position for position in anchor_lines if position > anchor_index],
                default=len(lines),
            )
            next_anchor = min(
                (position for position in anchor_lines if position > anchor_index),
                default=len(lines),
            )
            next_anchor_start = next_anchor
            if next_anchor < len(lines):
                next_boundary = max(
                    [position for position, _ in headings if position < next_anchor]
                    + [position for position in anchor_lines if position < next_anchor],
                    default=-1,
                )
                candidates = [
                    position
                    for position in range(next_boundary + 1, next_anchor)
                    if LEGISTAR_TITLE_START.match(lines[position].text)
                ]
                if candidates:
                    next_anchor_start = candidates[0]
            next_boundary = min(next_heading, next_anchor_start)
            title_parts.extend(
                lines[index].text for index in range(anchor_index + 1, next_boundary)
            )
            title = _title(title_parts, LEGISTAR_TITLE_STOP)
        entries.append((lines[title_start], "business", anchor.group(2), title))

    if lines:
        first_boundary = min(
            (entry[0] for entry in entries), key=lambda line: (line.unit, line.offset), default=None
        )
        if (
            first_boundary is not None
            and (first_boundary.unit, first_boundary.offset) != (lines[0].unit, lines[0].offset)
        ) or (not headings and not anchors):
            entries.append((lines[0], "section", "PREAMBLE", "Preamble"))
    return _make_spans(entries, units)


def _segment_granicus(units: list[tuple[int, str]], lines: list[Line]) -> list[ItemSpan]:
    anchors: list[tuple[int, Literal["business", "section"], str, str]] = []
    subitem_lines: list[int] = []
    seen: set[str] = set()
    for index, line in enumerate(lines):
        subitem = GRANICUS_SUBITEM.fullmatch(line.text)
        if subitem:
            subitem_lines.append(index)
            if subitem.group(1) not in seen:
                seen.add(subitem.group(1))
                anchors.append((index, "business", subitem.group(1), subitem.group(2)))
            continue
        top = GRANICUS_TOP.fullmatch(line.text)
        if top and top.group(1) not in seen:
            identifier = f"{top.group(1)}."
            seen.add(top.group(1))
            anchors.append((index, "section", identifier, top.group(2)[:300]))

    entries: list[tuple[Line, Literal["business", "section"], str, str]] = []
    for index, kind, identifier, title_start in anchors:
        title = title_start
        if kind == "business":
            next_anchor = min(
                [entry[0] for entry in anchors if entry[0] > index]
                + [position for position in subitem_lines if position > index],
                default=len(lines),
            )
            title = _title(
                [
                    title_start,
                    *(lines[position].text for position in range(index + 1, next_anchor)),
                ],
                GRANICUS_TITLE_STOP,
            )
        entries.append((lines[index], kind, identifier, title))
    if lines and ((anchors and anchors[0][0] > 0) or not anchors):
        entries.append((lines[0], "section", "PREAMBLE", "Preamble"))
    return _make_spans(entries, units)


def segment_document(
    fmt: Literal["legistar", "granicus"], units: list[tuple[int, str]]
) -> list[ItemSpan]:
    """Split ordered page or transcript units into non-overlapping items."""
    lines = _lines(units)
    if fmt == "legistar":
        return _segment_legistar(units, lines)
    return _segment_granicus(units, lines)


def run(conn: psycopg.Connection[TupleRow], document_id: str) -> dict[str, object]:
    """Replace a document's items."""
    row = conn.execute(
        "SELECT d.kind, d.meeting_id, c.item_format FROM documents d "
        "JOIN cities c ON c.id = d.city_id WHERE d.id = %s",
        (document_id,),
    ).fetchone()
    if row is None:
        return {"items": 0, "business": 0}
    kind, meeting_id, fmt = row
    if kind == "transcript":
        return {"items": 0, "business": 0}
    conn.execute("DELETE FROM items WHERE document_id = %s", (document_id,))
    units = conn.execute(
        "SELECT unit_index, text FROM units WHERE document_id = %s ORDER BY unit_index",
        (document_id,),
    ).fetchall()
    spans = segment_document(fmt, units)
    for span in spans:
        conn.execute(
            "INSERT INTO items (document_id, meeting_id, kind, identifier, title, ordinal, "
            "start_unit, start_offset, end_unit, end_offset) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (document_id, meeting_id, *astuple(span)),
        )
    return {"items": len(spans), "business": sum(span.kind == "business" for span in spans)}
