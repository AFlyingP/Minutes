from datetime import date

import psycopg
from psycopg.rows import TupleRow

WINDOW = 1200
STEP = 1000


def fixed_chunks(unit_text: str) -> list[tuple[int, int]]:
    """(start, end) windows of 1,200 characters that overlap by 200."""
    if not unit_text.strip():
        return []
    windows = []
    start = 0
    while True:
        end = min(start + WINDOW, len(unit_text))
        windows.append((start, end))
        if end == len(unit_text):
            return windows
        start += STEP


def item_chunks(lines: list[tuple[int, int]], unit_text: str) -> list[tuple[int, int]]:
    chunks: list[tuple[int, int]] = []
    chunk_start: int | None = None
    chunk_end = 0

    def add(start: int, end: int) -> None:
        if unit_text[start:end].strip():
            chunks.append((start, end))

    for start, end in lines:
        if end - start > WINDOW:
            if chunk_start is not None:
                add(chunk_start, chunk_end)
                chunk_start = None
            for part_start in range(start, end, WINDOW):
                add(part_start, min(part_start + WINDOW, end))
            continue

        if chunk_start is None:
            chunk_start, chunk_end = start, end
        elif end - chunk_start <= WINDOW:
            chunk_end = end
        else:
            add(chunk_start, chunk_end)
            chunk_start, chunk_end = start, end

    if chunk_start is not None:
        add(chunk_start, chunk_end)
    return chunks


def header_for(
    city_name: str,
    body: str,
    meeting_date: date,
    kind: str,
    identifier: str,
    title: str,
) -> str:
    value = f"{city_name} {body} {meeting_date.isoformat()} {kind} | {identifier} {title}"
    return value[:200]


def _line_ranges(start: int, end: int, text: str) -> list[tuple[int, int]]:
    ranges = []
    line_start = start
    while line_start < end:
        newline = text.find("\n", line_start, end)
        if newline == -1:
            ranges.append((line_start, end))
            break
        ranges.append((line_start, newline))
        line_start = newline + 1
    return ranges


def _time_title(start_ms: int) -> str:
    seconds = start_ms // 1000
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def run(conn: psycopg.Connection[TupleRow], document_id: str) -> dict[str, object]:
    """Replace the document's fixed and item chunks."""
    conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
    document = conn.execute(
        "SELECT d.kind, c.name, m.body, m.meeting_date, d.city_id "
        "FROM documents d JOIN cities c ON c.id = d.city_id "
        "JOIN meetings m ON m.id = d.meeting_id WHERE d.id = %s",
        (document_id,),
    ).fetchone()
    assert document is not None
    kind, city_name, body, meeting_date, city_id = document
    units = conn.execute(
        "SELECT unit_index, text, start_ms FROM units WHERE document_id = %s ORDER BY unit_index",
        (document_id,),
    ).fetchall()

    fixed_ordinal = 0
    for unit_index, text, _ in units:
        for start, end in fixed_chunks(text):
            fixed_ordinal += 1
            piece = text[start:end]
            conn.execute(
                "INSERT INTO chunks (document_id, unit_index, chunker, ordinal, start_offset, "
                "end_offset, text, embed_text, city_id, body, meeting_date, doc_kind) "
                "VALUES (%s, %s, 'fixed', %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    document_id,
                    unit_index,
                    fixed_ordinal,
                    start,
                    end,
                    piece,
                    piece,
                    city_id,
                    body,
                    meeting_date,
                    kind,
                ),
            )

    item_ordinal = 0
    if kind in ("agenda", "minutes"):
        items = conn.execute(
            "SELECT id, identifier, title, start_unit, start_offset, end_unit, end_offset "
            "FROM items WHERE document_id = %s ORDER BY ordinal",
            (document_id,),
        ).fetchall()
        for item_id, identifier, title, start_unit, start_offset, end_unit, end_offset in items:
            for unit_index, text, _ in units:
                if not start_unit <= unit_index <= end_unit:
                    continue
                start = start_offset if unit_index == start_unit else 0
                end = end_offset if unit_index == end_unit else len(text)
                ranges = item_chunks(_line_ranges(start, end, text), text)
                header = header_for(city_name, body, meeting_date, kind, identifier, title)
                for chunk_start, chunk_end in ranges:
                    item_ordinal += 1
                    piece = text[chunk_start:chunk_end]
                    conn.execute(
                        "INSERT INTO chunks (document_id, unit_index, item_id, chunker, ordinal, "
                        "start_offset, end_offset, text, embed_text, city_id, body, meeting_date, "
                        "doc_kind) VALUES (%s, %s, %s, 'item', %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        (
                            document_id,
                            unit_index,
                            item_id,
                            item_ordinal,
                            chunk_start,
                            chunk_end,
                            piece,
                            f"{header}\n{piece}",
                            city_id,
                            body,
                            meeting_date,
                            kind,
                        ),
                    )
    else:
        for unit_index, text, start_ms in units:
            header = header_for(city_name, body, meeting_date, kind, "", _time_title(start_ms))
            item_ordinal += 1
            conn.execute(
                "INSERT INTO chunks (document_id, unit_index, chunker, ordinal, start_offset, "
                "end_offset, text, embed_text, city_id, body, meeting_date, doc_kind) "
                "VALUES (%s, %s, 'item', %s, 0, %s, %s, %s, %s, %s, %s, %s)",
                (
                    document_id,
                    unit_index,
                    item_ordinal,
                    len(text),
                    text,
                    f"{header}\n{text}",
                    city_id,
                    body,
                    meeting_date,
                    kind,
                ),
            )
    return {"fixed": fixed_ordinal, "item": item_ordinal}
