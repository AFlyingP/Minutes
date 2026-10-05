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


def run(conn: psycopg.Connection[TupleRow], document_id: str) -> dict[str, object]:
    """Replace the document's chunks with fixed windows over each unit."""
    conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
    units = conn.execute(
        "SELECT u.unit_index, u.text, d.city_id, m.body, m.meeting_date, d.kind "
        "FROM units u JOIN documents d ON d.id = u.document_id "
        "JOIN meetings m ON m.id = d.meeting_id "
        "WHERE u.document_id = %s ORDER BY u.unit_index",
        (document_id,),
    ).fetchall()
    ordinal = 0
    for unit_index, text, city_id, body, meeting_date, kind in units:
        for start, end in fixed_chunks(text):
            ordinal += 1
            piece = text[start:end]
            conn.execute(
                "INSERT INTO chunks (document_id, unit_index, chunker, ordinal, start_offset, "
                "end_offset, text, embed_text, city_id, body, meeting_date, doc_kind) "
                "VALUES (%s, %s, 'fixed', %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    document_id,
                    unit_index,
                    ordinal,
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
    return {"fixed": ordinal, "item": 0}
