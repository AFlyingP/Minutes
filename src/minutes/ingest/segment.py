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


def run(conn: psycopg.Connection[TupleRow], document_id: str) -> dict[str, object]:
    """Replace the document's items; for now an agenda or minutes is one item covering it all."""
    row = conn.execute(
        "SELECT kind, meeting_id FROM documents WHERE id = %s", (document_id,)
    ).fetchone()
    if row is None or row[0] == "transcript":
        return {"items": 0}
    conn.execute("DELETE FROM items WHERE document_id = %s", (document_id,))
    units = conn.execute(
        "SELECT unit_index, text FROM units WHERE document_id = %s ORDER BY unit_index",
        (document_id,),
    ).fetchall()
    if not units:
        return {"items": 0, "business": 0}
    first, last = units[0], units[-1]
    span = ItemSpan("section", "PREAMBLE", "Preamble", 1, first[0], 0, last[0], len(last[1]))
    conn.execute(
        "INSERT INTO items (document_id, meeting_id, kind, identifier, title, ordinal, "
        "start_unit, start_offset, end_unit, end_offset) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (document_id, row[1], *astuple(span)),
    )
    return {"items": 1, "business": 0}
