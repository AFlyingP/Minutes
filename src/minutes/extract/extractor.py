import psycopg
from psycopg.rows import TupleRow
from psycopg.types.json import Jsonb

from minutes.extract.schema import EXTRACTION_SCHEMA, FACT_MODELS
from minutes.llm import get_client
from minutes.llm.prompts import EXTRACT_SYSTEM, ItemText, extract_user

MIN_ITEM_CHARS = 40
MAX_QUOTE_CHARS = 600


def item_texts(conn: psycopg.Connection[TupleRow], document_id: str) -> list[ItemText]:
    """The document's items that are long enough to hold a fact, numbered from 1."""
    units: dict[int, str] = dict(
        conn.execute(
            "SELECT unit_index, text FROM units WHERE document_id = %s", (document_id,)
        ).fetchall()
    )
    items = conn.execute(
        "SELECT id, identifier, title, start_unit, start_offset, end_unit, end_offset "
        "FROM items WHERE document_id = %s ORDER BY ordinal",
        (document_id,),
    ).fetchall()
    texts: list[ItemText] = []
    for item_id, identifier, title, start_unit, start_offset, end_unit, end_offset in items:
        parts = []
        for index in range(start_unit, end_unit + 1):
            first = start_offset if index == start_unit else 0
            last = end_offset if index == end_unit else None
            parts.append(units[index][first:last])
        text = "\n".join(parts)
        if len(text) >= MIN_ITEM_CHARS:
            texts.append(ItemText(len(texts) + 1, item_id, identifier, title, text, start_unit))
    return texts


def run(conn: psycopg.Connection[TupleRow], document_id: str) -> dict[str, object]:
    """Replace the document's facts with what the model reads from its items; not yet verified."""
    conn.execute("DELETE FROM facts WHERE document_id = %s", (document_id,))
    row = conn.execute(
        "SELECT d.kind, d.meeting_id, d.city_id, c.name, m.body, m.meeting_date "
        "FROM documents d JOIN meetings m ON m.id = d.meeting_id "
        "JOIN cities c ON c.id = d.city_id WHERE d.id = %s",
        (document_id,),
    ).fetchone()
    if row is None or row[0] == "transcript":
        return {"facts": 0, "rejected": 0}
    kind, meeting_id, city_id, city_name, body, meeting_date = row
    items = item_texts(conn, document_id)
    if not items:
        return {"facts": 0, "rejected": 0}
    user = extract_user(city_name, body, meeting_date.isoformat(), kind, items)
    result = get_client().chat(
        purpose="extract",
        messages=[
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": user},
        ],
        schema=EXTRACTION_SCHEMA,
        schema_name="extraction",
    )
    assert result.parsed is not None
    by_number = {item.number: item for item in items}
    records = result.parsed["records"]
    assert isinstance(records, list)
    for record in records:
        item = by_number[record["item"]]
        data = FACT_MODELS[record["kind"]].model_validate(record[record["kind"]]).model_dump()
        conn.execute(
            "INSERT INTO facts (kind, document_id, meeting_id, city_id, body, meeting_date, "
            "item_id, unit_index, quote, data, verified, verify_error, model) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, false, 'unverified', %s)",
            (
                record["kind"],
                document_id,
                meeting_id,
                city_id,
                body,
                meeting_date,
                item.item_id,
                item.start_unit,
                record["quote"][:MAX_QUOTE_CHARS],
                Jsonb(data),
                result.model,
            ),
        )
    return {"facts": len(records), "rejected": 0}
