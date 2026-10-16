from dataclasses import dataclass

import psycopg
from psycopg.rows import TupleRow
from psycopg.types.json import Jsonb
from pydantic import ValidationError as PydanticValidationError

from minutes import llm
from minutes.config import load_models
from minutes.errors import LLMOutputError, LLMTransportError, StageError
from minutes.extract.schema import EXTRACTION_SCHEMA, FACT_MODELS
from minutes.extract.verify import verify
from minutes.llm.prompts import EXTRACT_SYSTEM, ItemText, extract_user
from minutes.log import get_logger

MIN_ITEM_CHARS = 40
MAX_ITEM_CHARS = 24_000
MAX_QUOTE_CHARS = 600
_LOG = get_logger("extract")


def get_client() -> llm.LLMClient:
    """Get the current LLM client, allowing callers to replace the factory."""
    return llm.get_client()


def extract_model() -> str:
    """The model ID configured for the extract purpose."""
    models = load_models()
    return models.cheap if models.purposes["extract"].model == "cheap" else models.strong


@dataclass(frozen=True)
class RawFact:
    item_id: int | None
    kind: str
    quote: str
    data: dict[str, object]
    error: str | None = None


def batches(items: list[ItemText], controls: bool) -> list[list[ItemText]]:
    """Truncate long items and pack consecutive items under the prompt size limit."""
    prepared = []
    for item in items:
        if len(item.text) > MAX_ITEM_CHARS:
            _LOG.warning(
                "item truncated",
                extra={
                    "event": "item_truncated",
                    "fields": {"item": item.number, "characters": len(item.text)},
                },
            )
            item = item._replace(text=item.text[:MAX_ITEM_CHARS])
        prepared.append(item)

    if not controls:
        return [[item] for item in prepared]

    result: list[list[ItemText]] = []
    current: list[ItemText] = []
    size = 0
    for item in prepared:
        if current and size + len(item.text) > MAX_ITEM_CHARS:
            result.append(current)
            current = []
            size = 0
        current.append(item)
        size += len(item.text)
    if current:
        result.append(current)
    return result


def extract_items(
    items: list[ItemText],
    *,
    controls: bool,
    sample_index: int = 0,
    city: str = "",
    body: str = "",
    meeting_date: str = "",
    document_kind: str = "",
) -> list[RawFact]:
    """Extract schema checked records from the document's items."""
    facts: list[RawFact] = []
    groups = batches(items, controls)
    if not groups:
        return facts
    client = get_client()
    for batch in groups:
        numbered = [item._replace(number=index) for index, item in enumerate(batch, start=1)]
        result = client.chat(
            purpose="extract",
            messages=[
                {"role": "system", "content": EXTRACT_SYSTEM},
                {
                    "role": "user",
                    "content": extract_user(city, body, meeting_date, document_kind, numbered),
                },
            ],
            schema=EXTRACTION_SCHEMA,
            schema_name="extraction",
            model=extract_model(),
            sample_index=sample_index,
        )
        assert result.parsed is not None
        records = result.parsed["records"]
        assert isinstance(records, list)
        by_number = {index: item for index, item in enumerate(batch, start=1)}
        for record in records:
            kind = record["kind"]
            item = by_number.get(record["item"])
            payload = record[kind]
            data: dict[str, object] = {}
            error = None
            if payload is not None:
                try:
                    data = FACT_MODELS[kind].model_validate(payload).model_dump()
                except PydanticValidationError:
                    error = "invalid_record"
            quote = record["quote"].strip()[:MAX_QUOTE_CHARS]
            item_id = item.item_id if item is not None else None
            facts.append(RawFact(item_id, kind, quote, data, error))
    return facts


def item_texts(conn: psycopg.Connection[TupleRow], document_id: str) -> list[ItemText]:
    """Return the document's long enough items in ordinal order, numbered from 1."""
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


def quote_unit_offsets(
    item: ItemText,
    item_start: int,
    item_end: int,
    bounds: tuple[int, int, int, int],
    units: dict[int, str],
) -> tuple[int, int, int]:
    """Map item offsets to the source unit containing the quote."""
    start_unit, start_offset, end_unit, end_offset = bounds
    cursor = 0
    for unit_index in range(start_unit, end_unit + 1):
        text = units[unit_index]
        segment_start = start_offset if unit_index == start_unit else 0
        segment_end = end_offset if unit_index == end_unit else len(text)
        length = segment_end - segment_start
        if cursor <= item_start < cursor + length:
            quote_start = segment_start + item_start - cursor
            quote_end = min(segment_start + item_end - cursor, len(text))
            return unit_index, quote_start, quote_end
        cursor += length
        if unit_index < end_unit:
            cursor += 1
    raise ValueError(f"quote offset is outside item {item.item_id}")


def _insert_transcript_facts(
    conn: psycopg.Connection[TupleRow],
    document_id: str,
    meeting_id: str,
    city_id: str,
    body: str,
    meeting_date: object,
) -> int:
    rows = conn.execute(
        "SELECT unit_index, text, speaker FROM units WHERE document_id = %s "
        "AND speaker IS NOT NULL ORDER BY unit_index",
        (document_id,),
    ).fetchall()
    for unit_index, quote, speaker in rows:
        prefix = f"{speaker}:"
        summary = quote[len(prefix) :].lstrip(" ") if quote.startswith(prefix) else quote
        data = (
            FACT_MODELS["statement"]
            .model_validate({"speaker": speaker, "role": None, "summary": summary[:300]})
            .model_dump()
        )
        conn.execute(
            "INSERT INTO facts (kind, document_id, meeting_id, city_id, body, meeting_date, "
            "item_id, unit_index, quote_start, quote_end, quote, data, verified, "
            "verify_error, model) "
            "VALUES ('statement', %s, %s, %s, %s, %s, NULL, %s, 0, %s, %s, %s, true, NULL, NULL)",
            (
                document_id,
                meeting_id,
                city_id,
                body,
                meeting_date,
                unit_index,
                len(quote),
                quote,
                Jsonb(data),
            ),
        )
    return len(rows)


def run(
    conn: psycopg.Connection[TupleRow],
    document_id: str,
    *,
    controls: bool = True,
    sample_index: int = 0,
) -> dict[str, object]:
    """Replace a document's facts with extracted, verified, and located records."""
    conn.execute("DELETE FROM facts WHERE document_id = %s", (document_id,))
    row = conn.execute(
        "SELECT d.kind, d.meeting_id, d.city_id, c.name, m.body, m.meeting_date "
        "FROM documents d JOIN meetings m ON m.id = d.meeting_id "
        "JOIN cities c ON c.id = d.city_id WHERE d.id = %s",
        (document_id,),
    ).fetchone()
    if row is None:
        return {"facts": 0, "rejected": 0}
    kind, meeting_id, city_id, city_name, body, meeting_date = row
    if kind == "transcript":
        count = _insert_transcript_facts(conn, document_id, meeting_id, city_id, body, meeting_date)
        return {"facts": count, "rejected": 0}

    items = item_texts(conn, document_id)
    if not items:
        return {"facts": 0, "rejected": 0}
    try:
        raw_facts = extract_items(
            items,
            controls=controls,
            sample_index=sample_index,
            city=city_name,
            body=body,
            meeting_date=meeting_date.isoformat(),
            document_kind=kind,
        )
    except (LLMOutputError, LLMTransportError) as err:
        raise StageError(f"extract failed: {err}") from err

    units: dict[int, str] = dict(
        conn.execute(
            "SELECT unit_index, text FROM units WHERE document_id = %s", (document_id,)
        ).fetchall()
    )
    item_bounds = {
        item_id: (start_unit, start_offset, end_unit, end_offset)
        for item_id, start_unit, start_offset, end_unit, end_offset in conn.execute(
            "SELECT id, start_unit, start_offset, end_unit, end_offset FROM items "
            "WHERE document_id = %s",
            (document_id,),
        ).fetchall()
    }
    by_id = {item.item_id: item for item in items}
    fallback_unit = min(units) if units else 0
    rejected = 0
    for fact in raw_facts:
        item = by_id.get(fact.item_id) if fact.item_id is not None else None
        verify_error: str | None
        if item is None or not fact.data:
            verified = False
            verify_error = fact.error or "kind_mismatch"
            if item is None:
                unit_index, quote_start, quote_end = fallback_unit, None, None
            else:
                unit_index, quote_start, quote_end = item.start_unit, None, None
        else:
            verify_result = verify(fact.kind, fact.data | {"quote": fact.quote}, item.text)
            verified = verify_result.ok
            verify_error = verify_result.error
            if verify_result.quote_start is None or verify_result.quote_end is None:
                unit_index, quote_start, quote_end = item.start_unit, None, None
            else:
                unit_index, quote_start, quote_end = quote_unit_offsets(
                    item,
                    verify_result.quote_start,
                    verify_result.quote_end,
                    item_bounds[item.item_id],
                    units,
                )
        conn.execute(
            "INSERT INTO facts (kind, document_id, meeting_id, city_id, body, meeting_date, "
            "item_id, unit_index, quote_start, quote_end, quote, data, verified, "
            "verify_error, model) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                fact.kind,
                document_id,
                meeting_id,
                city_id,
                body,
                meeting_date,
                fact.item_id,
                unit_index,
                quote_start,
                quote_end,
                fact.quote,
                Jsonb(fact.data),
                verified,
                verify_error,
                extract_model(),
            ),
        )
        rejected += not verified
    return {"facts": len(raw_facts), "rejected": rejected}
