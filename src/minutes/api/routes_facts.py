from datetime import date
from typing import Annotated

from fastapi import APIRouter, Query
from psycopg.rows import dict_row

from minutes.api.routes_core import Conn, checked_filters
from minutes.api.schemas import FactCollection, FactRowOut, FactsOut
from minutes.errors import NotFoundError, ValidationError
from minutes.retrieval.search import span_label

router = APIRouter(prefix="/api")

FACT_KINDS: dict[str, tuple[FactCollection, str]] = {
    "votes": ("votes", "vote"),
    "motions": ("motions", "motion"),
    "ordinances": ("ordinances", "ordinance"),
    "amounts": ("amounts", "amount"),
    "statements": ("statements", "statement"),
}


@router.get("/facts/{kind}")
def facts(
    conn: Conn,
    kind: str,
    city: str | None = None,
    body: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    q: str | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> FactsOut:
    selected_kind = FACT_KINDS.get(kind)
    if selected_kind is None:
        raise NotFoundError(f"unknown fact kind {kind}")
    collection, fact_kind = selected_kind
    filters = checked_filters(conn, city, body, date_from, date_to, None)
    query = q.strip() if q is not None else None
    if query is not None and not 1 <= len(query) <= 100:
        raise ValidationError("q: must be 1 to 100 characters")

    where = ["f.kind = %s", "f.verified"]
    params: list[str | date | int] = [fact_kind]
    if filters.city is not None:
        where.append("f.city_id = %s")
        params.append(filters.city)
    if filters.body is not None:
        where.append("f.body = %s")
        params.append(filters.body)
    if filters.date_from is not None:
        where.append("f.meeting_date >= %s")
        params.append(filters.date_from)
    if filters.date_to is not None:
        where.append("f.meeting_date <= %s")
        params.append(filters.date_to)
    if query is not None:
        where.append("strpos(lower(f.quote), lower(%s)) > 0")
        params.append(query)

    clause = " AND ".join(where)
    source = (
        " FROM facts f JOIN documents d ON d.id = f.document_id "
        "JOIN units u ON u.document_id = f.document_id AND u.unit_index = f.unit_index"
    )
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(f"SELECT count(*) AS total{source} WHERE {clause}", params)
        count = cur.fetchone()
        assert count is not None
        cur.execute(
            "SELECT f.id, f.document_id, f.meeting_id, f.city_id, f.body, f.meeting_date, "
            "d.kind AS doc_kind, f.unit_index, u.unit_kind, f.quote_start, f.quote_end, "
            "f.quote, u.start_ms, f.data"
            f"{source} WHERE {clause} "
            'ORDER BY f.meeting_date DESC, f.document_id COLLATE "C" ASC, '
            "f.unit_index ASC, f.id ASC LIMIT %s OFFSET %s",
            [*params, limit, offset],
        )
        rows: list[FactRowOut] = []
        for row in cur.fetchall():
            start_ms = row.pop("start_ms")
            row["label"] = span_label(row["unit_kind"], row["unit_index"], start_ms)
            rows.append(FactRowOut(**row))

    return FactsOut(kind=collection, total=count["total"], rows=rows)
