from dataclasses import asdict, dataclass
from datetime import date
from typing import Literal

import psycopg
from psycopg.rows import TupleRow

from minutes.errors import ValidationError
from minutes.models import get_embedder, get_reranker

CANDIDATES = 50

# any query word may match; ts_rank_cd puts chunks that match more of them first
KEYWORD_QUERY = """
WITH q AS (
  SELECT NULLIF(replace(plainto_tsquery('english', %(query)s)::text, '&', '|'), '')::tsquery AS tsq
)
SELECT c.id, ts_rank_cd(c.tsv, q.tsq, 32) AS score
FROM chunks c, q
WHERE q.tsq IS NOT NULL AND c.tsv @@ q.tsq
  AND c.chunker = %(chunker)s
  AND (%(city)s::text IS NULL OR c.city_id = %(city)s)
  AND (%(date_from)s::date IS NULL OR c.meeting_date >= %(date_from)s)
  AND (%(date_to)s::date IS NULL OR c.meeting_date <= %(date_to)s)
  AND (%(body)s::text IS NULL OR c.body = %(body)s)
  AND (%(kind)s::text IS NULL OR c.doc_kind = %(kind)s)
ORDER BY score DESC, c.id ASC
LIMIT %(limit)s
"""

VECTOR_QUERY = """
SELECT c.id, 1 - (c.embedding <=> %(qvec)s::vector) AS score
FROM chunks c
WHERE c.embedding IS NOT NULL
  AND c.chunker = %(chunker)s
  AND (%(city)s::text IS NULL OR c.city_id = %(city)s)
  AND (%(date_from)s::date IS NULL OR c.meeting_date >= %(date_from)s)
  AND (%(date_to)s::date IS NULL OR c.meeting_date <= %(date_to)s)
  AND (%(body)s::text IS NULL OR c.body = %(body)s)
  AND (%(kind)s::text IS NULL OR c.doc_kind = %(kind)s)
ORDER BY c.embedding <=> %(qvec)s::vector ASC, c.id ASC
LIMIT %(limit)s
"""

CHUNK_ROWS = """
SELECT c.id, c.document_id, c.unit_index, u.unit_kind, c.start_offset, c.end_offset,
       c.text, c.item_id, c.city_id, c.body, c.meeting_date, c.doc_kind, u.start_ms,
       i.identifier, i.title
FROM chunks c
LEFT JOIN items i ON i.id = c.item_id
JOIN units u ON u.document_id = c.document_id AND u.unit_index = c.unit_index
WHERE c.id = ANY(%s)
"""


def rrf(lists: list[list[int]], c: int = 60) -> list[tuple[int, float]]:
    """Fuse ranked lists by reciprocal rank, breaking ties by ID."""
    scores: dict[int, float] = {}
    for ids in lists:
        for rank, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (c + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


@dataclass(frozen=True)
class Filters:
    city: str | None = None
    date_from: date | None = None
    date_to: date | None = None
    body: str | None = None
    kind: str | None = None


@dataclass(frozen=True)
class Span:
    chunk_id: int
    document_id: str
    unit_index: int
    unit_kind: str
    start_offset: int
    end_offset: int
    text: str


@dataclass(frozen=True)
class Hit:
    rank: int
    score: float
    chunk_id: int
    item_id: int | None
    document_id: str
    city_id: str
    body: str
    meeting_date: date
    doc_kind: str
    item_identifier: str | None
    item_title: str | None
    start_ms: int | None
    spans: list[Span]


def span_label(unit_kind: str, unit_index: int, start_ms: int | None) -> str:
    """How a span is shown to a reader: a page number, or the time into the recording."""
    if unit_kind == "page" or start_ms is None:
        return f"p. {unit_index}"
    seconds = start_ms // 1000
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def hit_heading(city_name: str, hit: Hit) -> str:
    heading = f"{city_name} {hit.body}, {hit.meeting_date.isoformat()}, {hit.doc_kind}"
    if hit.item_identifier:
        heading += f", item {hit.item_identifier}"
    return heading


def city_names(conn: psycopg.Connection[TupleRow]) -> dict[str, str]:
    return dict(conn.execute("SELECT id, name FROM cities").fetchall())


def search(
    conn: psycopg.Connection[TupleRow],
    query: str,
    filters: Filters,
    *,
    mode: Literal["keyword", "vector", "hybrid", "hybrid_rerank"],
    chunker: Literal["fixed", "item"],
    k: int,
) -> list[Hit]:
    """Ranked hits for a query; each hit carries the text spans a citation can point at."""
    query = query.strip()
    if not 1 <= len(query) <= 500:
        raise ValidationError("q: must be 1 to 500 characters")
    if not 1 <= k <= 50:
        raise ValidationError("k: must be between 1 and 50")
    if mode not in ("keyword", "vector", "hybrid", "hybrid_rerank"):
        raise ValidationError(f"mode {mode} is not available")
    if chunker not in ("fixed", "item"):
        raise ValidationError(f"chunker {chunker} is not available")
    params = {**asdict(filters), "query": query, "chunker": chunker, "limit": CANDIDATES}
    keyword = []
    vector = []
    if mode != "vector":
        keyword = conn.execute(KEYWORD_QUERY, params).fetchall()
    if mode != "keyword":
        conn.execute("SET LOCAL hnsw.ef_search = 100")
        conn.execute("SET LOCAL hnsw.iterative_scan = strict_order")
        qvec = get_embedder().embed_query(query)
        vector = conn.execute(VECTOR_QUERY, {**params, "qvec": qvec}).fetchall()
    if mode == "keyword":
        ranked = [(row[0], float(row[1])) for row in keyword]
    elif mode == "vector":
        ranked = [(row[0], float(row[1])) for row in vector]
    else:
        ranked = rrf([[row[0] for row in keyword], [row[0] for row in vector]])
        if mode == "hybrid_rerank":
            ranked = ranked[:CANDIDATES]
    if not ranked:
        return []
    rows = conn.execute(CHUNK_ROWS, ([chunk_id for chunk_id, _ in ranked],)).fetchall()
    chunks = {row[0]: row for row in rows}
    if mode == "hybrid_rerank":
        scores = get_reranker().score(query, [chunks[chunk_id][6] for chunk_id, _ in ranked])
        ranked = sorted(
            [(candidate[0], score) for candidate, score in zip(ranked, scores, strict=True)],
            key=lambda pair: (-pair[1], pair[0]),
        )
    hits: list[Hit] = []
    for chunk_id, score in ranked:
        row = chunks[chunk_id]
        item_id, city_id, body, meeting_date, doc_kind, start_ms = row[7:13]
        identifier, title = row[13:]
        spans = [Span(*row[:7])]
        hits.append(
            Hit(
                rank=len(hits) + 1,
                score=float(score),
                chunk_id=chunk_id,
                item_id=item_id,
                document_id=row[1],
                city_id=city_id,
                body=body,
                meeting_date=meeting_date,
                doc_kind=doc_kind,
                item_identifier=identifier,
                item_title=title,
                start_ms=start_ms,
                spans=spans,
            )
        )
        if len(hits) == k:
            break
    return hits
