import json
import random
from dataclasses import dataclass

import psycopg
from psycopg.rows import TupleRow

from minutes.config import CONFIG_DIR
from minutes.errors import CorpusError

Connection = psycopg.Connection[TupleRow]
COUNTS = {"agenda": 4, "minutes": 4, "transcript": 2}


@dataclass
class CityStats:
    documents: int = 0
    agenda: int = 0
    minutes: int = 0
    transcript: int = 0
    duplicate: int = 0
    skipped: int = 0
    failed: int = 0
    pages: int = 0
    segments: int = 0
    ocr_pages: int = 0
    bytes: int = 0


@dataclass(frozen=True)
class CorpusStats:
    per_city: dict[str, CityStats]
    total_pages: int


def make_dev_subset(conn: Connection, seed: int = 7) -> list[str]:
    rows = conn.execute(
        "SELECT id, city_id, kind FROM documents WHERE status = 'downloaded' "
        "AND city_id = ANY(%s) ORDER BY id",
        (["seattle", "lincoln"],),
    ).fetchall()
    available: dict[tuple[str, str], list[str]] = {}
    for document_id, city_id, kind in rows:
        available.setdefault((city_id, kind), []).append(document_id)

    selected: list[str] = []
    for city_id in ("seattle", "lincoln"):
        for kind, count in COUNTS.items():
            candidates = available.get((city_id, kind), [])
            if len(candidates) < count:
                raise CorpusError(f"{city_id} has fewer than {count} downloaded {kind} documents")
            selected.extend(random.Random(seed).sample(candidates, count))

    selected.sort()
    path = CONFIG_DIR / "dev_subset.json"
    path.write_text(
        json.dumps({"seed": seed, "documents": selected}, indent=2) + "\n",
        encoding="utf-8",
    )
    return selected


def corpus_stats(conn: Connection) -> CorpusStats:
    """Summarize downloaded documents, document statuses, and units per city."""
    documents = conn.execute(
        "SELECT city_id, kind, status, count(*), coalesce(sum(byte_size), 0) "
        "FROM documents GROUP BY city_id, kind, status ORDER BY city_id, kind, status"
    ).fetchall()
    per_city: dict[str, CityStats] = {}
    for city_id, kind, status, count, byte_size in documents:
        city = per_city.setdefault(city_id, CityStats())
        if status == "downloaded":
            city.documents += count
            city.bytes += byte_size
            if kind == "agenda":
                city.agenda += count
            elif kind == "minutes":
                city.minutes += count
            elif kind == "transcript":
                city.transcript += count
        elif status == "duplicate":
            city.duplicate += count
        elif status == "skipped":
            city.skipped += count
        elif status == "failed":
            city.failed += count

    units = conn.execute(
        "SELECT d.city_id, u.unit_kind, u.text_source, count(*) "
        "FROM units u JOIN documents d ON d.id = u.document_id "
        "GROUP BY d.city_id, u.unit_kind, u.text_source "
        "ORDER BY d.city_id, u.unit_kind, u.text_source"
    ).fetchall()
    for city_id, unit_kind, text_source, count in units:
        city = per_city[city_id]
        if unit_kind == "page":
            city.pages += count
        elif unit_kind == "segment":
            city.segments += count
        if text_source == "ocr":
            city.ocr_pages += count

    return CorpusStats(per_city=per_city, total_pages=sum(city.pages for city in per_city.values()))


def assert_minimums(stats: CorpusStats) -> None:
    """Require at least 200 downloaded documents per city and 3,000 pages."""
    for city_id in sorted(stats.per_city):
        documents = stats.per_city[city_id].documents
        if documents < 200:
            raise CorpusError(f"{city_id} has {documents} documents, minimum is 200")
    if stats.total_pages < 3000:
        raise CorpusError(f"corpus has {stats.total_pages} pages, minimum is 3000")
