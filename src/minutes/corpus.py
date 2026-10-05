import json
import random

import psycopg
from psycopg.rows import TupleRow

from minutes.config import CONFIG_DIR
from minutes.errors import CorpusError

Connection = psycopg.Connection[TupleRow]
COUNTS = {"agenda": 4, "minutes": 4, "transcript": 2}


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
