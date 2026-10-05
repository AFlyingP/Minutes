import json
import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import TupleRow
from psycopg.types.json import Jsonb
from pydantic import ValidationError as PayloadError

from minutes.config import get_settings, load_cities
from minutes.errors import (
    CorpusError,
    LabelsFrozenError,
    LabelsNotFrozenError,
    LabelValidationError,
    NotFoundError,
    ValidationError,
)
from minutes.extract.verify import verify
from minutes.labels.schema import (
    AgendaCount,
    AgentTask,
    ExtractionCase,
    Label,
    LabelIn,
    Passage,
    QuestionAnswerable,
    QuestionUnanswerable,
    parse_payload,
)

_COLUMNS = "id, label_type, city_id, payload, author, created_utc, human_reviewed, note"
_PREFIXES = {
    "question": ("q", 3),
    "agenda_count": ("c", 2),
    "extraction": ("x", 3),
    "agent_task": ("a", 2),
}


@dataclass(frozen=True)
class TextSearchResult:
    unit_count: int
    matches: list[tuple[str, int, str]]


@dataclass(frozen=True)
class _Source:
    meeting_id: str
    kind: str
    unit_text: str | None


def is_frozen(labels_dir: Path = Path("eval/labels")) -> bool:
    manifest = labels_dir / "manifest.json"
    if not manifest.exists():
        return False
    return json.loads(manifest.read_text(encoding="utf-8")).get("frozen") is True


def require_frozen(labels_dir: Path = Path("eval/labels")) -> None:
    if not is_frozen(labels_dir):
        raise LabelsNotFrozenError("labels are not frozen")


def require_unfrozen(labels_dir: Path = Path("eval/labels")) -> None:
    if is_frozen(labels_dir):
        raise LabelsFrozenError("labels are frozen")


def _passage_source(
    conn: psycopg.Connection[TupleRow], p: Passage, city: str | None
) -> _Source | None:
    row = conn.execute(
        "SELECT d.city_id, d.meeting_id, d.kind, u.text FROM documents d "
        "LEFT JOIN units u ON u.document_id = d.id AND u.unit_index = %s "
        "WHERE d.id = %s AND d.status = 'downloaded'",
        (p.unit_index, p.document_id),
    ).fetchone()
    if row is None or (city is not None and row[0] != city):
        return None
    return _Source(row[1], row[2], row[3])


def _passage_messages(passages: list[Passage], sources: list[_Source | None]) -> list[str]:
    pairs = list(zip(passages, sources, strict=True))
    messages = []
    for number, (p, source) in enumerate(pairs, 1):
        if source is None:
            messages.append(f"passage {number}: unknown document {p.document_id}")
    for number, (p, source) in enumerate(pairs, 1):
        if source is not None and (
            source.unit_text is None or not 0 <= p.start < p.end <= len(source.unit_text)
        ):
            messages.append(f"passage {number}: offsets out of range")
    for number, (p, source) in enumerate(pairs, 1):
        if (
            source is not None
            and source.unit_text is not None
            and 0 <= p.start < p.end <= len(source.unit_text)
            and source.unit_text[p.start : p.end] != p.text
        ):
            messages.append(f"passage {number}: text does not match stored unit text")
    for number, (p, _) in enumerate(pairs, 1):
        if not 10 <= p.end - p.start <= 2000:
            messages.append(f"passage {number}: length must be 10 to 2000 characters")
    return messages


def validate_passage(conn: psycopg.Connection[TupleRow], p: Passage) -> None:
    source = _passage_source(conn, p, None)
    messages = _passage_messages([p], [source])
    if messages:
        raise LabelValidationError(messages)


def text_search(conn: psycopg.Connection[TupleRow], city: str, q: str) -> TextSearchResult:
    if not 2 <= len(q) <= 100:
        raise ValidationError("query must be 2 to 100 characters")
    rows = conn.execute(
        "SELECT u.document_id, u.unit_index, u.text, "
        "position(lower(%s) in lower(u.text)), count(*) OVER () "
        "FROM units u JOIN documents d ON d.id = u.document_id "
        "WHERE d.city_id = %s AND d.status = 'downloaded' "
        "AND position(lower(%s) in lower(u.text)) > 0 "
        "ORDER BY u.document_id, u.unit_index LIMIT 20",
        (q, city, q),
    ).fetchall()
    matches = [
        (document, index, text[max(0, position - 1 - 80) : position - 1 + len(q) + 80])
        for document, index, text, position, _ in rows
    ]
    return TextSearchResult(rows[0][4] if rows else 0, matches)


def sample_meetings(conn: psycopg.Connection[TupleRow], city: str) -> list[str]:
    ids = [
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT m.id FROM meetings m JOIN documents d ON d.meeting_id = m.id "
            "WHERE m.city_id = %s AND d.kind = 'agenda' AND d.status = 'downloaded'",
            (city,),
        ).fetchall()
    ]
    if len(ids) < 10:
        raise CorpusError(f"city {city} needs at least 10 downloaded agendas")
    return random.Random(load_cities()[city].sample_seed).sample(sorted(ids), 10)


def _answer_valid(answer: str, answer_type: str) -> bool:
    if answer_type == "number":
        return re.fullmatch(r"-?\d+(\.\d+)?", answer) is not None
    if answer_type == "date":
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", answer):
            return False
        try:
            date.fromisoformat(answer)
        except ValueError:
            return False
        return True
    return 1 <= len(answer) <= 300


def _agenda_messages(
    conn: psycopg.Connection[TupleRow], city: str, payload: AgendaCount, label_id: str | None
) -> list[str]:
    messages = []
    row = conn.execute(
        "SELECT m.city_id, d.unit_count FROM meetings m "
        "LEFT JOIN documents d ON d.meeting_id = m.id "
        "AND d.kind = 'agenda' AND d.status = 'downloaded' WHERE m.id = %s",
        (payload.meeting_id,),
    ).fetchone()
    if row is None or row[0] != city:
        messages.append("unknown meeting")
    else:
        if payload.meeting_id not in sample_meetings(conn, city):
            messages.append("meeting is not sampled")
        if row[1] is None:
            messages.append("meeting has no downloaded agenda")
        elif any(not 1 <= item.start_page <= row[1] for item in payload.items):
            messages.append("start_page out of range")
    if payload.count != len(payload.items):
        messages.append("count does not match items")
    if len({item.identifier for item in payload.items}) != len(payload.items):
        messages.append("identifiers must be distinct")
    if conn.execute(
        "SELECT 1 FROM labels WHERE label_type = 'agenda_count' "
        "AND payload->>'meeting_id' = %s AND (%s::text IS NULL OR id != %s)",
        (payload.meeting_id, label_id, label_id),
    ).fetchone():
        messages.append("meeting already has a label")
    return [f"agenda_count: {message}" for message in messages]


def _validate(
    conn: psycopg.Connection[TupleRow], label: LabelIn, label_id: str | None = None
) -> None:
    messages = []
    if get_settings().corpus != "fixture":
        row = conn.execute(
            "SELECT EXISTS (SELECT 1 FROM items) OR EXISTS (SELECT 1 FROM chunks) "
            "OR EXISTS (SELECT 1 FROM facts)"
        ).fetchone()
        assert row is not None
        if row[0]:
            messages.append("derived outputs exist; labels must be written from source pages only")
    payload = None
    try:
        payload = parse_payload(label.type, label.payload)
    except PayloadError as err:
        messages.extend(
            f"payload: {'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in err.errors()
        )
    city_exists = conn.execute("SELECT 1 FROM cities WHERE id = %s", (label.city,)).fetchone()
    if not city_exists:
        messages.append(f"unknown city {label.city}")
    passages = []
    if isinstance(payload, QuestionAnswerable):
        passages = payload.passages
    elif isinstance(payload, ExtractionCase):
        passages = [payload.passage]
    sources = [_passage_source(conn, passage, label.city) for passage in passages]
    messages.extend(_passage_messages(passages, sources))
    if isinstance(payload, QuestionAnswerable):
        tokens = re.findall(r"\w+", payload.question.lower())
        runs = {tuple(tokens[i : i + 6]) for i in range(len(tokens) - 5)}
        for number, passage in enumerate(passages, 1):
            words = re.findall(r"\w+", passage.text.lower())
            if any(tuple(words[i : i + 6]) in runs for i in range(len(words) - 5)):
                messages.append(
                    f"question copies more than 5 consecutive words from passage {number}"
                )
        if payload.question_type == "multi_passage" and (
            len({(p.document_id, p.unit_index) for p in passages}) < 2
            or any(source is None for source in sources)
            or len({source.meeting_id for source in sources if source is not None}) != 1
        ):
            messages.append("multi_passage needs passages on two different pages of one meeting")
    if isinstance(payload, (QuestionAnswerable, AgentTask)) and not _answer_valid(
        payload.answer, payload.answer_type
    ):
        messages.append(f"answer does not match answer_type {payload.answer_type}")
    if isinstance(payload, QuestionUnanswerable):
        queries = [search.query.lower() for search in payload.absence_searches]
        if len(set(queries)) != len(queries):
            messages.append("absence searches must have distinct lowercased queries")
        for number, search in enumerate(payload.absence_searches, 1):
            actual = text_search(conn, label.city, search.query).unit_count
            if search.hits != actual:
                messages.append(
                    f"absence search {number}: hits is {search.hits}, stored text has {actual}"
                )
    if isinstance(payload, AgendaCount) and city_exists:
        messages.extend(_agenda_messages(conn, label.city, payload, label_id))
    if isinstance(payload, ExtractionCase):
        source = sources[0]
        if source is not None:
            if source.kind not in ("agenda", "minutes"):
                messages.append("extraction: document kind must be agenda or minutes")
            if source.unit_text is not None:
                result = verify(
                    payload.kind, payload.record | {"quote": payload.passage.text}, source.unit_text
                )
                if not result.ok:
                    messages.append(f"extraction: {result.error}")
    if isinstance(payload, AgentTask):
        if len(set(payload.document_ids)) < 2:
            messages.append("agent_task: needs at least two distinct documents")
        for document in payload.document_ids:
            if not conn.execute(
                "SELECT 1 FROM documents WHERE id = %s AND city_id = %s AND status = 'downloaded'",
                (document, label.city),
            ).fetchone():
                messages.append(f"agent_task: unknown document {document}")
    if isinstance(payload, (QuestionAnswerable, QuestionUnanswerable, AgentTask)) and (
        not 10 <= len(payload.question) <= 300 or not payload.question.endswith("?")
    ):
        messages.append("question must be 10 to 300 characters and end with a question mark")
    if messages:
        raise LabelValidationError(messages)


def _label(row: tuple[Any, ...]) -> Label:
    label_id, label_type, city, payload, author, created_utc, reviewed, note = row
    return Label(
        id=label_id,
        type=label_type,
        city=city,
        payload=payload,
        author=author,
        created_utc=created_utc,
        human_reviewed=reviewed,
        note=note,
    )


def create(
    conn: psycopg.Connection[TupleRow],
    label: LabelIn,
    *,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    labels_dir: Path = Path("eval/labels"),
) -> Label:
    require_unfrozen(labels_dir)
    _validate(conn, label)
    prefix, width = _PREFIXES[label.type]
    stem = f"{prefix}-{label.city}-"
    ids = conn.execute(
        "SELECT id FROM labels WHERE label_type = %s AND city_id = %s", (label.type, label.city)
    ).fetchall()
    number = 1 + max((int(row[0].removeprefix(stem)) for row in ids), default=0)
    label_id = f"{stem}{number:0{width}d}"
    created = now().astimezone(UTC).replace(microsecond=0)
    conn.execute(
        "INSERT INTO labels (id, label_type, city_id, payload, author, created_utc, note) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (label_id, label.type, label.city, Jsonb(label.payload), label.author, created, label.note),
    )
    return get(conn, label_id)


def update(
    conn: psycopg.Connection[TupleRow],
    label_id: str,
    label: LabelIn,
    *,
    labels_dir: Path = Path("eval/labels"),
) -> Label:
    require_unfrozen(labels_dir)
    existing = get(conn, label_id)
    _validate(conn, label, label_id)
    if label.type != existing.type or label.city != existing.city:
        raise LabelValidationError(["label type and city cannot change"])
    conn.execute(
        "UPDATE labels SET payload = %s, author = %s, note = %s WHERE id = %s",
        (Jsonb(label.payload), label.author, label.note, label_id),
    )
    return get(conn, label_id)


def delete(
    conn: psycopg.Connection[TupleRow],
    label_id: str,
    *,
    labels_dir: Path = Path("eval/labels"),
) -> None:
    require_unfrozen(labels_dir)
    if not conn.execute("DELETE FROM labels WHERE id = %s RETURNING id", (label_id,)).fetchone():
        raise NotFoundError(f"unknown label {label_id}")


def get(conn: psycopg.Connection[TupleRow], label_id: str) -> Label:
    row = conn.execute(f"SELECT {_COLUMNS} FROM labels WHERE id = %s", (label_id,)).fetchone()
    if row is None:
        raise NotFoundError(f"unknown label {label_id}")
    return _label(row)


def list_labels(
    conn: psycopg.Connection[TupleRow], label_type: str | None = None, city: str | None = None
) -> list[Label]:
    rows = conn.execute(
        f"SELECT {_COLUMNS} FROM labels WHERE (%s::text IS NULL OR label_type = %s) "
        "AND (%s::text IS NULL OR city_id = %s) ORDER BY id",
        (label_type, label_type, city, city),
    ).fetchall()
    return [_label(row) for row in rows]


def set_reviewed(
    conn: psycopg.Connection[TupleRow],
    label_id: str,
    reviewed: bool,
    *,
    labels_dir: Path = Path("eval/labels"),
) -> Label:
    require_unfrozen(labels_dir)
    if not conn.execute(
        "UPDATE labels SET human_reviewed = %s WHERE id = %s RETURNING id", (reviewed, label_id)
    ).fetchone():
        raise NotFoundError(f"unknown label {label_id}")
    return get(conn, label_id)
