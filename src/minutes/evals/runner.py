import json
import subprocess
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Literal, cast

import psycopg
from psycopg.rows import TupleRow

from minutes import corpus, db
from minutes.config import Corpus, corpus_database_url, get_settings
from minutes.errors import ValidationError
from minutes.evals.metrics import (
    mean_or_none,
    percentile,
    recall_for_question,
    segmentation_metrics,
)
from minutes.ingest import quality
from minutes.labels import export, store
from minutes.labels.schema import (
    AgendaCount,
    Label,
    LabelType,
    QuestionAnswerable,
    QuestionUnanswerable,
    parse_payload,
)
from minutes.retrieval.search import Filters, Hit, search

Labels = list[Label]
Connection = psycopg.Connection[TupleRow]
Result = dict[str, object]
Suite = Literal["gate", "full"]
Mode = Literal["keyword", "vector"]
LABEL_TYPES: tuple[LabelType, ...] = ("question", "agenda_count", "extraction", "agent_task")


class EvalContext:
    def __init__(self, suite: Suite, corpus_name: Corpus, conn: Connection, labels: Labels) -> None:
        self.suite = suite
        self.corpus = corpus_name
        self.conn = conn
        self.labels = labels


def load_labels(labels_dir: Path) -> Labels:
    """Read and validate the four JSONL evaluation label files."""
    labels = []
    for label_type, filename in export.LABEL_FILES.items():
        path = labels_dir / filename
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                label = Label.model_validate_json(line)
                if label.type != label_type:
                    raise ValidationError(f"wrong label type in {filename}")
                parse_payload(label.type, label.payload)
                labels.append(label)
    return sorted(labels, key=lambda label: label.id)


def _corpus_block(conn: Connection) -> Result:
    stats = corpus.corpus_stats(conn)
    quality_by_city = {
        item.city_id: item.parse_quality for item in quality.parse_quality_report(conn)
    }
    per_city: dict[str, Result] = {}
    for city_id, city in sorted(stats.per_city.items()):
        per_city[city_id] = {
            "documents": city.documents,
            "pages": city.pages,
            "segments": city.segments,
            "ocr_pages": city.ocr_pages,
            "parse_quality": quality_by_city.get(city_id),
        }
    return {
        "cities": len(per_city),
        "documents": sum(city.documents for city in stats.per_city.values()),
        "pages": stats.total_pages,
        "segments": sum(city.segments for city in stats.per_city.values()),
        "per_city": per_city,
    }


def _labels_block(labels: Labels) -> Result:
    counts = Counter(label.type for label in labels)
    reviewed = Counter(label.type for label in labels if label.human_reviewed)
    authors = Counter(label.author for label in labels)
    return {
        **{label_type: counts[label_type] for label_type in LABEL_TYPES},
        "human_reviewed": {label_type: reviewed[label_type] for label_type in LABEL_TYPES},
        "authors": {author: authors[author] for author in ("agent", "human")},
    }


def _question_labels(
    labels: Labels,
) -> list[tuple[Label, QuestionAnswerable | QuestionUnanswerable]]:
    questions = []
    for label in labels:
        if label.type == "question":
            payload = parse_payload(label.type, label.payload)
            if isinstance(payload, (QuestionAnswerable, QuestionUnanswerable)):
                questions.append((label, payload))
    return questions


def _search(context: EvalContext, label: Label, query: str, mode: Mode, k: int) -> list[Hit]:
    return search(
        context.conn,
        query,
        Filters(city=label.city),
        mode=mode,
        chunker="fixed",
        k=k,
    )


def _retrieval(context: EvalContext) -> Result:
    questions = _question_labels(context.labels)
    answerable = [
        (label, payload) for label, payload in questions if isinstance(payload, QuestionAnswerable)
    ]
    result: Result = {"fixed": {}}
    fixed = cast(dict[str, object], result["fixed"])
    latency: dict[str, object] = {}
    for mode in ("keyword", "vector"):
        values: dict[int, list[float]] = {5: [], 10: [], 20: []}
        for label, payload in answerable:
            hits = _search(context, label, payload.question, mode, 20)
            for k in values:
                values[k].append(recall_for_question(payload.passages, hits, k))
        fixed[mode] = {f"recall_at_{k}": mean_or_none(scores) for k, scores in values.items()}
        if context.suite == "full" and context.corpus == "full":
            for label, question_payload in questions[:10]:
                _search(context, label, question_payload.question, mode, 20)
            timings = []
            for _ in range(5):
                for label, question_payload in questions:
                    start = perf_counter()
                    _search(context, label, question_payload.question, mode, 20)
                    timings.append((perf_counter() - start) * 1000)
            latency[mode] = {
                "p50": percentile(timings, 50),
                "p95": percentile(timings, 95),
            }
    if latency:
        result["latency_ms"] = latency
    return result


def _segmentation(context: EvalContext) -> Result:
    counts = []
    for label in context.labels:
        if label.type == "agenda_count":
            payload = cast(AgendaCount, parse_payload(label.type, label.payload))
            counts.append((label.city, payload.meeting_id, payload.count))
    prediction_rows = context.conn.execute(
        "SELECT d.meeting_id, count(i.id) FROM documents d "
        "LEFT JOIN items i ON i.document_id = d.id AND i.kind = 'business' "
        "WHERE d.kind = 'agenda' GROUP BY d.meeting_id"
    ).fetchall()
    predictions: dict[str, int] = {
        meeting_id: int(prediction) for meeting_id, prediction in prediction_rows
    }
    rows = [
        (city, meeting_id, gold, int(predictions.get(meeting_id, 0)))
        for city, meeting_id, gold in counts
    ]
    overall = segmentation_metrics([(meeting_id, gold, pred) for _, meeting_id, gold, pred in rows])
    per_city = {
        city: segmentation_metrics(
            [
                (meeting_id, gold, pred)
                for row_city, meeting_id, gold, pred in rows
                if row_city == city
            ]
        )
        for city in sorted({city for city, _, _, _ in rows})
    }
    return {
        "overall": {"exact_rate": overall["exact_rate"], "mae": overall["mae"]},
        "per_city": {
            city: {"exact_rate": metrics["exact_rate"], "mae": metrics["mae"]}
            for city, metrics in per_city.items()
        },
        "per_meeting": [
            {
                "meeting_id": meeting_id,
                "gold": gold,
                "pred": pred,
                "signed_error": pred - gold,
            }
            for _, meeting_id, gold, pred in sorted(rows, key=lambda item: item[1])
        ],
    }


SECTIONS: dict[str, Callable[[EvalContext], Result]] = {
    "retrieval": _retrieval,
    "segmentation": _segmentation,
}


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, check=False, text=True
        )
    except OSError:
        return "uncommitted"
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else "uncommitted"


def _round_values(value: object, in_cost: bool = False) -> object:
    if isinstance(value, float):
        return round(value, 6 if in_cost else 4)
    if isinstance(value, dict):
        return {key: _round_values(item, in_cost or key == "cost") for key, item in value.items()}
    if isinstance(value, list):
        return [_round_values(item, in_cost) for item in value]
    return value


def run_suite(
    suite: Suite,
    *,
    controls: bool,
    labels_dir: Path,
    out_path: Path,
    only: set[str] | None = None,
) -> dict[str, object]:
    """Run registered evaluation sections and merge their results into a JSON file."""
    names = list(SECTIONS) if only is None else sorted(only)
    for name in names:
        if name not in SECTIONS:
            raise ValidationError(f"unknown section {name}")
    settings = get_settings()
    if settings.corpus != "fixture":
        store.require_frozen()
    labels = load_labels(labels_dir)
    existing: dict[str, object] = {}
    if out_path.exists():
        existing = cast(dict[str, object], json.loads(out_path.read_text(encoding="utf-8")))
    with db.connect(corpus_database_url(settings.corpus)) as conn:
        context = EvalContext(suite, settings.corpus, conn, labels)
        results = {
            **existing,
            "suite": suite,
            "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": _git_commit(),
            "corpus": _corpus_block(conn),
            "labels": _labels_block(labels),
        }
        for name in names:
            results[name] = SECTIONS[name](context)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = _round_values(results)
    out_path.write_text(
        json.dumps(serialized, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return cast(dict[str, object], serialized)
