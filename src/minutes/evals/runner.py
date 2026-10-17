import json
import subprocess
from collections import Counter
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Literal, cast

import psycopg
from psycopg.rows import TupleRow

from minutes import corpus, db
from minutes.answer.pipeline import answer as answer_question
from minutes.config import Corpus, corpus_database_url, get_settings
from minutes.errors import ConfigError, LLMOutputError, LLMTransportError, ValidationError
from minutes.evals.grader import grade_answer, grade_support
from minutes.evals.metrics import (
    mean_or_none,
    percentile,
    recall_for_question,
    segmentation_metrics,
    summarize_runs,
)
from minutes.extract import extractor
from minutes.extract.verify import verify
from minutes.ingest import pipeline, quality
from minutes.labels import export, store
from minutes.labels.schema import (
    AgendaCount,
    ExtractionCase,
    Label,
    LabelType,
    QuestionAnswerable,
    QuestionUnanswerable,
    parse_payload,
)
from minutes.log import bind_correlation_id, get_correlation_id, get_logger, new_correlation_id
from minutes.retrieval.search import Filters, Hit, search

Labels = list[Label]
Connection = psycopg.Connection[TupleRow]
Result = dict[str, object]
Suite = Literal["gate", "full"]
Mode = Literal["keyword", "vector", "hybrid", "hybrid_rerank"]
Chunker = Literal["fixed", "item"]
LABEL_TYPES: tuple[LabelType, ...] = ("question", "agenda_count", "extraction", "agent_task")
_LOG = get_logger("evals")


@dataclass(frozen=True)
class _ExtractedFact:
    kind: str
    data: dict[str, object]
    unit_index: int
    quote_start: int
    quote_end: int


@dataclass(frozen=True)
class _ExtractionDocumentResult:
    document_id: str
    correct: dict[int, dict[str, bool]]
    rejected: int
    produced: int


class EvalContext:
    def __init__(
        self,
        suite: Suite,
        corpus_name: Corpus,
        conn: Connection,
        labels: Labels,
        controls: bool,
        previous: dict[str, object] | None = None,
    ) -> None:
        self.suite = suite
        # results already in the output file; a stored baseline answer run is kept (not re-run)
        self.previous = previous or {}
        self.corpus = corpus_name
        self.conn = conn
        self.labels = labels
        self.controls = controls


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


def _search(
    context: EvalContext, label: Label, query: str, mode: Mode, chunker: Chunker, k: int
) -> list[Hit]:
    return search(
        context.conn,
        query,
        Filters(city=label.city),
        mode=mode,
        chunker=chunker,
        k=k,
    )


def _retrieval(context: EvalContext) -> Result:
    questions = _question_labels(context.labels)
    answerable = [
        (label, payload) for label, payload in questions if isinstance(payload, QuestionAnswerable)
    ]
    result: Result = {}
    latency: dict[str, object] = {}
    modes: dict[Chunker, tuple[Mode, ...]] = {
        "fixed": ("keyword", "vector"),
        "item": ("keyword", "vector", "hybrid", "hybrid_rerank"),
    }
    for chunker, chunker_modes in modes.items():
        results: Result = {}
        result[chunker] = results
        for mode in chunker_modes:
            values: dict[int, list[float | None]] = {5: [], 10: [], 20: []}
            for label, payload in answerable:
                hits = _search(context, label, payload.question, mode, chunker, 20)
                for k in values:
                    values[k].append(recall_for_question(payload.passages, hits, k))
            results[mode] = {f"recall_at_{k}": mean_or_none(scores) for k, scores in values.items()}
            if chunker == "item" and context.suite == "full" and context.corpus == "full":
                for label, question_payload in questions[:10]:
                    _search(context, label, question_payload.question, mode, chunker, 20)
                timings = []
                for _ in range(5):
                    for label, question_payload in questions:
                        start = perf_counter()
                        _search(context, label, question_payload.question, mode, chunker, 20)
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


def _last_token(value: object) -> str | None:
    if value is None:
        return None
    words = str(value).strip().split()
    return words[-1].casefold() if words else ""


def _same_record(kind: str, predicted: dict[str, object], gold: dict[str, object]) -> bool:
    if kind == "motion":
        return (
            _last_token(predicted["mover"]) == _last_token(gold["mover"])
            and _last_token(predicted["seconder"]) == _last_token(gold["seconder"])
            and str(predicted["outcome"]).strip().casefold()
            == str(gold["outcome"]).strip().casefold()
        )
    if kind == "vote":
        if (
            any(predicted[field] != gold[field] for field in ("ayes", "noes", "abstain", "absent"))
            or str(predicted["outcome"]).strip().casefold()
            != str(gold["outcome"]).strip().casefold()
        ):
            return False
        gold_members = cast(list[dict[str, object]], gold["members"])
        if not gold_members:
            return True
        predicted_members = cast(list[dict[str, object]], predicted["members"])
        return {
            (_last_token(member["name"]), str(member["value"]).casefold())
            for member in predicted_members
        } == {
            (_last_token(member["name"]), str(member["value"]).casefold())
            for member in gold_members
        }
    if kind == "ordinance":
        predicted_identifier = "".join(str(predicted["identifier"]).split()).casefold()
        gold_identifier = "".join(str(gold["identifier"]).split()).casefold()
        return (
            predicted_identifier == gold_identifier
            and str(predicted["status"]).strip().casefold()
            == str(gold["status"]).strip().casefold()
        )
    if kind == "amount":
        predicted_amount = cast(float | int, predicted["amount_usd"])
        gold_amount = cast(float | int, gold["amount_usd"])
        return abs(float(predicted_amount) - float(gold_amount)) <= 0.005
    if kind == "statement":
        return _last_token(predicted["speaker"]) == _last_token(gold["speaker"])
    raise ValueError(f"unknown extraction kind {kind}")


def _extract_document(
    document_id: str,
    cases: list[tuple[Label, ExtractionCase]],
    database_url: str,
    controls: bool,
) -> _ExtractionDocumentResult:
    with db.connect(database_url) as conn:
        row = conn.execute(
            "SELECT d.kind, c.name, m.body, m.meeting_date FROM documents d "
            "JOIN meetings m ON m.id = d.meeting_id JOIN cities c ON c.id = d.city_id "
            "WHERE d.id = %s",
            (document_id,),
        ).fetchone()
        if row is None:
            raise ValidationError(f"unknown extraction document {document_id}")
        document_kind, city, body, meeting_date = row
        items = extractor.item_texts(conn, document_id)
        units = dict(
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
    items_by_id = {item.item_id: item for item in items}
    correct: dict[int, dict[str, bool]] = {}
    rejected = produced = 0
    for sample_index in range(5):
        predictions: list[_ExtractedFact] = []
        try:
            raw_facts = extractor.extract_items(
                items,
                controls=controls,
                sample_index=sample_index,
                city=city,
                body=body,
                meeting_date=meeting_date.isoformat(),
                document_kind=document_kind,
            )
        except (LLMOutputError, LLMTransportError) as err:
            _LOG.error(
                "extraction evaluation failed",
                extra={
                    "event": "extraction_evaluation_failed",
                    "fields": {
                        "document_id": document_id,
                        "sample_index": sample_index,
                        "error": str(err)[:200],
                    },
                },
            )
            raw_facts = []
        for fact in raw_facts:
            produced += 1
            item = items_by_id.get(fact.item_id) if fact.item_id is not None else None
            if item is None or not fact.data:
                rejected += 1
                continue
            result = verify(fact.kind, fact.data | {"quote": fact.quote}, item.text)
            if not result.ok or result.quote_start is None or result.quote_end is None:
                rejected += 1
                continue
            unit_index, quote_start, quote_end = extractor.quote_unit_offsets(
                item,
                result.quote_start,
                result.quote_end,
                item_bounds[item.item_id],
                units,
            )
            predictions.append(
                _ExtractedFact(fact.kind, fact.data, unit_index, quote_start, quote_end)
            )
        correct[sample_index] = {}
        for label, case in cases:
            passage = case.passage
            correct[sample_index][label.id] = any(
                prediction.kind == case.kind
                and prediction.unit_index == passage.unit_index
                and prediction.quote_start < passage.end
                and prediction.quote_end > passage.start
                and _same_record(case.kind, prediction.data, case.record)
                for prediction in predictions
            )
    return _ExtractionDocumentResult(document_id, correct, rejected, produced)


def _extraction(context: EvalContext) -> Result:
    cases: list[tuple[Label, ExtractionCase]] = []
    for label in context.labels:
        if label.type == "extraction":
            cases.append((label, cast(ExtractionCase, parse_payload(label.type, label.payload))))
    by_document: dict[str, list[tuple[Label, ExtractionCase]]] = {}
    for label, case in cases:
        by_document.setdefault(case.passage.document_id, []).append((label, case))
    database_url = corpus_database_url(context.corpus)
    results: list[_ExtractionDocumentResult] = []
    with ThreadPoolExecutor(max_workers=pipeline.EXTRACTION_WORKERS) as pool:
        futures: dict[Future[_ExtractionDocumentResult], str] = {
            pool.submit(
                _extract_document,
                document_id,
                document_cases,
                database_url,
                context.controls,
            ): document_id
            for document_id, document_cases in sorted(by_document.items())
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda result: result.document_id)

    correctness: dict[int, dict[str, bool]] = {sample_index: {} for sample_index in range(5)}
    rejected = sum(result.rejected for result in results)
    produced = sum(result.produced for result in results)
    for result in results:
        for sample_index, values in result.correct.items():
            correctness[sample_index].update(values)

    def accuracy(selected: list[tuple[Label, ExtractionCase]], sample_index: int) -> float | None:
        if not selected:
            return None
        return sum(correctness[sample_index].get(label.id, False) for label, _ in selected) / len(
            selected
        )

    overall = summarize_runs([accuracy(cases, sample_index) for sample_index in range(5)])
    per_kind = {
        kind: summarize_runs(
            [
                accuracy(
                    [(label, case) for label, case in cases if case.kind == kind], sample_index
                )
                for sample_index in range(5)
            ]
        )
        for kind in sorted({case.kind for _, case in cases})
    }
    per_city = {
        city: summarize_runs(
            [
                accuracy(
                    [(label, case) for label, case in cases if label.city == city], sample_index
                )
                for sample_index in range(5)
            ]
        )
        for city in sorted({label.city for label, _ in cases})
    }
    return {
        "overall": overall,
        "per_kind": per_kind,
        "per_city": per_city,
        "verifier_reject_rate": rejected / produced if produced else 0.0,
    }


def _answers(context: EvalContext) -> Result:
    if context.suite not in ("full", "gate"):
        return {}
    questions = _question_labels(context.labels)
    answerable = [
        (label, payload) for label, payload in questions if isinstance(payload, QuestionAnswerable)
    ]
    unanswerable = [
        (label, payload)
        for label, payload in questions
        if isinstance(payload, QuestionUnanswerable)
    ]
    stored = cast(dict[str, object], context.previous.get("answers") or {}).get("baseline")
    pipeline_results: Result = {"baseline": stored} if context.suite == "full" and stored else {}
    pipelines: tuple[Literal["baseline", "final"], ...] = (
        ("final",)
        if context.suite == "gate" or "baseline" in pipeline_results
        else ("baseline", "final")
    )
    for pipeline_name in pipelines:
        answer_accuracy: list[float | None] = []
        decline_accuracy: list[float | None] = []
        false_decline_rate: list[float | None] = []
        faithfulness: list[float | None] = []
        sample_zero_latency: list[float] = []
        for sample_index in range(5):
            # the response cache ignores the sample, so each sample starts from an empty cache
            context.conn.execute("DELETE FROM answer_cache")
            correct_answers = 0
            declined_answerable = 0
            declined_unanswerable = 0
            supported_sentences = 0
            sentence_count = 0
            for label, payload in questions:
                result = answer_question(
                    context.conn,
                    payload.question,
                    Filters(city=label.city),
                    pipeline=pipeline_name,
                    controls=context.controls,
                    sample_index=sample_index,
                )
                if sample_index == 0:
                    sample_zero_latency.append(float(result.latency_ms))
                if not result.declined:
                    citation_texts = {
                        citation.number: citation.quote for citation in result.citations
                    }
                    support_inputs = [
                        (sentence.text, [citation_texts[number] for number in sentence.citations])
                        for sentence in result.sentences
                    ]
                    support_results = grade_support(support_inputs, sample_index)
                    supported_sentences += sum(support_results)
                    sentence_count += len(support_results)
                if isinstance(payload, QuestionAnswerable):
                    if result.declined:
                        declined_answerable += 1
                        continue
                    answer_text = " ".join(sentence.text for sentence in result.sentences)
                    correct_answers += grade_answer(label, answer_text, sample_index)
                elif result.declined:
                    declined_unanswerable += 1
            answer_accuracy.append(correct_answers / len(answerable) if answerable else None)
            decline_accuracy.append(
                declined_unanswerable / len(unanswerable) if unanswerable else None
            )
            false_decline_rate.append(declined_answerable / len(answerable) if answerable else None)
            faithfulness.append(supported_sentences / sentence_count if sentence_count else None)
        pipeline_results[pipeline_name] = {
            "answer_accuracy": summarize_runs(answer_accuracy),
            "decline_accuracy": summarize_runs(decline_accuracy),
            "false_decline_rate": summarize_runs(false_decline_rate),
            "faithfulness": summarize_runs(faithfulness),
            "latency_ms": {
                "p50": percentile(sample_zero_latency, 50) if sample_zero_latency else None,
                "p95": percentile(sample_zero_latency, 95) if sample_zero_latency else None,
            },
        }
    return pipeline_results


def _cost(context: EvalContext) -> Result:
    questions = _question_labels(context.labels)
    by_city: dict[str, list[tuple[Label, QuestionAnswerable | QuestionUnanswerable]]] = {}
    for question in questions:
        by_city.setdefault(question[0].city, []).append(question)
    workload = questions[:60] + [
        question for city in sorted(by_city) for question in by_city[city][:15]
    ]
    context.conn.execute("DELETE FROM answer_cache")
    total_cost = 0.0
    previous_id = get_correlation_id()
    try:
        for label, payload in workload:
            correlation_id = new_correlation_id()
            bind_correlation_id(correlation_id)
            answer_question(
                context.conn,
                payload.question,
                Filters(city=label.city),
                pipeline="final",
                controls=context.controls,
            )
            row = context.conn.execute(
                "SELECT coalesce(sum(cost_usd), 0) FROM llm_calls WHERE correlation_id = %s",
                (correlation_id,),
            ).fetchone()
            assert row is not None
            total_cost += float(row[0])
    finally:
        bind_correlation_id(previous_id)
    key = "controls_on" if context.controls else "controls_off"
    cost = cast(dict[str, object], context.previous.get("cost") or {}).copy()
    per_query = round(total_cost / len(workload), 6)
    cost[key] = {"queries": len(workload), "cost_per_query_usd": per_query}
    other_key = "controls_off" if context.controls else "controls_on"
    if other_key in cost:
        on = cast(dict[str, float], cost["controls_on"])["cost_per_query_usd"]
        off = cast(dict[str, float], cost["controls_off"])["cost_per_query_usd"]
        cost["reduction"] = 1 - on / off if off else None
    else:
        cost.pop("reduction", None)
    return cost


SECTIONS: dict[str, Callable[[EvalContext], Result]] = {
    "retrieval": _retrieval,
    "segmentation": _segmentation,
    "extraction": _extraction,
    "answers": _answers,
    "cost": _cost,
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


def write_results(out_path: Path, blocks: dict[str, object]) -> dict[str, object]:
    """Merge blocks into the results file and return what was written."""
    existing: dict[str, object] = {}
    if out_path.exists():
        existing = cast(dict[str, object], json.loads(out_path.read_text(encoding="utf-8")))
    results = cast(dict[str, object], _round_values({**existing, **blocks}))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return results


def run_suite(
    suite: Suite,
    *,
    controls: bool,
    labels_dir: Path,
    out_path: Path,
    only: set[str] | None = None,
) -> dict[str, object]:
    """Run registered evaluation sections and merge their results into a JSON file."""
    available = (
        {"retrieval", "segmentation", "extraction", "answers", "cost"}
        if suite == "gate"
        else set(SECTIONS)
    )
    if only is None and suite == "gate":
        names = ["retrieval", "segmentation", "extraction", "answers", "cost"]
    elif only is None:
        names = list(SECTIONS)
    else:
        names = sorted(only)
    for name in names:
        if name not in available:
            raise ValidationError(f"unknown section {name}")
    settings = get_settings()
    if settings.corpus != "fixture":
        store.require_frozen()
    if suite == "gate" and not store.is_frozen(labels_dir):
        raise ConfigError("label files are not frozen")
    labels = load_labels(labels_dir)
    with db.connect(corpus_database_url(settings.corpus)) as conn:
        previous = (
            cast(dict[str, object], json.loads(out_path.read_text(encoding="utf-8")))
            if out_path.exists()
            else {}
        )
        context = EvalContext(suite, settings.corpus, conn, labels, controls, previous)
        results: dict[str, object] = {
            "suite": suite,
            "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": _git_commit(),
            "corpus": _corpus_block(conn),
            "labels": _labels_block(labels),
        }
        for name in names:
            results[name] = SECTIONS[name](context)
    return write_results(out_path, results)
