import json
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, NoReturn
from urllib.parse import urlsplit

import typer
import uvicorn
from pydantic import ValidationError as PayloadError
from typer.core import TyperGroup

from minutes import corpus, db, probe, queue, worker
from minutes.answer.pipeline import answer
from minutes.api.app import create_app
from minutes.api.schemas import Chunker, Pipeline, SearchMode
from minutes.config import Corpus, corpus_database_url, get_settings, load_cities
from minutes.corpus import assert_minimums, corpus_stats
from minutes.errors import (
    ConfigError,
    GateStaleError,
    LabelRuleError,
    LabelValidationError,
    LLMQuotaError,
    MinutesError,
    NotFoundError,
)
from minutes.ingest import pipeline, quality
from minutes.labels import export, store
from minutes.labels.rules import PRODUCTION_RULES, check
from minutes.labels.schema import LabelIn, LabelType
from minutes.log import bind_correlation_id, configure_logging, get_logger, new_correlation_id
from minutes.retrieval.search import Filters, city_names, hit_heading, search, span_label

EXIT_CODES: dict[type[MinutesError], int] = {
    ConfigError: 2,
    GateStaleError: 3,
    LabelRuleError: 4,
    LLMQuotaError: 75,
}


class ErrorGroup(TyperGroup):
    def invoke(self, ctx: Any) -> Any:
        try:
            return super().invoke(ctx)
        except MinutesError as err:
            messages = (
                err.args[0]
                if isinstance(err, (LabelRuleError, LabelValidationError))
                else [str(err)]
            )
            for message in messages:
                typer.echo(f"error: {message}", err=True)
            raise typer.Exit(EXIT_CODES.get(type(err), 1)) from err


app = typer.Typer(no_args_is_help=True, cls=ErrorGroup)
probe_app = typer.Typer(no_args_is_help=True)
db_app = typer.Typer(no_args_is_help=True)
app.add_typer(probe_app, name="probe")
app.add_typer(db_app, name="db")
ingest_app = typer.Typer(no_args_is_help=True)
worker_app = typer.Typer(no_args_is_help=True)
jobs_app = typer.Typer(no_args_is_help=True)
fixtures_app = typer.Typer(no_args_is_help=True)
corpus_app = typer.Typer(no_args_is_help=True)
labels_app = typer.Typer(no_args_is_help=True)
app.add_typer(ingest_app, name="ingest")
app.add_typer(worker_app, name="worker")
app.add_typer(jobs_app, name="jobs")
app.add_typer(fixtures_app, name="fixtures")
app.add_typer(corpus_app, name="corpus")
app.add_typer(labels_app, name="labels")

LABELS_DIR = Path("eval/labels")
LABEL_RULES = PRODUCTION_RULES


@app.callback()
def main() -> None:
    """Search and question answering over city council records."""
    configure_logging(get_settings().log_level)
    bind_correlation_id(new_correlation_id())


def report(results: list[probe.ProbeResult]) -> None:
    for result in results:
        if result.ok:
            typer.echo(f"ok {result.name} {result.detail}".rstrip())
        else:
            typer.echo(f"FAIL {result.name}: {result.detail}")
    if not all(result.ok for result in results):
        raise typer.Exit(1)


@probe_app.command("llm")
def probe_llm() -> None:
    """Check that both language models answer through the endpoint."""
    report(probe.probe_llm())


@probe_app.command("sources")
def probe_sources() -> None:
    """Check that each city's listing, PDFs, and captions download."""
    report(probe.probe_sources())


@probe_app.command("gpu")
def probe_gpu() -> None:
    """Check that the embedder, reranker, and OCR models load on the GPU."""
    report(probe.probe_gpu())


@probe_app.command("ocr")
def probe_ocr() -> None:
    """Measure GPU OCR character error rate on the scanned fixture page."""
    report(probe.probe_ocr())


def use_corpus(corpus: str | None) -> Corpus:
    """Apply a --corpus option to this process and return the corpus in effect."""
    if corpus is not None:
        os.environ["MINUTES_CORPUS"] = corpus
        get_settings.cache_clear()
    return get_settings().corpus


CorpusOption = Annotated[str | None, typer.Option("--corpus", help="fixture, dev, or full")]


@db_app.command("migrate")
def db_migrate(corpus: CorpusOption = None) -> None:
    """Apply the migrations that have not run yet."""
    applied = db.migrate(corpus_database_url(use_corpus(corpus)))
    for version in applied:
        typer.echo(f"applied {version}")
    if not applied:
        typer.echo("up to date")


@db_app.command("rollback")
def db_rollback(
    steps: Annotated[int, typer.Option(min=1)] = 1, corpus: CorpusOption = None
) -> None:
    """Undo the latest migrations."""
    for version in db.rollback(corpus_database_url(use_corpus(corpus)), steps):
        typer.echo(f"rolled back {version}")


@db_app.command("reset")
def db_reset(
    yes: Annotated[bool, typer.Option("--yes")] = False,
    everything: Annotated[bool, typer.Option("--all")] = False,
) -> None:
    """Drop, recreate, and migrate the test database; with --all the corpus database too."""
    if not yes:
        typer.echo("refusing without --yes")
        raise typer.Exit(2)
    settings = get_settings()
    urls = [settings.test_database_url]
    if everything:
        urls.insert(0, settings.database_url)
    for url in urls:
        db.reset(url)
        typer.echo(f"reset {urlsplit(url).path.lstrip('/')}")
        if everything and url == settings.database_url:
            with db.connect(url) as conn:
                # The reset removes cities, which the imported labels reference.
                for city, config in load_cities("full").items():
                    conn.execute(
                        "INSERT INTO cities (id, name, state, item_format) VALUES (%s, %s, %s, %s)",
                        (city, config.name, config.state, config.item_format),
                    )
                count = export.import_all(conn, LABELS_DIR)
            typer.echo(f"imported {count} labels")


@ingest_app.command("discover")
def ingest_discover(
    city: Annotated[str | None, typer.Option()] = None, corpus: CorpusOption = None
) -> None:
    """List meetings from the city sources and add the documents not seen before."""
    used = use_corpus(corpus)
    with db.connect(corpus_database_url(used)) as conn:
        added = pipeline.discover(conn, used, city)
    typer.echo(f"discovered {added} new documents")


def summary_line(summary: worker.WorkerSummary) -> str:
    return (
        f"done {summary.done} skipped {summary.skipped} failed {summary.failed} dead {summary.dead}"
    )


def quota_pause() -> NoReturn:
    typer.echo("paused: quota exhausted, rerun the same command to resume")
    raise typer.Exit(75)


@ingest_app.command("run")
def ingest_run(
    stages: Annotated[str, typer.Option()],
    city: Annotated[str | None, typer.Option()] = None,
    corpus: CorpusOption = None,
) -> None:
    """Run the selected ingestion stages through the job queue."""
    names = tuple(stages.split(","))
    if (
        len(set(names)) != len(names)
        or any(name not in pipeline.STAGES for name in names)
        or tuple(sorted(names, key=pipeline.STAGES.index)) != names
    ):
        raise ConfigError(
            "stages must be a subset of download,parse,ocr,segment,chunk,embed,extract "
            "in that order"
        )
    used = use_corpus(corpus)
    if used != "fixture" and any(stage in pipeline.LABEL_GATED_STAGES for stage in names):
        store.require_frozen(LABELS_DIR)
    totals = [0, 0, 0, 0]
    for stage in names:
        with db.connect(corpus_database_url(used)) as conn:
            for document_id in pipeline.documents_for(conn, used, city):
                queue.enqueue(conn, stage, document_id, used)
        try:
            result = worker.run(used, drain=True, stage=stage)
        except LLMQuotaError:
            quota_pause()
        typer.echo(f"{stage}: {summary_line(result)}")
        totals[0] += result.done
        totals[1] += result.skipped
        totals[2] += result.failed
        totals[3] += result.dead
    typer.echo(f"done {totals[0]} skipped {totals[1]} failed {totals[2]} dead {totals[3]}")
    if totals[3]:
        raise typer.Exit(1)


@worker_app.command("run")
def worker_run(
    drain: Annotated[bool, typer.Option("--drain/--no-drain")] = False,
    corpus: CorpusOption = None,
) -> None:
    """Run queued stage jobs until the queue drains or the process is stopped."""
    try:
        result = worker.run(use_corpus(corpus), drain=drain)
    except LLMQuotaError:
        quota_pause()
    typer.echo(summary_line(result))
    if result.dead:
        raise typer.Exit(1)


@jobs_app.command("list")
def jobs_list(
    status: Annotated[str | None, typer.Option()] = None, corpus: CorpusOption = None
) -> None:
    """Print the newest queued and completed jobs."""
    with db.connect(corpus_database_url(use_corpus(corpus))) as conn:
        rows = conn.execute(
            "SELECT id, stage, document_id, status, attempts, last_error FROM jobs "
            "WHERE (%s::text IS NULL OR status = %s) ORDER BY id DESC LIMIT 200",
            (status, status),
        ).fetchall()
    for job_id, stage, document_id, job_status, attempts, last_error in rows:
        typer.echo(f"{job_id} {stage} {document_id} {job_status} {attempts} {last_error or ''}")


@jobs_app.command("retry-dead")
def jobs_retry_dead(corpus: CorpusOption = None) -> None:
    """Return every dead job to the queue."""
    with db.connect(corpus_database_url(use_corpus(corpus))) as conn:
        count = queue.retry_dead(conn)
    typer.echo(f"requeued {count}")


@fixtures_app.command("load")
def fixtures_load() -> None:
    """Rebuild the test database from the tracked test corpus, with stub models."""
    os.environ.update(MINUTES_CORPUS="fixture", MINUTES_MODELS_MODE="stub", MINUTES_LLM_MODE="stub")
    get_settings.cache_clear()
    count = pipeline.load_fixture_corpus(get_settings().test_database_url)
    typer.echo(f"loaded {count} documents")


@corpus_app.command("make-dev-subset")
def corpus_make_dev_subset(seed: Annotated[int, typer.Option("--seed")] = 7) -> None:
    """Write the fixed development document list."""
    with db.connect(get_settings().database_url) as conn:
        documents = corpus.make_dev_subset(conn, seed)
    typer.echo(f"wrote {len(documents)} documents")


@corpus_app.command("stats")
def corpus_stats_command(
    check_minimums: Annotated[bool, typer.Option("--assert-minimums")] = False,
    corpus: CorpusOption = None,
) -> None:
    """Print document and unit counts for each city and optionally check minimums."""
    with db.connect(corpus_database_url(use_corpus(corpus))) as conn:
        stats = corpus_stats(conn)
    for city_id in sorted(stats.per_city):
        city = stats.per_city[city_id]
        typer.echo(
            f"{city_id} documents={city.documents} agenda={city.agenda} minutes={city.minutes} "
            f"transcript={city.transcript} pages={city.pages} segments={city.segments} "
            f"ocr_pages={city.ocr_pages} bytes={city.bytes}"
        )
    typer.echo(f"total pages={stats.total_pages}")
    if check_minimums:
        assert_minimums(stats)


@corpus_app.command("parse-quality")
def corpus_parse_quality(corpus: CorpusOption = None) -> None:
    """Print page quality and OCR counts for each city."""
    with db.connect(corpus_database_url(use_corpus(corpus))) as conn:
        rows = quality.parse_quality_report(conn)
    for row in rows:
        confidence = (
            f"{row.mean_ocr_confidence:.3f}" if row.mean_ocr_confidence is not None else "n/a"
        )
        score = f"{row.parse_quality:.3f}" if row.parse_quality is not None else "n/a"
        typer.echo(
            f"{row.city_id} pages={row.pages} ocr_pages={row.ocr_pages} "
            f"empty_pages={row.empty_pages} mean_ocr_confidence={confidence} parse_quality={score}"
        )
    if any(row.parse_quality is None or row.parse_quality < 0.90 for row in rows):
        raise typer.Exit(1)


CityOption = Annotated[str | None, typer.Option()]
BodyOption = Annotated[str | None, typer.Option()]
DateOption = Annotated[datetime | None, typer.Option(formats=["%Y-%m-%d"])]
KindOption = Annotated[str | None, typer.Option()]

PayloadFile = Annotated[Path, typer.Option("--payload-file", exists=True, dir_okay=False)]


def _read_payload(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise LabelValidationError([f"payload: {err}"]) from err
    if not isinstance(payload, dict):
        raise LabelValidationError(["payload: must be a JSON object"])
    return payload


def _label_input(**fields: object) -> LabelIn:
    try:
        return LabelIn.model_validate(fields)
    except PayloadError as err:
        raise LabelValidationError(
            [
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in err.errors()
            ]
        ) from err


@labels_app.command("add")
def labels_add(
    type: Annotated[LabelType, typer.Option("--type")],
    city: Annotated[str, typer.Option()],
    payload_file: PayloadFile,
    author: Annotated[Literal["agent", "human"], typer.Option()] = "agent",
    note: Annotated[str, typer.Option()] = "",
    corpus: CorpusOption = None,
) -> None:
    label = _label_input(
        type=type, city=city, author=author, note=note, payload=_read_payload(payload_file)
    )
    with db.connect(corpus_database_url(use_corpus(corpus))) as conn:
        created = store.create(conn, label, labels_dir=LABELS_DIR)
    typer.echo(created.id)


@labels_app.command("update")
def labels_update(
    id: str,
    payload_file: PayloadFile,
    note: Annotated[str | None, typer.Option()] = None,
    corpus: CorpusOption = None,
) -> None:
    payload = _read_payload(payload_file)
    with db.connect(corpus_database_url(use_corpus(corpus))) as conn:
        existing = store.get(conn, id)
        label = _label_input(
            type=existing.type,
            city=existing.city,
            author=existing.author,
            note=existing.note if note is None else note,
            payload=payload,
        )
        store.update(conn, id, label, labels_dir=LABELS_DIR)
    typer.echo(f"updated {id}")


@labels_app.command("delete")
def labels_delete(id: str, corpus: CorpusOption = None) -> None:
    with db.connect(corpus_database_url(use_corpus(corpus))) as conn:
        store.delete(conn, id, labels_dir=LABELS_DIR)
    typer.echo(f"deleted {id}")


@labels_app.command("show")
def labels_show(id: str, corpus: CorpusOption = None) -> None:
    with db.connect(corpus_database_url(use_corpus(corpus))) as conn:
        label = store.get(conn, id)
    typer.echo(label.model_dump_json(indent=2))


@labels_app.command("list")
def labels_list(
    type: Annotated[LabelType | None, typer.Option("--type")] = None,
    city: CityOption = None,
    corpus: CorpusOption = None,
) -> None:
    with db.connect(corpus_database_url(use_corpus(corpus))) as conn:
        labels = store.list_labels(conn, type, city)
    for label in labels:
        typer.echo(
            f"{label.id} {label.type} {label.city} reviewed={str(label.human_reviewed).lower()}"
        )


@labels_app.command("page")
def labels_page(
    document_id: str,
    unit_index: int,
    find: Annotated[str | None, typer.Option()] = None,
    corpus: CorpusOption = None,
) -> None:
    with db.connect(corpus_database_url(use_corpus(corpus))) as conn:
        row = conn.execute(
            "SELECT u.text FROM units u JOIN documents d ON d.id = u.document_id "
            "WHERE u.document_id = %s AND u.unit_index = %s AND d.status = 'downloaded'",
            (document_id, unit_index),
        ).fetchone()
    if row is None:
        raise NotFoundError(f"unknown unit {document_id} {unit_index}")
    text: str = row[0]
    if find is None:
        typer.echo(text)
        return
    if not find:
        raise LabelValidationError(["find must not be empty"])
    start = text.find(find)
    if start == -1:
        raise typer.Exit(1)
    while start != -1:
        end = start + len(find)
        typer.echo(f"start={start} end={end} text={text[start:end]}")
        start = text.find(find, end)


@labels_app.command("text-search")
def labels_text_search(
    query: str, city: Annotated[str, typer.Option()], corpus: CorpusOption = None
) -> None:
    with db.connect(corpus_database_url(use_corpus(corpus))) as conn:
        result = store.text_search(conn, city, query)
    typer.echo(f"unit_count={result.unit_count}")
    for document, index, snippet in result.matches:
        typer.echo(f"{document} {index} {snippet}")


@labels_app.command("sample-meetings")
def labels_sample_meetings(
    city: Annotated[str, typer.Option()], corpus: CorpusOption = None
) -> None:
    with db.connect(corpus_database_url(use_corpus(corpus))) as conn:
        ids = store.sample_meetings(conn, city)
        agendas = dict(
            conn.execute(
                "SELECT meeting_id, id FROM documents WHERE meeting_id = ANY(%s) "
                "AND kind = 'agenda' AND status = 'downloaded'",
                (ids,),
            ).fetchall()
        )
    for meeting in ids:
        typer.echo(f"{meeting} {agendas[meeting]}")


@labels_app.command("progress")
def labels_progress(corpus: CorpusOption = None) -> None:
    with db.connect(corpus_database_url(use_corpus(corpus))) as conn:
        labels = store.list_labels(conn)
        doc_kinds = export.document_kinds(conn)
    for label_type in export.LABEL_FILES:
        typer.echo(f"{label_type} count={sum(label.type == label_type for label in labels)}")
        for message in check(label_type, labels, LABEL_RULES, doc_kinds):
            typer.echo(message)


@labels_app.command("export")
def labels_export(
    type: Annotated[LabelType | None, typer.Option("--type")] = None,
    everything: Annotated[bool, typer.Option("--all")] = False,
    corpus: CorpusOption = None,
) -> None:
    if (type is None) == (not everything):
        raise typer.BadParameter("choose either --type or --all")
    types = list(export.LABEL_FILES) if everything else [type]
    with db.connect(corpus_database_url(use_corpus(corpus))) as conn:
        for label_type in types:
            assert label_type is not None
            path = export.export_type(conn, label_type, LABELS_DIR, LABEL_RULES)
            count = len(store.list_labels(conn, label_type))
            typer.echo(f"wrote {path.as_posix()} ({count} labels)")


@labels_app.command("import")
def labels_import(corpus: CorpusOption = None) -> None:
    with db.connect(corpus_database_url(use_corpus(corpus))) as conn:
        count = export.import_all(conn, LABELS_DIR)
    typer.echo(f"imported {count} labels")


@labels_app.command("backup")
def labels_backup(corpus: CorpusOption = None) -> None:
    with db.connect(corpus_database_url(use_corpus(corpus))) as conn:
        export.backup(conn)
        count = len(store.list_labels(conn))
    typer.echo(f"backed up {count} labels")


@labels_app.command("restore")
def labels_restore(corpus: CorpusOption = None) -> None:
    with db.connect(corpus_database_url(use_corpus(corpus))) as conn:
        count = export.restore(conn)
    typer.echo(f"restored {count} labels")


@labels_app.command("freeze")
def labels_freeze() -> None:
    export.freeze(LABELS_DIR, LABEL_RULES)
    typer.echo("frozen")


@labels_app.command("amend")
def labels_amend(
    file: Annotated[str, typer.Option()], reason: Annotated[str, typer.Option()]
) -> None:
    export.amend(file, reason, LABELS_DIR)
    typer.echo(f"amended {file}")


def as_filters(
    city: str | None,
    body: str | None,
    date_from: datetime | None,
    date_to: datetime | None,
    kind: str | None,
) -> Filters:
    return Filters(
        city=city,
        date_from=date_from.date() if date_from else None,
        date_to=date_to.date() if date_to else None,
        body=body,
        kind=kind,
    )


@app.command("search")
def search_command(
    query: str,
    city: CityOption = None,
    body: BodyOption = None,
    date_from: DateOption = None,
    date_to: DateOption = None,
    kind: KindOption = None,
    mode: Annotated[SearchMode, typer.Option()] = "hybrid_rerank",
    chunker: Annotated[Chunker, typer.Option()] = "item",
    k: Annotated[int, typer.Option()] = 10,
    corpus: CorpusOption = None,
) -> None:
    """Search the indexed records and print one line per hit."""
    used = use_corpus(corpus)
    if used != "fixture":
        store.require_frozen(LABELS_DIR)
    filters = as_filters(city, body, date_from, date_to, kind)
    with db.connect(corpus_database_url(used)) as conn:
        hits = search(conn, query, filters, mode=mode, chunker=chunker, k=k)
        names = city_names(conn)
    for hit in hits:
        span = hit.spans[0]
        label = span_label(span.unit_kind, span.unit_index, hit.start_ms)
        heading = hit_heading(names[hit.city_id], hit)
        typer.echo(f"{hit.rank} {hit.score:.3f} {hit.document_id} {label} {heading}")


@app.command("ask")
def ask_command(
    question: str,
    city: CityOption = None,
    body: BodyOption = None,
    date_from: DateOption = None,
    date_to: DateOption = None,
    kind: KindOption = None,
    pipeline_name: Annotated[Pipeline, typer.Option("--pipeline")] = "final",
    no_controls: Annotated[bool, typer.Option("--no-controls")] = False,
    corpus: CorpusOption = None,
) -> None:
    """Answer a question from the indexed records, with citations."""
    used = use_corpus(corpus)
    if used != "fixture":
        store.require_frozen(LABELS_DIR)
    filters = as_filters(city, body, date_from, date_to, kind)
    with db.connect(corpus_database_url(used)) as conn:
        result = answer(
            conn,
            question,
            filters,
            pipeline=pipeline_name,
            controls=not no_controls,
        )
    if result.declined:
        typer.echo(f"declined: {result.decline_reason}")
        return
    for sentence in result.sentences:
        typer.echo(sentence.text + " " + "".join(f"[{n}]" for n in sentence.citations))
    for citation in result.citations:
        label = span_label(citation.unit_kind, citation.unit_index, citation.start_ms)
        typer.echo(f"[{citation.number}] {citation.document_id} {label}")


@app.command("serve")
def serve(port: Annotated[int | None, typer.Option()] = None, corpus: CorpusOption = None) -> None:
    """Run the HTTP API on 127.0.0.1."""
    use_corpus(corpus)
    port = port or get_settings().api_port
    get_logger("api").info(
        f"api listening on http://127.0.0.1:{port}", extra={"event": "api_start"}
    )
    uvicorn.run(create_app(), host="127.0.0.1", port=port, log_level="warning")
