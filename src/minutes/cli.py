import os
from datetime import datetime
from typing import Annotated, Any, NoReturn
from urllib.parse import urlsplit

import typer
import uvicorn
from typer.core import TyperGroup

from minutes import corpus, db, probe, queue, worker
from minutes.answer.pipeline import answer
from minutes.api.app import create_app
from minutes.api.schemas import Chunker, Pipeline, SearchMode
from minutes.config import Corpus, corpus_database_url, get_settings
from minutes.errors import ConfigError, GateStaleError, LabelRuleError, LLMQuotaError, MinutesError
from minutes.ingest import pipeline, quality
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
            typer.echo(f"error: {err}", err=True)
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
app.add_typer(ingest_app, name="ingest")
app.add_typer(worker_app, name="worker")
app.add_typer(jobs_app, name="jobs")
app.add_typer(fixtures_app, name="fixtures")
app.add_typer(corpus_app, name="corpus")


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
    filters = as_filters(city, body, date_from, date_to, kind)
    with db.connect(corpus_database_url(use_corpus(corpus))) as conn:
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
    filters = as_filters(city, body, date_from, date_to, kind)
    with db.connect(corpus_database_url(use_corpus(corpus))) as conn:
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
