import os
from typing import Annotated, Any
from urllib.parse import urlsplit

import typer
from typer.core import TyperGroup

from minutes import db, probe
from minutes.config import Corpus, corpus_database_url, get_settings
from minutes.errors import ConfigError, GateStaleError, LabelRuleError, LLMQuotaError, MinutesError
from minutes.ingest import pipeline
from minutes.log import bind_correlation_id, configure_logging, new_correlation_id

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
fixtures_app = typer.Typer(no_args_is_help=True)
app.add_typer(ingest_app, name="ingest")
app.add_typer(fixtures_app, name="fixtures")


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


@fixtures_app.command("load")
def fixtures_load() -> None:
    """Rebuild the test database from the tracked test corpus, with stub models."""
    os.environ.update(MINUTES_CORPUS="fixture", MINUTES_MODELS_MODE="stub", MINUTES_LLM_MODE="stub")
    get_settings.cache_clear()
    count = pipeline.load_fixture_corpus(get_settings().test_database_url)
    typer.echo(f"loaded {count} documents")
