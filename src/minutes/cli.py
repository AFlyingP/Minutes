from typing import Any

import typer
from typer.core import TyperGroup

from minutes import probe
from minutes.config import get_settings
from minutes.errors import ConfigError, GateStaleError, LabelRuleError, LLMQuotaError, MinutesError
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
app.add_typer(probe_app, name="probe")


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
