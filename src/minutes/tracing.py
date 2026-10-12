from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Span, Status, StatusCode

from minutes.config import get_settings

_provider: TracerProvider | None = None
_test_exporter: InMemorySpanExporter | None = None
_global_provider_set = False


def configure_tracing(test: bool = False) -> None:
    global _provider, _test_exporter, _global_provider_set

    settings = get_settings()
    provider = TracerProvider(resource=Resource.create({"service.name": "minutes"}))
    _test_exporter = None
    if test:
        _test_exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(_test_exporter))
    elif settings.otel_endpoint:
        # the exporter pulls in protobuf and requests, so only load it when it is used
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        endpoint = settings.otel_endpoint.rstrip("/") + "/v1/traces"
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    _provider = provider
    if not _global_provider_set:
        trace.set_tracer_provider(provider)
        _global_provider_set = True


def get_test_exporter() -> InMemorySpanExporter | None:
    return _test_exporter


@contextmanager
def llm_span(purpose: str, model: str) -> Iterator[Span]:
    if _provider is None:
        configure_tracing()
    assert _provider is not None
    span = _provider.get_tracer("minutes.llm").start_span("llm.chat")
    span.set_attribute("gen_ai.system", "openai")
    span.set_attribute("gen_ai.request.model", model)
    span.set_attribute("minutes.purpose", purpose)
    with trace.use_span(span, end_on_exit=False):
        try:
            yield span
        except Exception as err:
            span.set_status(Status(StatusCode.ERROR, type(err).__name__))
            span.record_exception(err)
            raise
        finally:
            span.end()
