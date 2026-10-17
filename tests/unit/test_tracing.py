import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import httpx
import psycopg
import pytest
from opentelemetry.trace import StatusCode

from minutes import config
from minutes.errors import LLMTransportError
from minutes.llm.cache import DiskCache
from minutes.llm.client import ThinClient
from minutes.llm.prices import cost_usd
from minutes.llm.prompts import SYNTH_SYSTEM, SYNTH_USER
from minutes.llm.stub import StubClient
from minutes.log import bind_correlation_id
from minutes.tracing import configure_tracing, get_test_exporter

KEY = "trace-test-key"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.waits: list[float] = []

    def __call__(self) -> float:
        return self.now

    def wait(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds


@pytest.fixture(autouse=True)
def isolated_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest
) -> Iterator[None]:
    from minutes.llm import client as client_module

    monkeypatch.setattr(client_module, "_LAST_START", None)
    if request.node.get_closest_marker("integration") is not None:
        yield
        return
    for name in tuple(os.environ):
        if name.startswith("MINUTES_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MINUTES_DATA_DIR", "data")
    monkeypatch.setenv("MINUTES_LLM_MODE", "live")
    monkeypatch.setenv("MINUTES_LLM_BASE_URL", "http://proxy.test/v1")
    monkeypatch.setenv("MINUTES_LLM_API_KEY", KEY)
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


def test_llm_call_emits_span_with_token_and_cost_attributes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from minutes.llm import client as client_module

    monkeypatch.setattr(client_module, "_record_call", lambda **values: None)
    configure_tracing(test=True)
    exporter = get_test_exporter()
    assert exporter is not None
    http = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "hello"}}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 2},
                },
            )
        ),
    )
    clock = FakeClock()
    client = ThinClient(http=http, clock=clock, wait=clock.wait)
    client.chat(purpose="synth", messages=[{"role": "user", "content": "hello"}])
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "llm.chat"
    attrs = span.attributes
    assert attrs is not None
    assert attrs["gen_ai.system"] == "openai"
    assert attrs["gen_ai.request.model"] == config.load_models().strong
    assert attrs["gen_ai.usage.input_tokens"] == 12
    assert attrs["gen_ai.usage.output_tokens"] == 2
    assert isinstance(attrs["minutes.cost_usd"], float)
    assert attrs["minutes.purpose"] == "synth"
    assert attrs["minutes.cache_hit"] is False
    assert attrs["minutes.sample_index"] == 0
    assert "minutes.correlation_id" in attrs
    assert isinstance(attrs["minutes.latency_ms"], int)


def test_cached_call_emits_span_with_cache_hit_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from minutes.llm import client as client_module

    monkeypatch.setattr(client_module, "_record_call", lambda **values: None)
    configure_tracing(test=True)
    exporter = get_test_exporter()
    assert exporter is not None
    http = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "hello"}}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                },
            )
        ),
    )
    clock = FakeClock()
    client = ThinClient(
        http=http, cache=DiskCache(tmp_path / "cache"), clock=clock, wait=clock.wait
    )
    messages = [{"role": "user", "content": "hello"}]
    client.chat(purpose="synth", messages=messages)
    client.chat(purpose="synth", messages=messages)
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    attrs = spans[1].attributes
    assert attrs is not None
    assert attrs["minutes.cache_hit"] is True


def test_error_sets_span_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from minutes.llm import client as client_module

    monkeypatch.setattr(client_module, "_record_call", lambda **values: None)
    configure_tracing(test=True)
    exporter = get_test_exporter()
    assert exporter is not None
    http = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(400, text="bad")),
    )
    clock = FakeClock()
    client = ThinClient(http=http, clock=clock, wait=clock.wait)
    bind_correlation_id("failed-call")
    with pytest.raises(LLMTransportError):
        client.chat(purpose="synth", messages=[{"role": "user", "content": "bad"}])
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == "LLMTransportError"
    attrs = span.attributes
    assert attrs is not None
    assert attrs["minutes.sample_index"] == 0
    assert attrs["minutes.correlation_id"] == "failed-call"


@pytest.mark.integration
def test_stub_client_writes_llm_calls_row(fixture_corpus: str) -> None:
    configure_tracing(test=True)
    correlation_id = uuid.uuid4().hex
    bind_correlation_id(correlation_id)
    result = StubClient().chat(
        purpose="synth",
        messages=[
            {"role": "system", "content": SYNTH_SYSTEM},
            {
                "role": "user",
                "content": SYNTH_USER.format(
                    question="what happened?",
                    sources=(
                        "[1] Birch City Council, 2024-04-09, minutes, p. 2\n"
                        "The council repaired a sidewalk."
                    ),
                ),
            },
        ],
    )
    with psycopg.connect(fixture_corpus, autocommit=True, connect_timeout=5) as conn:
        rows = conn.execute(
            "SELECT purpose, cached, cost_usd FROM llm_calls WHERE correlation_id = %s",
            (correlation_id,),
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "synth"
    assert rows[0][1] is False
    assert float(rows[0][2]) == cost_usd(
        result.model, result.prompt_tokens, result.completion_tokens
    )
