import json
import logging
import os
import random
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import httpx
import psycopg
import pytest

from minutes import config
from minutes.errors import (
    BudgetExceeded,
    LLMOutputError,
    LLMQuotaError,
    LLMReplayMiss,
    LLMTransportError,
)
from minutes.llm import Budget
from minutes.llm.cache import DiskCache, ReplayStore, cache_key
from minutes.llm.client import ThinClient
from minutes.llm.client import _record_call as record_call
from minutes.llm.prices import cost_usd
from minutes.log import JsonFormatter
from minutes.tracing import configure_tracing, get_test_exporter

KEY = "unit-test-key-never-log"
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
    "additionalProperties": False,
}


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
def isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    for name in tuple(os.environ):
        if name.startswith("MINUTES_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MINUTES_DATA_DIR", "data")
    monkeypatch.setenv("MINUTES_LLM_MODE", "live")
    monkeypatch.setenv("MINUTES_LLM_BASE_URL", "http://proxy.test/v1")
    monkeypatch.setenv("MINUTES_LLM_API_KEY", KEY)
    config.get_settings.cache_clear()
    from minutes.llm import client as client_module

    monkeypatch.setattr(client_module, "_LAST_START", None)
    monkeypatch.setattr(client_module, "_record_call", lambda **values: None)
    yield
    config.get_settings.cache_clear()


def make_client(
    handler: httpx.MockTransport | httpx.BaseTransport,
    tmp_path: Path,
    clock: FakeClock,
    *,
    cache: DiskCache | ReplayStore | None = None,
) -> ThinClient:
    http = httpx.Client(
        transport=handler,
        timeout=httpx.Timeout(120.0, connect=10.0),
    )
    return ThinClient(
        http=http,
        cache=cache,
        clock=clock,
        wait=clock.wait,
    )


def success(content: str = "ok", *, usage: dict[str, int] | None = None) -> httpx.Response:
    body: dict[str, object] = {"choices": [{"message": {"content": content}}]}
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(200, json=body)


def test_request_body_matches_spec(tmp_path: Path) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return success('{"answer":5}', usage={"prompt_tokens": 4, "completion_tokens": 3})

    clock = FakeClock()
    client = make_client(httpx.MockTransport(handler), tmp_path, clock)
    client.chat(
        purpose="synth",
        messages=[{"role": "user", "content": "add these"}],
        schema=SCHEMA,
        schema_name="answer",
    )
    request = captured[0]
    body = json.loads(request.content)
    assert list(body) == [
        "model",
        "messages",
        "max_completion_tokens",
        "reasoning_effort",
        "response_format",
    ]
    assert "temperature" not in body
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    assert str(request.url) == "http://proxy.test/v1/chat/completions"


def test_cache_hit_makes_no_request_and_marks_cached(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return success()

    clock = FakeClock()
    client = make_client(
        httpx.MockTransport(handler), tmp_path, clock, cache=DiskCache(tmp_path / "cache")
    )
    messages = [{"role": "user", "content": "same"}]
    first = client.chat(purpose="synth", messages=messages)
    second = client.chat(purpose="synth", messages=messages)
    assert first.cached is False
    assert second.cached is True
    assert calls == 1


def test_sample_index_changes_cache_key(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return success()

    clock = FakeClock()
    root = tmp_path / "cache"
    client = make_client(httpx.MockTransport(handler), tmp_path, clock, cache=DiskCache(root))
    messages = [{"role": "user", "content": "sample"}]
    client.chat(purpose="synth", messages=messages, sample_index=0)
    client.chat(purpose="synth", messages=messages, sample_index=1)
    assert calls == 2
    assert len(list(root.rglob("*.json"))) == 2


def test_retry_on_500_three_times_then_transport_error(tmp_path: Path) -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(500, text="unavailable")

    clock = FakeClock()
    client = make_client(httpx.MockTransport(handler), tmp_path, clock)
    messages = [{"role": "user", "content": "retry"}]
    with pytest.raises(LLMTransportError):
        client.chat(purpose="synth", messages=messages)
    key = cache_key(
        "gpt-6.1-sol",
        messages,
        {
            "max_completion_tokens": 900,
            "reasoning_effort": "low",
            "schema_name": "output",
            "schema": None,
        },
        0,
    )
    expected = [
        2**attempt + random.Random(int(key[:8], 16) + attempt).uniform(0.0, 1.0)
        for attempt in range(1, 4)
    ]
    assert requests == 4
    assert clock.waits == expected


def test_429_uses_retry_after_header(tmp_path: Path) -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return success()

    clock = FakeClock()
    client = make_client(httpx.MockTransport(handler), tmp_path, clock)
    client.chat(purpose="synth", messages=[{"role": "user", "content": "quota"}])
    assert requests == 2
    assert clock.waits == [7.0]


def test_429_without_header_backs_off_60_120_240_480_960_then_quota_error(
    tmp_path: Path,
) -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(429)

    clock = FakeClock()
    client = make_client(httpx.MockTransport(handler), tmp_path, clock)
    with pytest.raises(LLMQuotaError):
        client.chat(purpose="synth", messages=[{"role": "user", "content": "quota"}])
    assert requests == 6
    assert clock.waits == [60.0, 120.0, 240.0, 480.0, 960.0]


def test_400_is_not_retried(tmp_path: Path) -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(400, text="bad request")

    clock = FakeClock()
    client = make_client(httpx.MockTransport(handler), tmp_path, clock)
    with pytest.raises(LLMTransportError, match=r"^http 400"):
        client.chat(purpose="synth", messages=[{"role": "user", "content": "bad"}])
    assert requests == 1


def test_invalid_json_triggers_one_corrective_retry(tmp_path: Path) -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return success("not-json", usage={"prompt_tokens": 7, "completion_tokens": 2})
        return success('{"answer":5}', usage={"prompt_tokens": 11, "completion_tokens": 3})

    clock = FakeClock()
    root = tmp_path / "cache"
    client = make_client(httpx.MockTransport(handler), tmp_path, clock, cache=DiskCache(root))
    messages = [{"role": "user", "content": "answer"}]
    result = client.chat(
        purpose="synth",
        messages=messages,
        schema=SCHEMA,
        schema_name="answer",
    )
    assert result.parsed == {"answer": 5}
    assert result.prompt_tokens == 18
    assert result.completion_tokens == 5
    assert result.cost_usd == cost_usd(result.model, 18, 5)
    assert len(bodies) == 2
    corrected_messages = bodies[1]["messages"]
    assert isinstance(corrected_messages, list)
    assert corrected_messages[-1] == {
        "role": "user",
        "content": "Your previous reply was not valid JSON for the schema. Reply with JSON only.",
    }
    repeated = client.chat(
        purpose="synth",
        messages=messages,
        schema=SCHEMA,
        schema_name="answer",
    )
    assert repeated.cached is True
    assert repeated.prompt_tokens == 18
    assert repeated.completion_tokens == 5
    assert repeated.cost_usd == cost_usd(repeated.model, 18, 5)
    assert len(bodies) == 2
    assert len(list(root.rglob("*.json"))) == 2


def test_invalid_twice_raises_output_error(tmp_path: Path) -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return success("not-json")

    clock = FakeClock()
    client = make_client(httpx.MockTransport(handler), tmp_path, clock)
    with pytest.raises(LLMOutputError):
        client.chat(
            purpose="synth",
            messages=[{"role": "user", "content": "answer"}],
            schema=SCHEMA,
        )
    assert requests == 2


def test_schema_violation_counts_as_invalid(tmp_path: Path) -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return success('{"answer":"five"}' if requests == 1 else '{"answer":5}')

    clock = FakeClock()
    client = make_client(httpx.MockTransport(handler), tmp_path, clock)
    result = client.chat(
        purpose="synth",
        messages=[{"role": "user", "content": "answer"}],
        schema=SCHEMA,
    )
    assert result.parsed == {"answer": 5}
    assert requests == 2


def test_daily_cap_raises_quota_error_before_sending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from minutes import config

    monkeypatch.setenv("MINUTES_LLM_DAILY_CAP", "1")
    config.get_settings.cache_clear()
    usage_path = tmp_path / "data" / "cache" / "llm_usage.json"
    usage_path.parent.mkdir(parents=True)
    usage_path.write_text(
        json.dumps({"date": datetime.now(UTC).date().isoformat(), "requests": 1}),
        encoding="utf-8",
    )
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return success()

    clock = FakeClock()
    client = make_client(httpx.MockTransport(handler), tmp_path, clock)
    with pytest.raises(LLMQuotaError, match="daily cap of 1 uncached requests reached"):
        client.chat(purpose="synth", messages=[{"role": "user", "content": "cap"}])
    assert requests == 0


def test_rate_limit_spaces_request_starts_by_3_seconds(tmp_path: Path) -> None:
    clock = FakeClock()
    starts: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        starts.append(clock())
        return success()

    client = make_client(httpx.MockTransport(handler), tmp_path, clock)
    client.chat(purpose="synth", messages=[{"role": "user", "content": "one"}])
    clock.now += 0.75
    client.chat(purpose="synth", messages=[{"role": "user", "content": "two"}])
    assert clock.waits == [2.25]
    assert starts[1] - starts[0] == 3.0


def test_replay_miss_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINUTES_LLM_MODE", "replay")
    config.get_settings.cache_clear()

    def unexpected_request(request: httpx.Request) -> httpx.Response:
        pytest.fail("replay sent an http request")

    client = make_client(
        httpx.MockTransport(unexpected_request),
        tmp_path,
        FakeClock(),
        cache=ReplayStore(tmp_path / "missing.jsonl.gz"),
    )
    with pytest.raises(LLMReplayMiss):
        client.chat(purpose="synth", messages=[{"role": "user", "content": "missing"}])


def test_budget_exceeded_before_request(tmp_path: Path) -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return success()

    client = make_client(httpx.MockTransport(handler), tmp_path, FakeClock())
    with pytest.raises(BudgetExceeded):
        client.chat(
            purpose="synth",
            messages=[{"role": "user", "content": "expensive"}],
            budget=Budget(0.0001),
        )
    assert requests == 0


def test_usage_missing_falls_back_to_estimate(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    clock = FakeClock()
    client = make_client(httpx.MockTransport(lambda request: success("five")), tmp_path, clock)
    with caplog.at_level(logging.WARNING, logger="minutes.llm"):
        result = client.chat(purpose="synth", messages=[{"role": "user", "content": "12345"}])
    assert result.prompt_tokens == 2
    assert result.completion_tokens == 1
    assert any(getattr(record, "event", "") == "usage_missing" for record in caplog.records)


def test_api_key_never_appears_in_logs_or_cache(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    from minutes.llm import client as client_module

    monkeypatch.setattr(client_module, "_record_call", record_call)
    monkeypatch.setattr(logging.getLogger("minutes"), "propagate", True)

    def unavailable_database(*args: object, **kwargs: object) -> None:
        raise psycopg.OperationalError("database unavailable")

    monkeypatch.setattr(psycopg, "connect", unavailable_database)
    configure_tracing(test=True)
    exporter = get_test_exporter()
    assert exporter is not None
    clock = FakeClock()
    root = tmp_path / "data"

    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["messages"][0]["content"] == "fail":
            raise httpx.ConnectError(f"connection rejected {KEY}")
        return success()

    client = make_client(
        httpx.MockTransport(handler),
        tmp_path,
        clock,
        cache=DiskCache(root / "cache" / "llm"),
    )
    with caplog.at_level(logging.INFO, logger="minutes"):
        client.chat(purpose="synth", messages=[{"role": "user", "content": "private"}])
        with pytest.raises(LLMTransportError):
            try:
                client.chat(purpose="synth", messages=[{"role": "user", "content": "fail"}])
            except LLMTransportError:
                logging.getLogger("minutes.llm").exception("request failed")
                raise
    formatter = JsonFormatter()
    structured = [json.loads(formatter.format(record)) for record in caplog.records]
    calls = [row for row in structured if row["event"] == "llm_call"]
    assert len(calls) == 1
    assert calls[0]["fields"]["purpose"] == "synth"
    assert KEY not in json.dumps(structured)
    assert KEY not in caplog.text
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    assert any(event.name == "exception" for span in spans for event in span.events)
    assert all(KEY not in span.to_json() for span in spans)
    files = [path for path in root.rglob("*") if path.is_file()]
    assert root / "cache" / "llm_usage.json" in files
    assert any(path.parent.parent.name == "llm" for path in files)
    assert all(KEY.encode() not in path.read_bytes() for path in files)


@pytest.mark.parametrize("kind", ["transport", "response", "schema"])
def test_error_tracebacks_hide_key(tmp_path: Path, kind: str) -> None:
    import traceback

    def handler(request: httpx.Request) -> httpx.Response:
        if kind == "transport":
            raise httpx.ConnectError(KEY)
        if kind == "response":
            return httpx.Response(200, json={KEY: "invalid"})
        return success(json.dumps({"answer": KEY}))

    client = make_client(httpx.MockTransport(handler), tmp_path, FakeClock())
    with pytest.raises((LLMTransportError, LLMOutputError)) as caught:
        client.chat(purpose="synth", messages=[], schema=SCHEMA)
    assert caught.value.__cause__ is None
    assert KEY not in "".join(traceback.format_exception(caught.value))


def test_empty_key_does_not_change_error_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = make_client(
        httpx.MockTransport(lambda request: httpx.Response(400, text="bad request")),
        tmp_path,
        FakeClock(),
    )
    monkeypatch.setattr(client.settings, "llm_api_key", "")
    with pytest.raises(LLMTransportError, match=r"^http 400: bad request$"):
        client.chat(purpose="synth", messages=[])
