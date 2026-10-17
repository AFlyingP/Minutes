import json
import math
import os
import random
import tempfile
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
import jsonschema
import psycopg
from opentelemetry.trace import Span

from minutes.config import PurposeConfig, corpus_database_url, get_settings, load_models
from minutes.errors import (
    BudgetExceeded,
    ConfigError,
    LLMOutputError,
    LLMQuotaError,
    LLMReplayMiss,
    LLMTransportError,
)
from minutes.llm import Budget, ChatResult
from minutes.llm.cache import DiskCache, ReplayStore, cache_key
from minutes.llm.prices import cost_usd, estimate_usd
from minutes.log import get_correlation_id, get_logger
from minutes.tracing import llm_span

_REQUEST_SLOTS = threading.Semaphore(8)
_START_LOCK = threading.Lock()
_USAGE_LOCK = threading.Lock()
_LAST_START: float | None = None
PROXY_TIMEOUT = httpx.Timeout(600.0, connect=10.0)
_HTTP_LOCK = threading.Lock()
_HTTP: httpx.Client | None = None
_CORRECTION = "Your previous reply was not valid JSON for the schema. Reply with JSON only."
PER_CALL_CEILINGS = {
    "extract": 0.08,
    "synth": 0.03,
    "judge": 0.004,
    "grade": 0.01,
    "agent": 0.06,
}
_LOG = get_logger("llm")


def _default_wait(seconds: float) -> None:
    threading.Event().wait(seconds)


def _purpose_config(purpose: str) -> tuple[str | None, int, str, float | None]:
    if purpose == "probe":
        return None, 200, "low", None
    models = load_models()
    try:
        config: PurposeConfig = models.purposes[purpose]
    except KeyError as err:
        raise ConfigError(f"unknown llm purpose {purpose}") from err
    model = models.strong if config.model == "strong" else models.cheap
    return model, config.max_tokens, config.reasoning_effort, PER_CALL_CEILINGS[purpose]


def request_cache_key(
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    reasoning_effort: str,
    schema: dict[str, object] | None,
    schema_name: str,
    sample_index: int,
) -> str:
    """Build the cache key for a resolved chat request."""
    params: dict[str, object] = {
        "max_completion_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
        "schema_name": schema_name,
        "schema": schema,
    }
    return cache_key(model, messages, params, sample_index)


@lru_cache(maxsize=32)
def _validator(schema_json: str) -> Any:
    """A checked validator per schema; checking the schema on every reply was slow."""
    schema = json.loads(schema_json)
    validator_class = jsonschema.validators.validator_for(schema)
    validator_class.check_schema(schema)
    return validator_class(schema)


CALL_INSERT = (
    "INSERT INTO llm_calls "
    "(correlation_id, purpose, model, cache_key, cached, prompt_tokens, "
    "completion_tokens, cost_usd, latency_ms) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
)
# one call log connection per thread and database: opening one per call took longer
# than a stub call itself
_call_log = threading.local()


def _insert_call_row(url: str, params: tuple[object, ...]) -> None:
    connections: dict[str, Any] = vars(_call_log).setdefault("connections", {})
    conn = connections.pop(url, None)
    if conn is not None:
        try:
            conn.execute(CALL_INSERT, params)
            connections[url] = conn
            return
        except psycopg.Error:
            # a database reset ends open sessions, so the row goes through a new connection
            conn.close()
    conn = psycopg.connect(url, autocommit=True, connect_timeout=10)
    conn.execute(CALL_INSERT, params)
    connections[url] = conn


def _record_call(
    *,
    purpose: str,
    model: str,
    key: str,
    cached: bool,
    prompt_tokens: int,
    completion_tokens: int,
    cost: float,
    latency_ms: int,
) -> None:
    settings = get_settings()
    try:
        _insert_call_row(
            corpus_database_url(settings.corpus),
            (
                get_correlation_id(),
                purpose,
                model,
                key,
                cached,
                prompt_tokens,
                completion_tokens,
                cost,
                latency_ms,
            ),
        )
    except psycopg.Error as err:
        _LOG.warning(
            "could not record llm call",
            extra={"event": "llm_call_record_failed", "fields": {"error": str(err)}},
        )
    _LOG.info(
        "llm call",
        extra={
            "event": "llm_call",
            "fields": {
                "purpose": purpose,
                "model": model,
                "cached": cached,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": cost,
                "latency_ms": latency_ms,
            },
        },
    )


def record_completed_call(
    span: Span,
    *,
    purpose: str,
    model: str,
    key: str,
    cached: bool,
    prompt_tokens: int,
    completion_tokens: int,
    cost: float,
    latency_ms: int,
) -> None:
    span.set_attribute("gen_ai.usage.input_tokens", prompt_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", completion_tokens)
    span.set_attribute("minutes.cost_usd", cost)
    span.set_attribute("minutes.cache_hit", cached)
    span.set_attribute("minutes.latency_ms", latency_ms)
    _record_call(
        purpose=purpose,
        model=model,
        key=key,
        cached=cached,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost=cost,
        latency_ms=latency_ms,
    )


def _shared_http() -> httpx.Client:
    # building a client creates an SSL context, which costs about 10 ms
    global _HTTP
    with _HTTP_LOCK:
        if _HTTP is None or _HTTP.is_closed:
            _HTTP = httpx.Client(timeout=PROXY_TIMEOUT)
        return _HTTP


class ThinClient:
    """OpenAI-compatible chat client with cache, quotas, retries, and accounting."""

    def __init__(
        self,
        http: httpx.Client | None = None,
        cache: DiskCache | ReplayStore | None = None,
        clock: Callable[[], float] = time.monotonic,
        wait: Callable[[float], None] = _default_wait,
    ) -> None:
        self.settings = get_settings()
        self.cache = cache
        self.clock = clock
        self.wait = wait
        self.http = http if http is not None else _shared_http()

    def _key(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        reasoning_effort: str,
        schema: dict[str, object] | None,
        schema_name: str,
        sample_index: int,
    ) -> str:
        return request_cache_key(
            model,
            messages,
            max_tokens,
            reasoning_effort,
            schema,
            schema_name,
            sample_index,
        )

    def _body(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        reasoning_effort: str,
        schema: dict[str, object] | None,
        schema_name: str,
    ) -> dict[str, object]:
        body: dict[str, object] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
        }
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            }
        return body

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        return self.cache.get(key) if self.cache is not None else None

    def _cache_put(self, key: str, model: str, value: dict[str, Any]) -> None:
        if self.cache is not None and not isinstance(self.cache, ReplayStore):
            self.cache.put(
                key,
                {
                    "text": value["text"],
                    "model": model,
                    "prompt_tokens": value["prompt_tokens"],
                    "completion_tokens": value["completion_tokens"],
                    "latency_ms": value["latency_ms"],
                },
            )

    def _reserve_request(self) -> None:
        global _LAST_START

        with _START_LOCK:
            now = self.clock()
            if _LAST_START is not None:
                delay = max(0.0, 1.0 - (now - _LAST_START))
                if delay:
                    self.wait(delay)
            _LAST_START = self.clock()

    def _count_request(self) -> None:
        path = self.settings.data_dir / "cache" / "llm_usage.json"
        today = datetime.now(UTC).date().isoformat()
        with _USAGE_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                current = {"date": today, "requests": 0}
            requests = current["requests"] if current.get("date") == today else 0
            if requests >= self.settings.llm_daily_cap:
                raise LLMQuotaError(
                    f"daily cap of {self.settings.llm_daily_cap} uncached requests reached"
                )
            fd, temporary = tempfile.mkstemp(prefix=".llm-usage-", suffix=".tmp", dir=path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as file:
                    json.dump({"date": today, "requests": requests + 1}, file)
                    file.write("\n")
                os.replace(temporary, path)
            except BaseException:
                Path(temporary).unlink(missing_ok=True)
                raise

    def _request(
        self,
        *,
        key: str,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        reasoning_effort: str,
        schema: dict[str, object] | None,
        schema_name: str,
    ) -> dict[str, Any]:
        body = self._body(model, messages, max_tokens, reasoning_effort, schema, schema_name)
        headers = {"Authorization": f"Bearer {self.settings.llm_api_key}"}
        transport_attempt = 0
        quota_attempt = 0
        with _REQUEST_SLOTS:
            self._reserve_request()
            while True:
                try:
                    self._count_request()
                    response = self.http.post(
                        self.settings.llm_base_url.rstrip("/") + "/chat/completions",
                        json=body,
                        headers=headers,
                    )
                except httpx.TransportError as err:
                    transport_attempt += 1
                    if transport_attempt > 3:
                        message = str(err)
                        if self.settings.llm_api_key:
                            message = message.replace(self.settings.llm_api_key, "***")
                        raise LLMTransportError(f"request failed: {message}") from None
                    jitter = random.Random(int(key[:8], 16) + transport_attempt).uniform(0.0, 1.0)
                    self.wait(2 * 2 ** (transport_attempt - 1) + jitter)
                    continue

                if response.status_code == 429:
                    quota_attempt += 1
                    if quota_attempt > 5:
                        raise LLMQuotaError("http 429 after 5 retries")
                    try:
                        delay = float(response.headers["Retry-After"])
                    except (KeyError, ValueError):
                        delay = 60.0 * 2 ** (quota_attempt - 1)
                    self.wait(delay)
                    continue
                if 500 <= response.status_code <= 599:
                    transport_attempt += 1
                    if transport_attempt > 3:
                        raise LLMTransportError(f"http {response.status_code} after 3 retries")
                    jitter = random.Random(int(key[:8], 16) + transport_attempt).uniform(0.0, 1.0)
                    self.wait(2 * 2 ** (transport_attempt - 1) + jitter)
                    continue
                if 400 <= response.status_code <= 499:
                    detail = response.text
                    if self.settings.llm_api_key:
                        detail = detail.replace(self.settings.llm_api_key, "***")
                    detail = detail[:200]
                    raise LLMTransportError(f"http {response.status_code}: {detail}")
                try:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                except (ValueError, KeyError, IndexError, TypeError):
                    raise LLMTransportError("invalid response body") from None
                if not isinstance(content, str):
                    raise LLMTransportError("invalid response body")
                usage = data.get("usage")
                if (
                    isinstance(usage, dict)
                    and "prompt_tokens" in usage
                    and "completion_tokens" in usage
                ):
                    prompt_tokens = int(usage["prompt_tokens"])
                    completion_tokens = int(usage["completion_tokens"])
                else:
                    prompt_tokens = math.ceil(
                        sum(len(message["content"]) for message in messages) / 4
                    )
                    completion_tokens = math.ceil(len(content) / 4)
                    _LOG.warning(
                        "usage missing; estimated token counts",
                        extra={"event": "usage_missing", "fields": {"model": model}},
                    )
                return {
                    "text": content,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }

    @staticmethod
    def _parse(text: str, schema: dict[str, object] | None) -> dict[str, object] | None:
        if schema is None:
            return None
        try:
            value = json.loads(text)
            _validator(json.dumps(schema, sort_keys=True)).validate(value)
        except (ValueError, jsonschema.ValidationError):
            raise LLMOutputError("invalid JSON or schema mismatch") from None
        if not isinstance(value, dict):
            raise LLMOutputError("invalid JSON or schema mismatch")
        return value

    def _check_estimate(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        ceiling: float | None,
        budget: Budget | None,
    ) -> None:
        estimate = estimate_usd(model, messages, max_tokens)
        if ceiling is not None and estimate > ceiling:
            raise BudgetExceeded(f"call estimate {estimate:g} exceeds per-call ceiling {ceiling:g}")
        if budget is not None:
            budget.check(estimate)

    def chat(
        self,
        *,
        purpose: str,
        messages: list[dict[str, str]],
        schema: dict[str, object] | None = None,
        schema_name: str = "output",
        model: str | None = None,
        sample_index: int = 0,
        budget: Budget | None = None,
    ) -> ChatResult:
        default_model, max_tokens, reasoning_effort, ceiling = _purpose_config(purpose)
        resolved_model = model or default_model
        if resolved_model is None:
            raise ConfigError(f"model is required for purpose {purpose}")
        request_messages = [dict(message) for message in messages]
        key = self._key(
            resolved_model,
            request_messages,
            max_tokens,
            reasoning_effort,
            schema,
            schema_name,
            sample_index,
        )
        started = self.clock()
        sent_request = False

        with llm_span(purpose, resolved_model) as span:
            span.set_attribute("minutes.sample_index", sample_index)
            span.set_attribute("minutes.correlation_id", get_correlation_id())
            value = self._cache_get(key)
            if value is None and self.settings.llm_mode == "replay":
                raise LLMReplayMiss(f"no replay entry for {key}")
            if value is None:
                self._check_estimate(
                    model=resolved_model,
                    messages=request_messages,
                    max_tokens=max_tokens,
                    ceiling=ceiling,
                    budget=budget,
                )
                request_started = self.clock()
                value = self._request(
                    key=key,
                    model=resolved_model,
                    messages=request_messages,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    schema=schema,
                    schema_name=schema_name,
                )
                value["latency_ms"] = max(0, int((self.clock() - request_started) * 1000))
                self._cache_put(key, resolved_model, value)
                sent_request = True

            values = [value]
            try:
                parsed = self._parse(str(value["text"]), schema)
            except LLMOutputError:
                corrected_messages = [
                    *request_messages,
                    {"role": "user", "content": _CORRECTION},
                ]
                corrected_key = self._key(
                    resolved_model,
                    corrected_messages,
                    max_tokens,
                    reasoning_effort,
                    schema,
                    schema_name,
                    sample_index,
                )
                corrected = self._cache_get(corrected_key)
                if corrected is None and self.settings.llm_mode == "replay":
                    raise LLMReplayMiss(f"no replay entry for {corrected_key}") from None
                if corrected is None:
                    self._check_estimate(
                        model=resolved_model,
                        messages=corrected_messages,
                        max_tokens=max_tokens,
                        ceiling=ceiling,
                        budget=None,
                    )
                    request_started = self.clock()
                    corrected = self._request(
                        key=corrected_key,
                        model=resolved_model,
                        messages=corrected_messages,
                        max_tokens=max_tokens,
                        reasoning_effort=reasoning_effort,
                        schema=schema,
                        schema_name=schema_name,
                    )
                    corrected["latency_ms"] = max(0, int((self.clock() - request_started) * 1000))
                    self._cache_put(corrected_key, resolved_model, corrected)
                    sent_request = True
                values.append(corrected)
                value = corrected
                parsed = self._parse(str(value["text"]), schema)

            prompt_tokens = sum(int(response["prompt_tokens"]) for response in values)
            completion_tokens = sum(int(response["completion_tokens"]) for response in values)
            cached = not sent_request
            latency_ms = (
                max(0, int((self.clock() - started) * 1000))
                if sent_request
                else sum(int(response["latency_ms"]) for response in values)
            )
            cost = cost_usd(resolved_model, prompt_tokens, completion_tokens)
            result = ChatResult(
                text=str(value["text"]),
                parsed=parsed,
                model=resolved_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost,
                latency_ms=latency_ms,
                cached=cached,
            )
            if budget is not None:
                budget.add(cost)
            record_completed_call(
                span,
                purpose=purpose,
                model=resolved_model,
                key=key,
                cached=cached,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=cost,
                latency_ms=latency_ms,
            )
            return result
