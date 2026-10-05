import socket
from collections.abc import Iterator
from typing import Any

import psycopg
import pytest
from psycopg.rows import TupleRow

from minutes import db
from minutes.config import get_settings
from minutes.ingest import pipeline

STUB_ENV = {
    "MINUTES_LLM_MODE": "stub",
    "MINUTES_MODELS_MODE": "stub",
    "MINUTES_CORPUS": "fixture",
    "MINUTES_FAULT": "",
    "MINUTES_OTEL_ENDPOINT": "",
}


@pytest.fixture(scope="session", autouse=True)
def stub_env() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        for key, value in STUB_ENV.items():
            patch.setenv(key, value)
        get_settings.cache_clear()
        yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    real_connect = socket.socket.connect

    def connect(self: socket.socket, address: Any) -> None:
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "localhost"):
            raise RuntimeError("network access in tests")
        real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)


@pytest.fixture(scope="session")
def test_db_url() -> str:
    url = get_settings().test_database_url
    db.reset(url)
    return url


@pytest.fixture
def conn(test_db_url: str) -> Iterator[psycopg.Connection[TupleRow]]:
    with db.get_pool(test_db_url).connection() as connection:
        yield connection
        connection.rollback()


@pytest.fixture(scope="session")
def fixture_corpus(test_db_url: str) -> str:
    pipeline.load_fixture_corpus(test_db_url)
    return test_db_url
