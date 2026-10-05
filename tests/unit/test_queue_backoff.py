import random

import pytest

from minutes.config import get_settings
from minutes.queue import backoff_seconds


def test_backoff_formula() -> None:
    for attempt in range(1, 6):
        base = min(600, 5 * 2 ** (attempt - 1))
        jitter = random.Random(7 * 1000 + attempt).random() * 0.25
        assert backoff_seconds(7, attempt) == base * (1 + jitter)

    assert 600 <= backoff_seconds(7, 8) <= 750


def test_backoff_is_zero_with_fault_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINUTES_FAULT", "stage:parse:birch-agenda-201:2")
    get_settings.cache_clear()
    try:
        assert backoff_seconds(7, 1) == 0.0
    finally:
        get_settings.cache_clear()
