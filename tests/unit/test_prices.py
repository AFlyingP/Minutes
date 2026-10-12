import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from minutes import config
from minutes.errors import ConfigError
from minutes.llm.prices import cost_usd, estimate_usd


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    for name in tuple(os.environ):
        if name.startswith("MINUTES_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MINUTES_DATA_DIR", "data")
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


def test_cost_formula() -> None:
    assert cost_usd("gpt-6.1-sol", 1000, 500) == 0.007
    assert cost_usd("gpt-6-luna", 1_000_000, 0) == 0.10


def test_unknown_model_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="no price configured for model unknown"):
        cost_usd("unknown", 10, 10)


def test_estimate_uses_quarter_characters_and_max_tokens() -> None:
    assert estimate_usd("gpt-6.1-sol", [{"role": "user", "content": "123456789"}], 10) == 0.000106
