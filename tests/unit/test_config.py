import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from minutes import config
from minutes.config import (
    Settings,
    corpus_database_url,
    get_settings,
    load_cities,
    load_models,
    load_prices,
)
from minutes.errors import ConfigError


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    # an empty directory keeps a developer's .env out of the settings
    for key in os.environ:
        if key.startswith("MINUTES_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults_load() -> None:
    settings = Settings()
    assert settings.llm_mode == "stub"
    assert settings.corpus == "fixture"
    assert settings.query_budget_usd == 0.05


def test_live_mode_requires_url_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINUTES_LLM_MODE", "live")
    with pytest.raises(ConfigError) as raised:
        get_settings()
    assert str(raised.value).startswith("invalid config: MINUTES_LLM_BASE_URL")


def test_fault_rejected_outside_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINUTES_FAULT", "llm_quota:1")
    monkeypatch.setenv("MINUTES_CORPUS", "dev")
    with pytest.raises(ConfigError):
        get_settings()


def test_load_models_and_prices() -> None:
    assert load_models().strong == "gpt-6.1-sol"
    assert load_prices()["gpt-6-luna"].output_per_mtok == 0.5


def test_load_cities_real_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINUTES_CORPUS", "dev")
    cities = load_cities()
    assert set(cities) == {"seattle", "lincoln"}
    assert cities["lincoln"].sample_seed == 1102


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("MINUTES_DATABASE_URL", "mysql://minutes@127.0.0.1/minutes"),
        ("MINUTES_TEST_DATABASE_URL", "postgresql://minutes:minutes@127.0.0.1:5433/minutes"),
        ("MINUTES_SQL_TOOL_DATABASE_URL", "postgresql://minutes:minutes@127.0.0.1:5433/minutes"),
        ("MINUTES_LLM_BASE_URL", "ftp://models.test/v1"),
        ("MINUTES_LLM_MODE", "remote"),
        ("MINUTES_DATA_DIR", "../elsewhere"),
        ("MINUTES_OTEL_ENDPOINT", "https://127.0.0.1:4318"),
        ("MINUTES_QUERY_BUDGET_USD", "0"),
        ("MINUTES_LLM_DAILY_CAP", "5001"),
        ("MINUTES_LOG_FILE", "elsewhere/minutes.log"),
        ("MINUTES_API_PORT", "80"),
        ("MINUTES_FAULT", "llm_quota"),
    ],
)
def test_invalid_value_names_its_key(monkeypatch: pytest.MonkeyPatch, key: str, value: str) -> None:
    monkeypatch.setenv(key, value)
    with pytest.raises(ConfigError) as raised:
        get_settings()
    assert str(raised.value).startswith(f"invalid config: {key}: ")


def test_fixture_cities_need_the_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "FIXTURE_MANIFEST", tmp_path / "manifest.json")
    with pytest.raises(ConfigError, match="fixture manifest missing"):
        load_cities()


def test_missing_key_in_config_file_is_a_config_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "prices.toml").write_text('[models."small"]\ninput_per_mtok = 1.0\n')
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    with pytest.raises(ConfigError, match=r"invalid config: prices\.toml"):
        load_prices()
    with pytest.raises(ConfigError, match=r"config file missing: models\.toml"):
        load_models()


def test_corpus_database_url_picks_the_test_database_for_fixture() -> None:
    assert corpus_database_url("fixture").endswith("/minutes_test")
    assert corpus_database_url("full").endswith("/minutes")
