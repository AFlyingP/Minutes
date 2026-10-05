import json
import re
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

import pydantic
from pydantic import Field
from pydantic.dataclasses import dataclass
from pydantic_settings import BaseSettings, SettingsConfigDict

from minutes.errors import ConfigError

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
FIXTURE_MANIFEST = ROOT / "tests" / "fixtures" / "corpus" / "manifest.json"

Corpus = Literal["fixture", "dev", "full"]
FAULT = re.compile(r"stage:[a-z]+:[^:]+:\d+|llm_quota:\d+")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MINUTES_", env_file=".env", extra="ignore")

    database_url: str = "postgresql://minutes:minutes@127.0.0.1:5433/minutes"
    test_database_url: str = "postgresql://minutes:minutes@127.0.0.1:5433/minutes_test"
    sql_tool_database_url: str = "postgresql://minutes_ro:minutes_ro@127.0.0.1:5433/minutes"
    sql_tool_test_database_url: str = (
        "postgresql://minutes_ro:minutes_ro@127.0.0.1:5433/minutes_test"
    )
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_mode: Literal["stub", "live", "replay"] = "stub"
    models_mode: Literal["stub", "gpu", "replay"] = "stub"
    corpus: Corpus = "fixture"
    data_dir: Path = Path("data")
    otel_endpoint: str = ""
    query_budget_usd: Annotated[float, Field(gt=0, le=5)] = 0.05
    llm_daily_cap: Annotated[int, Field(ge=1, le=5000)] = 500
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_file: str = ""
    api_port: Annotated[int, Field(ge=1024, le=65535)] = 8000
    fault: str = ""


@dataclass(frozen=True)
class CityConfig:
    name: str
    state: str
    source: str
    item_format: str
    bodies: list[str]
    sample_seed: int
    date_from: str
    date_to: str
    legistar_client: str | None = None
    granicus_host: str | None = None
    granicus_view_id: int | None = None


@dataclass(frozen=True)
class PurposeConfig:
    model: Literal["strong", "cheap"]
    max_tokens: int
    reasoning_effort: str


@dataclass(frozen=True)
class ModelsConfig:
    strong: str
    cheap: str
    embedder: str
    reranker: str
    purposes: dict[str, PurposeConfig]


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float
    as_of: str
    source: str


def _inside(path: Path, parent: Path) -> bool:
    return (Path.cwd() / path).resolve().is_relative_to((Path.cwd() / parent).resolve())


def _problems(s: Settings) -> Iterator[tuple[str, str]]:
    """Yield (key, reason) for each rule that pydantic's field types cannot express."""
    if not s.database_url.startswith("postgresql://"):
        yield "MINUTES_DATABASE_URL", "must be a postgresql:// URL"
    test_url = urlsplit(s.test_database_url)
    if test_url.scheme != "postgresql" or not test_url.path.endswith("_test"):
        yield (
            "MINUTES_TEST_DATABASE_URL",
            "must be a postgresql:// URL of a database ending in _test",
        )
    for key, url in (
        ("MINUTES_SQL_TOOL_DATABASE_URL", s.sql_tool_database_url),
        ("MINUTES_SQL_TOOL_TEST_DATABASE_URL", s.sql_tool_test_database_url),
    ):
        parts = urlsplit(url)
        if parts.scheme != "postgresql" or parts.username != "minutes_ro":
            yield key, "must be a postgresql:// URL with user minutes_ro"
    if s.llm_base_url and not s.llm_base_url.startswith(("http://", "https://")):
        yield "MINUTES_LLM_BASE_URL", "must be empty or an http(s) URL"
    if s.llm_mode == "live" and not s.llm_base_url:
        yield "MINUTES_LLM_BASE_URL", "required when MINUTES_LLM_MODE=live"
    if s.llm_mode == "live" and not s.llm_api_key:
        yield "MINUTES_LLM_API_KEY", "required when MINUTES_LLM_MODE=live"
    if not _inside(s.data_dir, Path(".")):
        yield "MINUTES_DATA_DIR", "must be inside the repository"
    if s.otel_endpoint and not s.otel_endpoint.startswith("http://"):
        yield "MINUTES_OTEL_ENDPOINT", "must be empty or an http:// URL"
    if s.log_file and not _inside(Path(s.log_file), s.data_dir):
        yield "MINUTES_LOG_FILE", "must be inside MINUTES_DATA_DIR"
    if s.fault and not FAULT.fullmatch(s.fault):
        yield "MINUTES_FAULT", "must be stage:<stage>:<document_id>:<count> or llm_quota:<count>"
    if s.fault and s.corpus != "fixture":
        yield "MINUTES_FAULT", "only allowed when MINUTES_CORPUS=fixture"


@lru_cache
def get_settings() -> Settings:
    try:
        settings = Settings()
    except pydantic.ValidationError as err:
        first = err.errors()[0]
        key = "MINUTES_" + str(first["loc"][0]).upper()
        raise ConfigError(f"invalid config: {key}: {first['msg']}") from err
    for key, reason in _problems(settings):
        raise ConfigError(f"invalid config: {key}: {reason}")
    return settings


def corpus_database_url(corpus: Corpus) -> str:
    settings = get_settings()
    return settings.test_database_url if corpus == "fixture" else settings.database_url


@contextmanager
def _reading(path: Path) -> Iterator[dict[str, Any]]:
    try:
        yield tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as err:
        raise ConfigError(f"config file missing: {path.name}") from err
    except tomllib.TOMLDecodeError as err:
        raise ConfigError(f"invalid config: {path.name}: {err}") from err
    except KeyError as err:
        raise ConfigError(f"invalid config: {path.name}: missing key {err}") from err
    except (TypeError, pydantic.ValidationError) as err:
        raise ConfigError(f"invalid config: {path.name}: {err}") from err


def load_cities(corpus: Corpus | None = None) -> dict[str, CityConfig]:
    """Cities of the given corpus, or of MINUTES_CORPUS; the fixture corpus has its own cities."""
    if (corpus or get_settings().corpus) == "fixture":
        if not FIXTURE_MANIFEST.exists():
            raise ConfigError("fixture manifest missing")
        cities = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))["cities"]
        date_from, date_to = "2024-01-01", "2024-12-31"
        return {
            key: CityConfig(**table, date_from=date_from, date_to=date_to)
            for key, table in cities.items()
        }
    with _reading(CONFIG_DIR / "cities.toml") as data:
        date_from, date_to = data["date_from"], data["date_to"]
        return {
            key: CityConfig(**table, date_from=date_from, date_to=date_to)
            for key, table in data["cities"].items()
        }


def load_models() -> ModelsConfig:
    with _reading(CONFIG_DIR / "models.toml") as data:
        return ModelsConfig(**data)


def load_prices() -> dict[str, Price]:
    with _reading(CONFIG_DIR / "prices.toml") as data:
        return {model: Price(**table) for model, table in data["models"].items()}
