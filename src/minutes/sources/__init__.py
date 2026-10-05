from minutes.config import CityConfig
from minutes.errors import ConfigError
from minutes.sources.base import Source
from minutes.sources.fixture import FixtureSource


def get_source(city_id: str, city: CityConfig) -> Source:
    if city.source == "fixture":
        return FixtureSource(city_id)
    raise ConfigError(f"source {city.source} is not available")
