from datetime import date

import httpx

from minutes.config import CityConfig
from minutes.errors import ConfigError
from minutes.sources.base import Source
from minutes.sources.fixture import FixtureSource
from minutes.sources.granicus import GranicusSource
from minutes.sources.legistar import LegistarSource


def get_source(
    city_id: str,
    city: CityConfig,
    client: httpx.Client | None = None,
) -> Source:
    if city.source == "fixture":
        return FixtureSource(city_id)
    date_from = date.fromisoformat(city.date_from)
    date_to = date.fromisoformat(city.date_to)
    if city.source == "legistar":
        return LegistarSource(city_id, city, date_from, date_to, client)
    if city.source == "granicus":
        return GranicusSource(city_id, city, date_from, date_to, client)
    raise ConfigError(f"source {city.source} is not available")
