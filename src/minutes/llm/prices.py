from math import ceil

from minutes.config import load_prices
from minutes.errors import ConfigError


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Compute the list-price equivalent for one model response."""
    prices = load_prices()
    if model not in prices:
        raise ConfigError(f"no price configured for model {model}")
    price = prices[model]
    return round(
        (prompt_tokens * price.input_per_mtok + completion_tokens * price.output_per_mtok)
        / 1_000_000,
        6,
    )


def estimate_usd(model: str, messages: list[dict[str, str]], max_tokens: int) -> float:
    prompt_tokens = ceil(sum(len(message["content"]) for message in messages) / 4)
    return cost_usd(model, prompt_tokens, max_tokens)
