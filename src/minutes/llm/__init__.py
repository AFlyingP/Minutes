from dataclasses import dataclass
from typing import Protocol

from minutes.config import ROOT, get_settings
from minutes.errors import BudgetExceeded


@dataclass(frozen=True)
class ChatResult:
    text: str
    parsed: dict[str, object] | None
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: int
    cached: bool


class Budget:
    """Running cost of one request against its limit."""

    def __init__(self, limit_usd: float) -> None:
        self.limit_usd = limit_usd
        self.spent = 0.0

    def check(self, estimate_usd: float) -> None:
        if self.spent + estimate_usd > self.limit_usd:
            raise BudgetExceeded(f"budget of {self.limit_usd} usd exceeded")

    def add(self, cost_usd: float) -> None:
        self.spent += cost_usd


class LLMClient(Protocol):
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
    ) -> ChatResult: ...


def get_client() -> LLMClient:
    settings = get_settings()
    if settings.llm_mode == "stub":
        from minutes.llm.stub import StubClient

        return StubClient()
    from minutes.llm.cache import DiskCache, ReplayStore
    from minutes.llm.client import ThinClient

    if settings.llm_mode == "live":
        return ThinClient(cache=DiskCache(settings.data_dir / "cache" / "llm"))
    return ThinClient(cache=ReplayStore(ROOT / "eval" / "gate" / "llm.jsonl.gz"))
