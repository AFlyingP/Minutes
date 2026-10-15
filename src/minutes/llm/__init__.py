from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from minutes.config import ROOT, get_settings
from minutes.errors import BudgetExceeded

if TYPE_CHECKING:
    from minutes.llm.cache import DiskCache


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


REPLAY_DIR = ROOT / "eval" / "gate"
# set by the gate recorder while it records a run; builds the live cache in place of DiskCache
cache_recorder: "Callable[[Path], DiskCache] | None" = None


def get_client() -> LLMClient:
    settings = get_settings()
    if settings.llm_mode == "stub":
        from minutes.llm.stub import StubClient

        return StubClient()
    from minutes.llm.cache import DiskCache, ReplayStore
    from minutes.llm.client import ThinClient

    if settings.llm_mode == "live":
        root = settings.data_dir / "cache" / "llm"
        cache = DiskCache(root) if cache_recorder is None else cache_recorder(root)
        return ThinClient(cache=cache)
    return ThinClient(cache=ReplayStore(REPLAY_DIR / "llm.jsonl.gz"))
