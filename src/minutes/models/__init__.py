from typing import NamedTuple, Protocol

import numpy as np
from PIL import Image

from minutes.config import get_settings
from minutes.errors import ConfigError


class OcrWord(NamedTuple):
    """A word with its box as fractions of the image width and height."""

    text: str
    confidence: float
    x0: float
    y0: float
    x1: float
    y1: float


class Embedder(Protocol):
    model_id: str
    dim: int

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class Reranker(Protocol):
    model_id: str

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class OcrEngine(Protocol):
    def read(self, image: Image.Image, hint: str) -> list[OcrWord]: ...


def _require_stub() -> None:
    mode = get_settings().models_mode
    if mode != "stub":
        raise ConfigError(f"models mode {mode} is not available")


def get_embedder() -> Embedder:
    from minutes.models.stubs import StubEmbedder

    _require_stub()
    return StubEmbedder()


def get_reranker() -> Reranker:
    from minutes.models.stubs import StubReranker

    _require_stub()
    return StubReranker()


def get_ocr() -> OcrEngine:
    from minutes.models.stubs import StubOcr

    _require_stub()
    return StubOcr()
