import importlib
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


def _require_cuda() -> None:
    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError) as err:
        raise ConfigError(f"gpu models unavailable: {err}") from err
    if not torch.cuda.is_available():
        raise ConfigError("gpu models unavailable: cuda is unavailable")


def get_embedder() -> Embedder:
    mode = get_settings().models_mode
    if mode == "stub":
        from minutes.models.stubs import StubEmbedder

        return StubEmbedder()
    if mode == "gpu":
        _require_cuda()
        from minutes.models.embedder import GpuEmbedder

        return GpuEmbedder()
    raise ConfigError(f"models mode {mode} is not available")


def get_reranker() -> Reranker:
    mode = get_settings().models_mode
    if mode == "stub":
        from minutes.models.stubs import StubReranker

        return StubReranker()
    if mode == "gpu":
        _require_cuda()
        from minutes.models.reranker import GpuReranker

        return GpuReranker()
    raise ConfigError(f"models mode {mode} is not available")


def get_ocr() -> OcrEngine:
    from minutes.ingest.ocr import DoctrEngine
    from minutes.models.stubs import StubOcr

    mode = get_settings().models_mode
    if mode == "stub":
        return StubOcr()
    if mode == "gpu":
        return DoctrEngine()
    raise ConfigError(f"models mode {mode} is not available")
