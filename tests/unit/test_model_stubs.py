import importlib
import sys
from types import ModuleType

import numpy as np
import pytest

from minutes.config import get_settings
from minutes.errors import ConfigError
from minutes.models import get_embedder, get_ocr, get_reranker


def test_stub_mode_imports_no_gpu_library() -> None:
    get_embedder()
    get_reranker()
    get_ocr()
    assert not {"torch", "sentence_transformers", "doctr"} & sys.modules.keys()


def test_stub_embedder_shape_norm_and_determinism() -> None:
    embedder = get_embedder()
    texts = ["sidewalk repair", "zoning variance"]
    vectors = embedder.embed_passages(texts)
    assert vectors.shape == (2, 768)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), np.ones(2), atol=1e-6)
    assert np.array_equal(vectors, embedder.embed_passages(texts))


def test_stub_embedder_similar_texts_score_higher() -> None:
    embedder = get_embedder()
    query = embedder.embed_query("sidewalk repair")
    similar = embedder.embed_query("sidewalk repair contract")
    unrelated = embedder.embed_query("zoning variance")
    assert float(query @ similar) > float(query @ unrelated)


def test_stub_reranker_overlap_fraction() -> None:
    scores = get_reranker().score(
        "sidewalk repair contract",
        ["sidewalk repair contract", "sidewalk contract", "zoning variance"],
    )
    assert scores == [1.0, 2 / 3, 0.0]


def test_gpu_mode_without_cuda_raises_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINUTES_MODELS_MODE", "gpu")
    get_settings.cache_clear()
    original_import = importlib.import_module

    def fail_torch(name: str, package: str | None = None) -> ModuleType:
        if name == "torch":
            raise ImportError("torch is unavailable")
        if package is None:
            return original_import(name)
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fail_torch)
    try:
        with pytest.raises(ConfigError, match="gpu models unavailable"):
            get_embedder()
    finally:
        get_settings.cache_clear()
