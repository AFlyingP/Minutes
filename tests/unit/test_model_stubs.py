import gzip
import importlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pytest

from minutes.config import get_settings
from minutes.errors import ConfigError
from minutes.evals import gate
from minutes.models import get_embedder, get_ocr, get_reranker


def test_stub_mode_imports_no_gpu_library(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINUTES_MODELS_MODE", "stub")
    get_settings.cache_clear()
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


def test_replay_models_reuse_recordings_and_notice_rewrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MINUTES_MODELS_MODE", "replay")
    monkeypatch.setattr(gate, "REPLAY_DIR", tmp_path)
    get_settings.cache_clear()
    embeddings = tmp_path / "embeddings.npz"
    rerank = tmp_path / "rerank.jsonl.gz"

    def write(value: float) -> None:
        np.savez_compressed(
            embeddings,
            keys=np.asarray(["a" * 64], dtype="S64"),
            vectors=np.full((1, 768), value, dtype=np.float16),
        )
        with gzip.open(rerank, "wt", encoding="utf-8") as file:
            file.write(json.dumps({"key": "pair", "score": value}) + "\n")

    try:
        write(0.0)
        with (
            patch("numpy.load", wraps=np.load) as load,
            patch("gzip.open", wraps=gzip.open) as read,
        ):
            first_embedder = get_embedder()
            second_embedder = get_embedder()
            first_reranker = get_reranker()
            second_reranker = get_reranker()
            assert isinstance(first_embedder, gate.ReplayEmbedder)
            assert isinstance(second_embedder, gate.ReplayEmbedder)
            assert isinstance(first_reranker, gate.ReplayReranker)
            assert isinstance(second_reranker, gate.ReplayReranker)
            assert first_embedder.values is second_embedder.values
            assert first_reranker.values is second_reranker.values
            assert load.call_count == 1
            assert read.call_count == 1
        stamps = {path: path.stat().st_mtime_ns for path in (embeddings, rerank)}
        write(1.0)
        for path, stamp in stamps.items():
            os.utime(path, ns=(stamp + 1_000_000_000, stamp + 1_000_000_000))
        changed_embedder = get_embedder()
        changed_reranker = get_reranker()
        assert isinstance(changed_embedder, gate.ReplayEmbedder)
        assert isinstance(changed_reranker, gate.ReplayReranker)
        np.testing.assert_array_equal(changed_embedder.values["a" * 64], np.ones(768))
        assert changed_reranker.values["pair"] == 1.0
    finally:
        get_settings.cache_clear()
