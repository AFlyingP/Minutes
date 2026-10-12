import hashlib
import os
import tempfile
from pathlib import Path
from threading import Lock
from typing import Any, cast

import numpy as np

from minutes.config import get_settings

_model: Any = None
_model_lock = Lock()


def _cache_key(kind: str, text: str) -> str:
    value = f"embed\n{GpuEmbedder.model_id}\n{kind}\n{text}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_model() -> Any:
    global _model
    with _model_lock:
        if _model is None:
            settings = get_settings()
            os.environ["HF_HOME"] = str(settings.data_dir / "models" / "hf")
            import sentence_transformers

            _model = sentence_transformers.SentenceTransformer(
                GpuEmbedder.model_id, device="cuda"
            ).half()
    return _model


def _cache_path(key: str) -> Path:
    return get_settings().data_dir / "cache" / "embed" / key[:2] / f"{key}.npy"


def _read_cache(path: Path) -> np.ndarray | None:
    try:
        return cast(np.ndarray, np.load(path).astype(np.float32, copy=False))
    except FileNotFoundError:
        return None


def _write_cache(path: Path, vector: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".embed-cache-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as file:
            np.save(file, vector)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


class GpuEmbedder:
    model_id = "BAAI/bge-base-en-v1.5"
    dim = 768

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, texts, "passage")

    def embed_query(self, text: str) -> np.ndarray:
        encoded = self._embed(
            [text],
            [f"Represent this sentence for searching relevant passages: {text}"],
            "query",
        )
        return cast(np.ndarray, encoded[0])

    def _embed(self, cache_texts: list[str], input_texts: list[str], kind: str) -> np.ndarray:
        if not cache_texts:
            return np.empty((0, self.dim), dtype=np.float32)

        vectors = [np.empty(self.dim, dtype=np.float32) for _ in cache_texts]
        missing: list[tuple[int, str, str]] = []
        for index, (cache_text, input_text) in enumerate(
            zip(cache_texts, input_texts, strict=True)
        ):
            key = _cache_key(kind, cache_text)
            path = _cache_path(key)
            vector = _read_cache(path)
            if vector is None:
                missing.append((index, key, input_text))
            else:
                vectors[index] = vector

        if missing:
            model = _load_model()
            encoded = model.encode(
                [text for _, _, text in missing],
                batch_size=64,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            for (index, key, _), vector in zip(missing, encoded, strict=True):
                value = np.asarray(vector, dtype=np.float32)
                vectors[index] = value
                _write_cache(_cache_path(key), value)
        return np.stack(vectors).astype(np.float32, copy=False)
