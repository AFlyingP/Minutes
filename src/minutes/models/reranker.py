import hashlib
import json
import os
import tempfile
from pathlib import Path
from threading import Lock
from typing import Any

from minutes.config import get_settings

_model: Any = None
_model_lock = Lock()


def _cache_key(query: str, passage: str) -> str:
    value = f"rerank\n{GpuReranker.model_id}\n{query}\n{passage}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_model() -> Any:
    global _model
    with _model_lock:
        if _model is None:
            settings = get_settings()
            os.environ["HF_HOME"] = str(settings.data_dir / "models" / "hf")
            import sentence_transformers

            _model = sentence_transformers.CrossEncoder(
                GpuReranker.model_id, device="cuda", max_length=512
            )
            _model.model.half()
    return _model


def _cache_path(key: str) -> Path:
    return get_settings().data_dir / "cache" / "rerank" / key[:2] / f"{key}.json"


def _read_cache(path: Path) -> float | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    return float(value["score"])


def _write_cache(path: Path, score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".rerank-cache-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump({"score": score}, file, separators=(",", ":"))
            file.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


class GpuReranker:
    model_id = "BAAI/bge-reranker-v2-m3"

    def score(self, query: str, passages: list[str]) -> list[float]:
        scores = [0.0] * len(passages)
        missing: list[tuple[int, str, str]] = []
        for index, passage in enumerate(passages):
            key = _cache_key(query, passage)
            score = _read_cache(_cache_path(key))
            if score is None:
                missing.append((index, key, passage))
            else:
                scores[index] = score

        if missing:
            values = _load_model().predict(
                [(query, passage) for _, _, passage in missing], batch_size=32
            )
            for (index, key, _), value in zip(missing, values, strict=True):
                score = round(float(value), 6)
                scores[index] = score
                _write_cache(_cache_path(key), score)
        return scores
