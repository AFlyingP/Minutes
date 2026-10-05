"""Deterministic stand-ins for the GPU models, so tests and the demo need no GPU."""

import hashlib
import json
import re

import numpy as np
from PIL import Image

from minutes.config import FIXTURE_MANIFEST
from minutes.models import OcrWord

# fmt: off
STOPWORDS = frozenset({
    "the", "an", "of", "on", "in", "to", "and", "for", "by", "did", "how", "what", "which",
    "was", "is", "are", "about", "with", "at", "be", "it", "this", "that", "from", "as",
})
# fmt: on


def stub_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) >= 2 and t not in STOPWORDS]


class StubEmbedder:
    model_id = "stub-embedder"
    dim = 768

    def embed_query(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in stub_tokens(text):
            vector[int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % self.dim] += 1.0
        if not vector.any():
            vector[0] = 1.0
        return vector / np.linalg.norm(vector)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed_query(text) for text in texts])


class StubReranker:
    model_id = "stub-reranker"

    def score(self, query: str, passages: list[str]) -> list[float]:
        wanted = set(stub_tokens(query))
        if not wanted:
            return [0.0 for _ in passages]
        return [len(wanted & set(stub_tokens(passage))) / len(wanted) for passage in passages]


class StubOcr:
    def read(self, image: Image.Image, hint: str) -> list[OcrWord]:
        path = FIXTURE_MANIFEST.parent / "ocr" / f"{hint}.json"
        if not path.exists():
            return []
        return [OcrWord(**word) for word in json.loads(path.read_text(encoding="utf-8"))]
