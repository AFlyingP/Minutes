"""Checks that the outside pieces work on this machine: model endpoint, city sites, GPU."""

import io
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from itertools import islice
from typing import Any

import httpx
import pdfplumber
from PIL import Image

from minutes.config import FIXTURE_MANIFEST, CityConfig, get_settings, load_cities, load_models
from minutes.errors import ConfigError
from minutes.ingest import parse
from minutes.ingest.layout import Word, build_text
from minutes.ingest.ocr import DoctrEngine
from minutes.ingest.quality import levenshtein
from minutes.sources import get_source
from minutes.sources.base import DocRef, Source

CUE = re.compile(rb"\d{2}:\d{2}:\d{2}[.,]\d{3} --> ")
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class ProbeResult:
    name: str
    ok: bool
    detail: str


def _api_key() -> str:
    # a failure caused by bad settings still has to be reported
    try:
        return get_settings().llm_api_key
    except ConfigError:
        return ""


def _failure(name: str, err: Exception) -> ProbeResult:
    detail = f"{type(err).__name__}: {err}"
    key = _api_key()
    if key:
        detail = detail.replace(key, "***")
    return ProbeResult(name, False, detail[:200])


def _check(name: str, step: Callable[..., str], *args: Any) -> ProbeResult:
    # a probe reports every failure as a result, whatever the library raised
    try:
        return ProbeResult(name, True, step(*args))
    except Exception as err:
        return _failure(name, err)


def _guarded(name: str, results: Callable[[], list[ProbeResult]]) -> list[ProbeResult]:
    # covers what runs outside a single check: settings, config files, imports
    try:
        return results()
    except Exception as err:
        return [_failure(name, err)]


def _chat(client: httpx.Client, model: str, prompt: str, schema: dict[str, Any] | None) -> str:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 300,
        "reasoning_effort": "low",
    }
    if schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "answer", "strict": True, "schema": schema},
        }
    response = client.post("/chat/completions", json=body)
    response.raise_for_status()
    content: str = response.json()["choices"][0]["message"]["content"]
    return content


def _plain_chat(client: httpx.Client, model: str) -> str:
    if not _chat(client, model, "Reply with the word ok.", None).strip():
        raise ValueError("empty reply")
    return ""


def _schema_chat(client: httpx.Client, model: str) -> str:
    reply = _chat(client, model, "What is 2+3? Reply as JSON.", ANSWER_SCHEMA)
    if json.loads(reply)["answer"] != 5:
        raise ValueError(f"unexpected answer: {reply[:80]}")
    return ""


def probe_llm(transport: httpx.BaseTransport | None = None) -> list[ProbeResult]:
    return _guarded("config", lambda: _llm(transport))


def _llm(transport: httpx.BaseTransport | None) -> list[ProbeResult]:
    settings = get_settings()
    if not settings.llm_base_url or not settings.llm_api_key:
        detail = "MINUTES_LLM_BASE_URL and MINUTES_LLM_API_KEY are required"
        return [ProbeResult("config", False, detail)]
    models = load_models()
    results = []
    with httpx.Client(
        base_url=settings.llm_base_url,
        headers={"Authorization": f"Bearer {settings.llm_api_key}"},
        timeout=httpx.Timeout(120, connect=10),
        transport=transport,
    ) as client:
        for model in (models.strong, models.cheap):
            results.append(_check(f"{model} chat", _plain_chat, client, model))
            results.append(_check(f"{model} json_schema", _schema_chat, client, model))
    return results


def _first_pdf(source: Source, documents: list[DocRef]) -> str:
    if not documents:
        raise ValueError("none listed")
    source.fetch(documents[0])
    return ""


def _first_captions(source: Source, documents: list[DocRef]) -> str:
    reason = "none listed"
    for document in islice(documents, 5):
        try:
            body = source.fetch(document)
        except Exception as err:
            reason = f"{document.source_url}: {err}"
            continue
        if len(body) >= 2000 and CUE.search(body):
            return ""
        reason = f"{document.source_url}: status 200, {len(body)} bytes"
    raise ValueError(reason)


def _documents(count: int) -> str:
    if count < 200:
        raise ValueError(f"{count} documents")
    return str(count)


def _probe_city(client: httpx.Client, city_id: str, city: CityConfig) -> list[ProbeResult]:
    try:
        date_from = date.fromisoformat(city.date_from)
        date_to = date.fromisoformat(city.date_to)
        source = get_source(city_id, city, client)
        listed = source.list_meetings(date_from, date_to)
    except Exception as err:
        return [_failure(f"{city_id} listing", err)]
    documents = [document for _, meeting_docs in listed for document in meeting_docs]
    agendas = [document for document in documents if document.kind == "agenda"]
    minutes = [document for document in documents if document.kind == "minutes"]
    captions = [document for document in documents if document.kind == "transcript"]
    return [
        ProbeResult(f"{city_id} listing", True, ""),
        _check(f"{city_id} agenda pdf", _first_pdf, source, agendas),
        _check(f"{city_id} minutes pdf", _first_pdf, source, minutes),
        _check(f"{city_id} captions", _first_captions, source, captions),
        _check(f"{city_id} documents >= 200", _documents, len(documents)),
    ]


def probe_sources(transport: httpx.BaseTransport | None = None) -> list[ProbeResult]:
    return _guarded("config", lambda: _sources(transport))


def _sources(transport: httpx.BaseTransport | None) -> list[ProbeResult]:
    results = []
    with httpx.Client(
        timeout=httpx.Timeout(60, connect=10),
        headers={"User-Agent": "minutes-research/0.1"},
        max_redirects=5,
        transport=transport,
    ) as client:
        for city_id, city in load_cities("full").items():
            results += _probe_city(client, city_id, city)
    return results


def probe_gpu() -> list[ProbeResult]:
    return _guarded("cuda", _gpu)


def probe_ocr() -> list[ProbeResult]:
    return _guarded("ocr_cer", _ocr)


def _ocr() -> list[ProbeResult]:
    corpus_dir = FIXTURE_MANIFEST.parent
    content = json.loads((corpus_dir / "content.json").read_text(encoding="utf-8"))
    document = next(doc for doc in content["documents"] if doc["id"] == "birch-minutes-201")
    reference = " ".join("\n".join(document["pages"][2]["lines"]).split())
    file_path = str(corpus_dir / "birch-minutes-201.pdf")
    with pdfplumber.open(file_path) as pdf:
        width, height = pdf.pages[2].width, pdf.pages[2].height
    with Image.open(io.BytesIO(parse.render_page_png(file_path, 2, 200))) as image:
        ocr_words = DoctrEngine().read(image, "birch-minutes-201-3")
    words = [
        Word(word.text, word.x0 * width, word.y0 * height, word.x1 * width, word.y1 * height)
        for word in ocr_words
    ]
    text, _ = build_text(words, [], width, height)
    cer = levenshtein(" ".join(text.split()), reference) / len(reference)
    return [ProbeResult(f"ocr_cer={cer:.3f}", cer <= 0.05, "" if cer <= 0.05 else "above 0.05")]


def _gpu() -> list[ProbeResult]:
    models_dir = get_settings().data_dir / "models"
    os.environ["HF_HOME"] = str(models_dir / "hf")
    os.environ["DOCTR_CACHE_DIR"] = str(models_dir / "doctr")
    import numpy as np
    import torch
    from doctr.models import ocr_predictor
    from sentence_transformers import CrossEncoder, SentenceTransformer

    loaded: list[Any] = []

    def cuda() -> str:
        if not torch.cuda.is_available():
            raise ValueError("torch.cuda.is_available() is false")
        return ""

    def embedder() -> str:
        model = SentenceTransformer(load_models().embedder, device="cuda").half()
        loaded.append(model)
        dim = model.get_sentence_embedding_dimension()
        if dim != 768:
            raise ValueError(f"dimension {dim}")
        model.encode([f"council meeting item {i}" for i in range(64)])
        return f"dim={dim}"

    def reranker() -> str:
        model = CrossEncoder(load_models().reranker, device="cuda", max_length=512)
        loaded.append(model)
        model.predict([("sidewalk repair contract", f"agenda item {i}") for i in range(32)])
        return ""

    def ocr() -> str:
        predictor = ocr_predictor(
            det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True
        ).cuda()
        loaded.append(predictor)
        predictor([np.full((2200, 1700, 3), 255, dtype=np.uint8)])
        return ""

    results = [_check("cuda", cuda)]
    if not results[0].ok:
        return results
    results += [_check("embedder", embedder), _check("reranker", reranker), _check("ocr", ocr)]
    peak = torch.cuda.max_memory_allocated() / 2**30
    results.append(
        ProbeResult(f"peak_gib={peak:.2f}", peak < 6.0, "" if peak < 6.0 else "above 6.0")
    )
    return results
