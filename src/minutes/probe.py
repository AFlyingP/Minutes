"""Checks that the outside pieces work on this machine: model endpoint, city sites, GPU."""

import html
import json
import os
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import islice
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

from minutes.config import CityConfig, get_settings, load_cities, load_models
from minutes.errors import ConfigError

# city hosts get at most 2 requests per second
REQUEST_GAP = 0.5
S3_HOST = "granicus_production_attachments.s3.amazonaws.com"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ROW_DATE = re.compile(rf"({'|'.join(MONTHS)})[a-z]* +(\d{{1,2}}), +(\d{{4}})")
# the player's caption track; the file name is not always the video's name
CAPTION_TRACK = re.compile(
    r'tracks:\s*\[\s*\{\s*file:\s*"'
    r'(documents/seattlechannel/closedcaption/\d{4}/[A-Za-z0-9_]+\.(?:srt|vtt))"',
    re.I,
)
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


@dataclass
class Listing:
    agendas: list[str]
    minutes: list[str]
    captions: Iterator[str]
    count: int


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


def _get(client: httpx.Client, url: str, **kwargs: Any) -> httpx.Response:
    time.sleep(REQUEST_GAP)
    return client.get(url, **kwargs)


def _text(fragment: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", fragment)).split())


def _seattle_captions(client: httpx.Client, pages: list[str]) -> Iterator[str]:
    for page_url in islice(pages, 10):
        response = _get(client, page_url, follow_redirects=True)
        if response.status_code >= 400:
            continue
        track = CAPTION_TRACK.search(response.text)
        if track:
            yield "https://www.seattlechannel.org/" + track[1]


def _legistar(client: httpx.Client, city: CityConfig) -> Listing:
    bodies = " or ".join(f"EventBodyName eq '{body}'" for body in city.bodies)
    params = {
        "$filter": f"EventDate ge datetime'{city.date_from}' and "
        f"EventDate le datetime'{city.date_to}' and ({bodies})",
        "$orderby": "EventDate,EventId",
        "$top": "1000",
    }
    url = f"https://webapi.legistar.com/v1/{city.legistar_client}/events"
    events: list[dict[str, Any]] = []
    while True:
        response = _get(client, url, params={**params, "$skip": str(len(events))})
        response.raise_for_status()
        page = response.json()
        events += page
        if len(page) < 1000:
            break
    agendas = [e["EventAgendaFile"] for e in events if e["EventAgendaFile"]]
    minutes = [e["EventMinutesFile"] for e in events if e["EventMinutesFile"]]
    media = [e["EventMedia"] for e in events if e["EventMedia"] is not None]
    pages = [m for m in media if isinstance(m, str) and m.startswith("http")]
    count = len(agendas) + len(minutes) + len(media)
    return Listing(agendas, minutes, _seattle_captions(client, pages), count)


def _granicus(client: httpx.Client, city: CityConfig) -> Listing:
    host = city.granicus_host
    response = _get(
        client, f"https://{host}/ViewPublisher.php", params={"view_id": city.granicus_view_id}
    )
    response.raise_for_status()
    agendas, minutes, captions = [], [], []
    seen = set()
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", response.text, re.S):
        clip = re.search(r"clip_id=(\d+)", row)
        cell = re.search(r"<td[^>]*>(.*?)</td>", row, re.S)
        date = ROW_DATE.search(_text(row))
        if not clip or not cell or not date or clip[1] in seen:
            continue
        name = _text(cell[1])
        day = f"{date[3]}-{MONTHS.index(date[1]) + 1:02d}-{int(date[2]):02d}"
        if not city.date_from <= day <= city.date_to or re.search("cancel", name, re.I):
            continue
        if not re.search("planning commission|council", name, re.I):
            continue
        seen.add(clip[1])
        agenda = re.search(r'href="(//[^"]*AgendaViewer\.php\?[^"]+)"', row)
        if agenda:
            agendas.append("https:" + html.unescape(agenda[1]))
        link = re.search(
            r'<a href="(//[^"]*MinutesViewer\.php\?[^"]+)"[^>]*>'
            r"\s*(Minutes|Meeting Minutes)\s*</a>",
            row,
        )
        if link:
            minutes.append("https:" + html.unescape(link[1]))
        if "MediaPlayer.php" in row:
            captions.append(f"https://{host}/videos/{clip[1]}/captions.vtt")
    count = len(agendas) + len(minutes) + len(captions)
    return Listing(agendas, minutes, iter(captions), count)


def _redirect_target(url: str, location: str) -> str:
    location = urljoin(url, location)
    parts = urlsplit(location)
    if parts.hostname == S3_HOST:
        return "https://s3.amazonaws.com/granicus_production_attachments" + parts.path
    viewer = re.search(r"DocumentViewer\.php\?file=([^&]+\.pdf)", location)
    if viewer:
        return f"https://{urlsplit(url).hostname}/DocumentViewer.php?file={viewer[1]}"
    return location


def _first_pdf(client: httpx.Client, urls: list[str]) -> str:
    if not urls:
        raise ValueError("none listed")
    response = _get(client, urls[0])
    if response.is_redirect:
        target = _redirect_target(urls[0], response.headers["Location"])
        response = _get(client, target, follow_redirects=True)
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise ValueError("not a pdf")
    return ""


def _first_captions(client: httpx.Client, urls: Iterator[str]) -> str:
    # the newest meetings can be listed before their captions exist, so look a few rows deep
    reason = "none listed"
    for url in islice(urls, 5):
        response = _get(client, url, follow_redirects=True)
        body = response.content
        if response.status_code == 200 and len(body) >= 2000 and CUE.search(body):
            return ""
        reason = f"{url}: status {response.status_code}, {len(body)} bytes"
    raise ValueError(reason)


def _documents(count: int) -> str:
    if count < 200:
        raise ValueError(f"{count} documents")
    return str(count)


def _probe_city(client: httpx.Client, city_id: str, city: CityConfig) -> list[ProbeResult]:
    try:
        listing = (_legistar if city.source == "legistar" else _granicus)(client, city)
    except Exception as err:
        return [_failure(f"{city_id} listing", err)]
    return [
        ProbeResult(f"{city_id} listing", True, ""),
        _check(f"{city_id} agenda pdf", _first_pdf, client, listing.agendas),
        _check(f"{city_id} minutes pdf", _first_pdf, client, listing.minutes),
        _check(f"{city_id} captions", _first_captions, client, listing.captions),
        _check(f"{city_id} documents >= 200", _documents, listing.count),
    ]


def probe_sources(transport: httpx.BaseTransport | None = None) -> list[ProbeResult]:
    return _guarded("config", lambda: _sources(transport))


def _sources(transport: httpx.BaseTransport | None) -> list[ProbeResult]:
    results = []
    with httpx.Client(
        timeout=httpx.Timeout(60, connect=10), max_redirects=5, transport=transport
    ) as client:
        for city_id, city in load_cities("full").items():
            results += _probe_city(client, city_id, city)
    return results


def probe_gpu() -> list[ProbeResult]:
    return _guarded("cuda", _gpu)


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
