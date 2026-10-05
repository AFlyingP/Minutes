from datetime import date
from typing import Literal

from pydantic import BaseModel

DocKind = Literal["agenda", "minutes", "transcript"]
SearchMode = Literal["keyword", "vector", "hybrid", "hybrid_rerank"]
Chunker = Literal["fixed", "item"]
Pipeline = Literal["final", "baseline", "langgraph"]

DECLINE_MESSAGE = "I can't answer that from the indexed records."


class HealthOut(BaseModel):
    status: str
    corpus: str
    db: str


class CityOut(BaseModel):
    id: str
    name: str
    state: str
    bodies: list[str]
    date_min: date | None
    date_max: date | None
    documents: int


class CitiesOut(BaseModel):
    cities: list[CityOut]


class SpanOut(BaseModel):
    document_id: str
    unit_index: int
    unit_kind: str
    start_offset: int
    end_offset: int
    start_ms: int | None
    label: str


class HitOut(BaseModel):
    rank: int
    score: float
    document_id: str
    meeting_id: str
    city_id: str
    city_name: str
    body: str
    meeting_date: date
    doc_kind: str
    item_identifier: str | None
    item_title: str | None
    heading: str
    snippet: str
    spans: list[SpanOut]


class SearchOut(BaseModel):
    query: str
    mode: str
    chunker: str
    hits: list[HitOut]


class AskIn(BaseModel):
    question: str
    city: str | None = None
    body: str | None = None
    date_from: date | None = None
    date_to: date | None = None
    kind: DocKind | None = None
    pipeline: Pipeline = "baseline"


class SentenceOut(BaseModel):
    text: str
    citations: list[int]


class CitationOut(BaseModel):
    number: int
    document_id: str
    unit_index: int
    unit_kind: str
    start_offset: int
    end_offset: int
    quote: str
    city_id: str
    body: str
    meeting_date: date
    doc_kind: str
    start_ms: int | None
    label: str


class AskOut(BaseModel):
    declined: bool
    decline_reason: str | None
    decline_message: str | None
    sentences: list[SentenceOut]
    citations: list[CitationOut]
    pipeline: str
    model: str | None
    cost_usd: float
    latency_ms: int
    cached: bool
    dropped_sentences: int
