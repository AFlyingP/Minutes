from datetime import date
from typing import Literal

from pydantic import BaseModel

from minutes.ingest.layout import Box
from minutes.labels.schema import Label, LabelType

DocKind = Literal["agenda", "minutes", "transcript"]
UnitKind = Literal["page", "segment"]
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


class DocumentSummaryOut(BaseModel):
    id: str
    meeting_id: str
    city_id: str
    kind: DocKind
    body: str
    meeting_date: date
    unit_kind: UnitKind
    unit_count: int | None


class DocumentsOut(BaseModel):
    total: int
    documents: list[DocumentSummaryOut]


class MeetingSummaryOut(BaseModel):
    id: str
    body: str
    meeting_date: date
    title: str
    recording_url: str | None


class DocumentOut(BaseModel):
    id: str
    city_id: str
    kind: DocKind
    unit_kind: UnitKind
    unit_count: int | None
    source_url: str
    meeting: MeetingSummaryOut


class UnitOut(BaseModel):
    document_id: str
    unit_index: int
    unit_kind: UnitKind
    text: str
    boxes: list[Box] | None
    text_source: Literal["pdf", "ocr", "caption"]
    start_ms: int | None
    end_ms: int | None
    speaker: str | None
    width_pt: float | None
    height_pt: float | None
    label: str
    recording_link: str | None


class MeetingDocumentOut(BaseModel):
    id: str
    kind: DocKind
    status: str
    unit_count: int | None


class MeetingOut(MeetingSummaryOut):
    city_id: str
    documents: list[MeetingDocumentOut]


class LabelsOut(BaseModel):
    labels: list[Label]


class ReviewIn(BaseModel):
    human_reviewed: bool


class LabelProgress(BaseModel):
    count: int
    violations: list[str]


class LabelsProgressOut(BaseModel):
    frozen: bool
    types: dict[str, LabelProgress]
    human_reviewed: dict[str, int]


class TextMatchOut(BaseModel):
    document_id: str
    unit_index: int
    snippet: str


class TextSearchOut(BaseModel):
    unit_count: int
    matches: list[TextMatchOut]


class SampleMeetingOut(BaseModel):
    meeting_id: str
    body: str
    meeting_date: date
    agenda_document_id: str
    labelled: bool


class SampleMeetingsOut(BaseModel):
    meetings: list[SampleMeetingOut]


class LabelExportIn(BaseModel):
    type: LabelType


class LabelExportOut(BaseModel):
    path: str
    count: int
