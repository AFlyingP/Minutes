from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

LabelType = Literal["question", "agenda_count", "extraction", "agent_task"]
QuestionType = Literal["vote", "amount", "legislation", "person", "date", "multi_passage"]
AnswerType = Literal["number", "name", "date", "identifier", "free_text"]
Reason = Literal[
    "entity_not_in_corpus", "date_outside_corpus", "fact_not_recorded", "false_premise"
]
FactKind = Literal["motion", "vote", "ordinance", "amount", "statement"]
Count = Annotated[int, Field(ge=0)]
Text = Annotated[str, Field(max_length=300)]


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Passage(_Payload):
    document_id: str
    unit_index: int
    start: int
    end: int
    text: str


class QuestionAnswerable(_Payload):
    question: str
    answerable: Literal[True]
    question_type: QuestionType
    answer: str
    answer_type: AnswerType
    passages: Annotated[list[Passage], Field(min_length=1, max_length=3)]


class AbsenceSearch(_Payload):
    query: Annotated[str, Field(min_length=2, max_length=100)]
    hits: Count


class QuestionUnanswerable(_Payload):
    question: str
    answerable: Literal[False]
    reason: Reason
    absence_searches: Annotated[list[AbsenceSearch], Field(min_length=3)]


class AgendaItem(_Payload):
    identifier: str
    title: Annotated[str, Field(max_length=120)]
    start_page: Annotated[int, Field(ge=1)]


class AgendaCount(_Payload):
    meeting_id: str
    count: Count
    items: list[AgendaItem]


class _Fact(_Payload):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class _Motion(_Fact):
    text: Text
    mover: str | None
    seconder: str | None
    outcome: Literal["passed", "failed", "withdrawn", "tabled", "unknown"]


class _VoteMember(_Fact):
    name: str
    value: Literal["aye", "no", "abstain", "absent"]


class _Vote(_Fact):
    subject_identifier: str | None
    ayes: Count
    noes: Count
    abstain: Count
    absent: Count
    members: list[_VoteMember]
    outcome: Literal["passed", "failed"]


class _Ordinance(_Fact):
    identifier: str
    legislation_type: Literal["ordinance", "resolution", "council_bill", "other"]
    title: Text
    status: Literal["introduced", "passed", "adopted", "failed", "referred", "held", "unknown"]


class _Amount(_Fact):
    amount_usd: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    purpose: Text
    payee: str | None


class _Statement(_Fact):
    speaker: str
    role: str | None
    summary: Text


_FACTS: dict[str, type[BaseModel]] = {
    "motion": _Motion,
    "vote": _Vote,
    "ordinance": _Ordinance,
    "amount": _Amount,
    "statement": _Statement,
}


class ExtractionCase(_Payload):
    kind: FactKind
    passage: Passage
    record: dict[str, object]

    @field_validator("record")
    @classmethod
    def validate_record(cls, record: dict[str, object], info: ValidationInfo) -> dict[str, object]:
        kind = info.data.get("kind")
        if kind is None:
            return record
        return _FACTS[kind].model_validate(record).model_dump()


class AgentTask(_Payload):
    question: str
    answer: str
    answer_type: AnswerType
    document_ids: Annotated[list[str], Field(min_length=2)]


class LabelIn(_Payload):
    type: LabelType
    city: str
    author: Literal["agent", "human"]
    note: Annotated[str, Field(max_length=1000)] = ""
    payload: dict[str, object]


class Label(_Payload):
    id: str
    type: LabelType
    city: str
    author: Literal["agent", "human"]
    created_utc: datetime
    human_reviewed: bool = False
    note: Annotated[str, Field(max_length=1000)] = ""
    payload: dict[str, object]


def parse_payload(label_type: str, payload: dict[str, object]) -> BaseModel:
    if label_type == "question":
        model: type[BaseModel] = (
            QuestionAnswerable if payload.get("answerable") is True else QuestionUnanswerable
        )
    else:
        models: dict[str, type[BaseModel]] = {
            "agenda_count": AgendaCount,
            "extraction": ExtractionCase,
            "agent_task": AgentTask,
        }
        if label_type not in models:
            raise ValueError(f"unknown label type {label_type}")
        model = models[label_type]
    return model.model_validate(payload)
