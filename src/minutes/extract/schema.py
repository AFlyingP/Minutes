from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field

FACT_KINDS = ("motion", "vote", "ordinance", "amount", "statement")


def _strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _clip(value: Any) -> Any:
    return value.strip()[:300] if isinstance(value, str) else value


Name = Annotated[str, BeforeValidator(_strip)]
Text = Annotated[str, BeforeValidator(_clip)]
Count = Annotated[int, Field(ge=0)]


class Motion(BaseModel):
    text: Text
    mover: Name | None
    seconder: Name | None
    outcome: Literal["passed", "failed", "withdrawn", "tabled", "unknown"]


class VoteMember(BaseModel):
    name: Name
    value: Literal["aye", "no", "abstain", "absent"]


class Vote(BaseModel):
    subject_identifier: Name | None
    ayes: Count
    noes: Count
    abstain: Count
    absent: Count
    members: list[VoteMember]
    outcome: Literal["passed", "failed"]


class Ordinance(BaseModel):
    identifier: Name
    legislation_type: Literal["ordinance", "resolution", "council_bill", "other"]
    title: Text
    status: Literal["introduced", "passed", "adopted", "failed", "referred", "held", "unknown"]


class Amount(BaseModel):
    amount_usd: Annotated[float, Field(gt=0)]
    purpose: Text
    payee: Name | None


class Statement(BaseModel):
    speaker: Name
    role: Name | None
    summary: Text


FACT_MODELS: dict[str, type[BaseModel]] = {
    "motion": Motion,
    "vote": Vote,
    "ordinance": Ordinance,
    "amount": Amount,
    "statement": Statement,
}


def _object(properties: dict[str, object]) -> dict[str, object]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _nullable(schema: dict[str, object]) -> dict[str, object]:
    return {**schema, "type": [schema["type"], "null"]}


_STRING: dict[str, object] = {"type": "string"}
_COUNT: dict[str, object] = {"type": "integer"}
_OPTIONAL_STRING = _nullable(_STRING)


def _enum(*values: str) -> dict[str, object]:
    return {"type": "string", "enum": list(values)}


_PAYLOADS = {
    "motion": _object(
        {
            "text": _STRING,
            "mover": _OPTIONAL_STRING,
            "seconder": _OPTIONAL_STRING,
            "outcome": _enum("passed", "failed", "withdrawn", "tabled", "unknown"),
        }
    ),
    "vote": _object(
        {
            "subject_identifier": _OPTIONAL_STRING,
            "ayes": _COUNT,
            "noes": _COUNT,
            "abstain": _COUNT,
            "absent": _COUNT,
            "members": {
                "type": "array",
                "items": _object(
                    {"name": _STRING, "value": _enum("aye", "no", "abstain", "absent")}
                ),
            },
            "outcome": _enum("passed", "failed"),
        }
    ),
    "ordinance": _object(
        {
            "identifier": _STRING,
            "legislation_type": _enum("ordinance", "resolution", "council_bill", "other"),
            "title": _STRING,
            "status": _enum(
                "introduced", "passed", "adopted", "failed", "referred", "held", "unknown"
            ),
        }
    ),
    "amount": _object(
        {"amount_usd": {"type": "number"}, "purpose": _STRING, "payee": _OPTIONAL_STRING}
    ),
    "statement": _object({"speaker": _STRING, "role": _OPTIONAL_STRING, "summary": _STRING}),
}

EXTRACTION_SCHEMA: dict[str, object] = _object(
    {
        "records": {
            "type": "array",
            "items": _object(
                {
                    "item": {"type": "integer"},
                    "kind": _enum(*FACT_KINDS),
                    "quote": _STRING,
                    **{kind: _nullable(payload) for kind, payload in _PAYLOADS.items()},
                }
            ),
        }
    }
)
