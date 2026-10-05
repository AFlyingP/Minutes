# ruff: noqa: E501
from typing import NamedTuple


class SourceSpan(NamedTuple):
    number: int
    city: str
    body: str
    date: str
    kind: str
    label: str
    text: str


class ItemText(NamedTuple):
    number: int
    item_id: int
    identifier: str
    title: str
    text: str
    start_unit: int


EXTRACT_SYSTEM = """\
You extract structured facts from city council agenda and minutes text. Use only the text given.

Record kinds:
- motion: a motion, who moved it, who seconded it, and its outcome.
- vote: a recorded vote with its counts and the members listed for each side.
- ordinance: an ordinance, resolution, or council bill, with its printed identifier and its status at this meeting.
- amount: a dollar amount and what it pays for.
- statement: a named person who spoke, and what they addressed.

Rules:
- For every record, copy a quote from the item that contains the evidence. The quote must be copied exactly, be one continuous run of text, and be at most 600 characters.
- For a vote, the quote must include every line that lists ayes, noes, abstentions, and absences. The counts you give must equal what the quote shows. When names are listed without a number, count the names.
- Set the "item" field to the number of the item the record comes from.
- Fill exactly one of motion, vote, ordinance, amount, statement, matching "kind". Set the others to null.
- Use null for anything the text does not state. Do not guess.
- Return an empty list when the items contain no facts.
Reply with JSON only."""

EXTRACT_USER = """\
City: {city}
Body: {body}
Meeting date: {date}
Document kind: {kind}

{items}"""

SYNTH_SYSTEM = """\
You answer questions about city council records using only the numbered sources.

Rules:
- Write 1 to 5 short sentences.
- For each sentence, list the numbers of the sources that support it. Every sentence must cite at least one source.
- State only what the sources say. Do not use outside knowledge.
- Give vote counts, dollar amounts, dates, and identifiers exactly as the sources give them.
- If the sources do not contain the answer, set "decline" to true and return no sentences.
Reply with JSON only."""

SYNTH_USER = """\
Question: {question}

Sources:
{sources}"""

SYNTH_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "decline": {"type": "boolean"},
        "sentences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "citations": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["text", "citations"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["decline", "sentences"],
    "additionalProperties": False,
}


def synth_user(question: str, spans: list[SourceSpan]) -> str:
    sources = "\n\n".join(
        f"[{s.number}] {s.city} {s.body}, {s.date}, {s.kind}, {s.label}\n{s.text}" for s in spans
    )
    return SYNTH_USER.format(question=question, sources=sources)


def extract_user(city: str, body: str, date: str, kind: str, items: list[ItemText]) -> str:
    blocks = "\n".join(f"### ITEM {i.number}: {i.identifier} {i.title}\n{i.text}\n" for i in items)
    return EXTRACT_USER.format(city=city, body=body, date=date, kind=kind, items=blocks)
