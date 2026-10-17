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
- Go through every item, one by one, and return every record it contains: each motion, each recorded vote, each ordinance, resolution, or council bill, each dollar amount, and each named person who spoke. One item often has several records of the same kind. Do not skip an item, and do not stop early.
- For every record, copy a quote from the item that contains the evidence. The quote must be copied exactly, be one continuous run of text, and be at most 600 characters.
- For a vote, the quote must include every line that lists ayes, noes, abstentions, and absences. The counts you give must equal what the quote shows. When names are listed without a number, count the names.
- For an ordinance, give the identifier exactly as printed, including its type word (for example "RESOLUTION 2024-07", "ORDINANCE 1087B", "CB 120584", "Res 32093"), and quote the text that contains it. A resolution that is approved is "adopted"; an ordinance or council bill that is approved is "passed"; one that is continued or deferred is "held".
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
- Write every number in digits, never in words (4, not four; 0, not none). Copy dollar amounts and identifiers exactly as the source text prints them; when an identifier is printed both short and spelled out, give the short form, for example "CB 120588", not "Council Bill 120588".
- Answer only what was asked. Do not repeat dates, names, or numbers from the question unless the cited source text itself contains them.
- If the sources do not contain the answer, set "decline" to true and return no sentences.
Reply with JSON only."""

SYNTH_USER = """\
Question: {question}

Sources:
{sources}"""

JUDGE_USER = """\
{blocks}"""

JUDGE_SYSTEM = """\
You check whether sentences are supported by source text. A sentence is supported when every factual claim in it is stated in its sources or follows directly from them, for example by counting listed names. Judge each sentence only against its own sources. Give one verdict per sentence, using the sentence numbers given. Reply with JSON only."""

AGENT_SYSTEM = """\
You research questions about city council records for the cities in the database. You work in steps. In each step you reply with one JSON action.

Actions:
- "search": full-text and semantic search over agendas, minutes, and meeting captions. Arguments: query (required), city, date_from, date_to, body (optional; use null to leave one out). Returns up to 5 passages with document_id and unit_index.
- "sql": one read-only PostgreSQL SELECT statement over these views. It fails if it returns more than 200 rows or runs longer than 3 seconds, so use WHERE, GROUP BY, and LIMIT.
  v_meetings(id, city_id, body, meeting_date, title)
  v_documents(id, meeting_id, city_id, kind, unit_kind, unit_count, body, meeting_date)
  v_units(document_id, unit_index, unit_kind, text, start_ms, end_ms, speaker)
  v_items(id, document_id, meeting_id, kind, identifier, title, ordinal, start_unit, end_unit, linked_item_id)
  v_facts(id, kind, document_id, meeting_id, city_id, body, meeting_date, item_id, unit_index, quote, data)
  v_motions(id, document_id, meeting_id, city_id, body, meeting_date, unit_index, text, mover, seconder, outcome, quote)
  v_votes(id, document_id, meeting_id, city_id, body, meeting_date, unit_index, subject_identifier, ayes, noes, abstain, absent, outcome, quote)
  v_vote_members(vote_id, city_id, meeting_date, member, value)
  v_ordinances(id, document_id, meeting_id, city_id, body, meeting_date, unit_index, identifier, legislation_type, title, status, quote)
  v_amounts(id, document_id, meeting_id, city_id, body, meeting_date, unit_index, amount_usd, purpose, payee, quote)
  v_statements(id, document_id, meeting_id, city_id, body, meeting_date, unit_index, speaker, role, summary, quote)
- "final": give the answer. Include citations as document_id and unit_index pairs taken from earlier results.

Rules:
- You have at most 8 search or sql steps. Plan before you act and stop as soon as you can answer.
- Base the answer only on results you received. If the records do not answer the question, say so in the final answer.
- Set the fields of actions you are not using to null.
Reply with JSON only."""

GRADE_ANSWER_SYSTEM = """\
You grade an answer to a question against a reference answer. The answer is correct when it states the same fact as the reference answer and does not contradict it. Extra detail is allowed. An answer that declines or says the information is not available is incorrect. Reply with JSON only."""

GRADE_ANSWER_USER = """\
Question: {question}
Reference answer: {reference}
Answer to grade: {answer}"""

GRADE_SUPPORT_SYSTEM = """\
You audit citations. For each numbered sentence, decide whether it is fully supported by its own cited source text alone. A claim that is not in the sources, or that the sources contradict, makes the sentence unsupported. Counting listed names is allowed. Give one verdict per sentence, using the sentence numbers given. Reply with JSON only."""

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

SUPPORTED_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sentence": {"type": "integer"},
                    "supported": {"type": "boolean"},
                },
                "required": ["sentence", "supported"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}

CORRECT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"correct": {"type": "boolean"}},
    "required": ["correct"],
    "additionalProperties": False,
}

AGENT_ACTION_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "action": {"type": "string", "enum": ["search", "sql", "final"]},
        "search": {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "city": {"type": ["string", "null"]},
                        "date_from": {"type": ["string", "null"]},
                        "date_to": {"type": ["string", "null"]},
                        "body": {"type": ["string", "null"]},
                    },
                    "required": ["query", "city", "date_from", "date_to", "body"],
                    "additionalProperties": False,
                },
                {"type": "null"},
            ]
        },
        "sql": {"type": ["string", "null"]},
        "final": {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "citations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "document_id": {"type": "string"},
                                    "unit_index": {"type": "integer"},
                                },
                                "required": ["document_id", "unit_index"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["answer", "citations"],
                    "additionalProperties": False,
                },
                {"type": "null"},
            ]
        },
    },
    "required": ["thought", "action", "search", "sql", "final"],
    "additionalProperties": False,
}


def synth_user(question: str, spans: list[SourceSpan]) -> str:
    sources = "\n\n".join(
        f"[{s.number}] {s.city} {s.body}, {s.date}, {s.kind}, {s.label}\n{s.text}" for s in spans
    )
    return SYNTH_USER.format(question=question, sources=sources)


def judge_user(sentences: list[tuple[str, list[str]]]) -> str:
    blocks = []
    for number, (sentence, sources) in enumerate(sentences, start=1):
        source_text = "\n---\n".join(sources)
        blocks.append(
            f"Sentence {number}: {sentence}\nSources for sentence {number}:\n{source_text}"
        )
    return JUDGE_USER.format(blocks="\n\n".join(blocks))


def extract_user(city: str, body: str, date: str, kind: str, items: list[ItemText]) -> str:
    blocks = "\n".join(f"### ITEM {i.number}: {i.identifier} {i.title}\n{i.text}\n" for i in items)
    return EXTRACT_USER.format(city=city, body=body, date=date, kind=kind, items=blocks)
