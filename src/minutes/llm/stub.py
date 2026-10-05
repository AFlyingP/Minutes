"""A rule-based stand-in for the language model, so tests and the demo need no endpoint."""

import json
import math
import re
from typing import Any

from minutes.config import get_settings
from minutes.errors import ConfigError, LLMQuotaError
from minutes.llm import Budget, ChatResult

VOTE_LABELS = ("In Favor:", "Opposed:", "Ayes:", "Noes:", "Nays:", "Abstain:", "Absent:")
CATEGORY_LABELS = {
    "ayes": "In Favor|Ayes|Aye|Yeas|Yes",
    "noes": "Opposed|Noes|Nays|No",
    "abstain": "Abstain|Abstained|Abstentions|Abstaining",
    "absent": "Absent|Excused",
}
CATEGORY_WORDS = {
    "ayes": "Aye|Yes|Yea",
    "noes": "No|Nay",
    "abstain": "Abstain|Abstained",
    "absent": "Absent|Excused",
}
MEMBER_VALUES = {"ayes": "aye", "noes": "no", "abstain": "abstain", "absent": "absent"}
LABEL = re.compile(
    r"\b(?:" + "|".join(f"(?P<{c}>{labels})" for c, labels in CATEGORY_LABELS.items()) + r"):"
)
TABLE_LINE = re.compile(
    r"\|\s*(?:" + "|".join(f"(?P<{c}>{words})" for c, words in CATEGORY_WORDS.items()) + r")\s*$",
    re.I,
)
SOURCE = re.compile(r"^\[(\d+)\] [^\n]*\n(.*?)(?=\n\n\[\d+\] |\Z)", re.M | re.S)
ITEM = re.compile(r"^### ITEM (\d+): (\S*) ?([^\n]*)\n(.*?)(?=\n\n### ITEM \d+: |\Z)", re.M | re.S)
SUBJECT = re.compile(r"RESOLUTION \d{4}-\d+|ORDINANCE \d+[A-Z]?|(?:CB|Res|CF|Appt|Min|IRC|Inf) \d+")
LEGISLATION = re.compile(r"RESOLUTION \d{4}-\d+|ORDINANCE \d+[A-Z]?|CB \d+|Res \d+")
LEGISLATION_TYPES = {
    "RESOLUTION": "resolution",
    "Res": "resolution",
    "ORDINANCE": "ordinance",
    "CB": "council_bill",
}
AMOUNT = re.compile(r"\$\d[\d,]*(?:\.\d+)?")
MOTION = re.compile(
    r"(?:Councilmember|Commissioner) (\w+) motioned to (\w+), motion was seconded by"
    r"\s+(?:Councilmember|Commissioner)\s+(\w+)"
)
_quota_fault = ""
_quota_calls = 0


def _labelled(text: str) -> list[tuple[str, str]]:
    """(category, text up to the next label) for each vote label in the text."""
    matches = list(LABEL.finditer(text))
    starts = [m.start() for m in matches[1:]]
    return [
        (str(m.lastgroup), text[m.end() : end])
        for m, end in zip(matches, [*starts, len(text)], strict=False)
    ]


def _names(listed: str) -> list[str]:
    listed = re.sub(r"^\s*\d{1,2}\s*[-–]", "", listed).strip()
    if listed.startswith("None"):
        return []
    return [name.strip() for name in listed.split(",") if name.strip()]


def quoted_counts(quote: str) -> dict[str, int]:
    """Votes per category that a quote shows, from its label lines and its table lines."""
    counts = dict.fromkeys(CATEGORY_LABELS, 0)
    for category, listed in _labelled(" ".join(quote.split())):
        number = re.search(r"\b(\d{1,2})\s*[-–]", listed)
        counts[category] += int(number[1]) if number else len(_names(listed))
    for line in quote.split("\n"):
        row = TABLE_LINE.search(" ".join(line.split()))
        if row:
            counts[str(row.lastgroup)] += 1
    return counts


def _is_vote_line(line: str) -> bool:
    return any(label in line for label in VOTE_LABELS) or TABLE_LINE.search(line) is not None


def _vote(identifier: str, text: str) -> dict[str, Any] | None:
    lines = text.split("\n")
    marked = [i for i, line in enumerate(lines) if _is_vote_line(line)]
    if not marked:
        return None
    quoted = lines[marked[0] : marked[-1] + 1]
    members = []
    for line in quoted:
        for category, listed in _labelled(line):
            members += [{"name": n, "value": MEMBER_VALUES[category]} for n in _names(listed)]
        row = TABLE_LINE.search(line)
        if row:
            name = line[: line.rindex("|")].strip()
            members.append({"name": name, "value": MEMBER_VALUES[str(row.lastgroup)]})
    counts = quoted_counts("\n".join(quoted))
    subject = SUBJECT.search(text)
    return {
        "quote": "\n".join(quoted),
        "vote": {
            "subject_identifier": subject[0] if subject else identifier,
            **counts,
            "members": members,
            "outcome": "passed" if counts["ayes"] > counts["noes"] else "failed",
        },
    }


def _line_of(text: str, position: int) -> str:
    start = text.rfind("\n", 0, position) + 1
    end = text.find("\n", position)
    return text[start : len(text) if end == -1 else end]


def _records(number: int, identifier: str, title: str, text: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    vote = _vote(identifier, text)
    if vote:
        found.append({"kind": "vote", **vote})
    for amount in AMOUNT.finditer(text):
        value = float(amount[0].lstrip("$").replace(",", ""))
        payload = {"amount_usd": value, "purpose": title, "payee": None}
        found.append({"kind": "amount", "quote": _line_of(text, amount.start()), "amount": payload})
    for motion in MOTION.finditer(text):
        payload = {
            "text": motion[0],
            "mover": motion[1],
            "seconder": motion[3],
            "outcome": "passed" if "Passed by" in text else "unknown",
        }
        found.append({"kind": "motion", "quote": motion[0], "motion": payload})
    for name in dict.fromkeys(LEGISLATION.findall(text)):
        payload = {
            "identifier": name,
            "legislation_type": LEGISLATION_TYPES[name.split()[0]],
            "title": title,
            "status": "unknown",
        }
        quote = _line_of(text, text.index(name))
        found.append({"kind": "ordinance", "quote": quote, "ordinance": payload})
    empty = dict.fromkeys(("motion", "vote", "ordinance", "amount", "statement"))
    return [{"item": number, **empty, **record} for record in found]


def _extract(user: str) -> dict[str, object]:
    records = []
    for item in ITEM.finditer(user):
        records += _records(int(item[1]), item[2], item[3], item[4].rstrip("\n"))
    return {"records": records}


def _synth(user: str) -> dict[str, object]:
    sources = [(int(m[1]), m[2]) for m in SOURCE.finditer(user.split("Sources:\n", 1)[1])]
    for number, text in sources:
        if any(label in text for label in VOTE_LABELS):
            counts = quoted_counts(text)
            sentence = f"The roll call vote was {counts['ayes']} ayes and {counts['noes']} noes."
            return {"decline": False, "sentences": [{"text": sentence, "citations": [number]}]}
    start = sources[0][1].replace("\n", " ")
    if len(start) > 160:
        start = start[:160].rsplit(" ", 1)[0]
    return {"decline": False, "sentences": [{"text": start + ".", "citations": [1]}]}


class StubClient:
    def chat(
        self,
        *,
        purpose: str,
        messages: list[dict[str, str]],
        schema: dict[str, object] | None = None,
        schema_name: str = "output",
        model: str | None = None,
        sample_index: int = 0,
        budget: Budget | None = None,
    ) -> ChatResult:
        global _quota_fault, _quota_calls
        fault = get_settings().fault
        if fault != _quota_fault:
            _quota_fault = fault
            _quota_calls = 0
        if fault.startswith("llm_quota:"):
            quota_count = int(fault.split(":")[1])
            if _quota_calls < quota_count:
                _quota_calls += 1
                raise LLMQuotaError("injected quota fault")
        user = messages[-1]["content"]
        if purpose == "synth":
            parsed = _synth(user)
        elif purpose == "extract":
            parsed = _extract(user)
        else:
            raise ConfigError(f"stub has no rule for purpose {purpose}")
        text = json.dumps(parsed)
        return ChatResult(
            text=text,
            parsed=parsed,
            model="stub",
            prompt_tokens=math.ceil(sum(len(m["content"]) for m in messages) / 4),
            completion_tokens=math.ceil(len(text) / 4),
            cost_usd=0.0,
            latency_ms=0,
            cached=False,
        )
