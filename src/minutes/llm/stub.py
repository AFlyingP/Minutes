"""A rule-based stand-in for the language model, so tests and the demo need no endpoint."""

import json
import math
import re
from typing import Any

from minutes.config import get_settings, load_models
from minutes.errors import BudgetExceeded, ConfigError, LLMQuotaError
from minutes.extract.verify import MEMBER_VALUES, TABLE_LINE, labelled, names, quoted_counts
from minutes.llm import Budget, ChatResult
from minutes.llm.cache import cache_key
from minutes.llm.client import PER_CALL_CEILINGS, record_completed_call
from minutes.llm.prices import cost_usd, estimate_usd
from minutes.llm.prompts import CORRECT_SCHEMA, SUPPORTED_SCHEMA
from minutes.log import get_correlation_id
from minutes.models.stubs import stub_tokens
from minutes.tracing import llm_span

VOTE_LABELS = ("In Favor:", "Opposed:", "Ayes:", "Noes:", "Nays:", "Abstain:", "Absent:")
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
GRADE_SENTENCE = re.compile(
    r"(?:^|\n\n)Sentence (?P<number>\d+): (?P<sentence>.*?)\n"
    r"Sources for sentence (?P=number):\n(?P<sources>.*?)(?=\n\nSentence \d+: |\Z)",
    re.S,
)
_quota_fault = ""
_quota_calls = 0


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
        for category, listed in labelled(line):
            members += [{"name": n, "value": MEMBER_VALUES[category]} for n in names(listed)]
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


def _grade_support(user: str) -> dict[str, object]:
    verdicts = []
    for match in GRADE_SENTENCE.finditer(user):
        sentence_tokens = set(stub_tokens(match["sentence"]))
        source_tokens = set(stub_tokens(match["sources"]))
        supported = not sentence_tokens or (
            len(sentence_tokens & source_tokens) >= len(sentence_tokens) / 2
        )
        verdicts.append({"sentence": int(match["number"]), "supported": supported})
    return {"verdicts": verdicts}


def _grade_correct(user: str) -> dict[str, object]:
    graded, answer = user.split("\nAnswer to grade: ", 1)
    reference = graded.split("\nReference answer: ", 1)[1]
    return {"correct": " ".join(reference.lower().split()) in " ".join(answer.lower().split())}


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
        models = load_models()
        try:
            purpose_config = models.purposes[purpose]
        except KeyError as err:
            raise ConfigError(f"stub has no rule for purpose {purpose}") from err
        resolved_model = model or (
            models.strong if purpose_config.model == "strong" else models.cheap
        )
        token_params: dict[str, object] = {
            "max_completion_tokens": purpose_config.max_tokens,
            "reasoning_effort": purpose_config.reasoning_effort,
            "schema_name": schema_name,
            "schema": schema,
        }
        key = cache_key(resolved_model, messages, token_params, sample_index)
        prompt_tokens = math.ceil(sum(len(message["content"]) for message in messages) / 4)

        fault = get_settings().fault
        if fault != _quota_fault:
            _quota_fault = fault
            _quota_calls = 0
        with llm_span(purpose, resolved_model) as span:
            span.set_attribute("minutes.sample_index", sample_index)
            span.set_attribute("minutes.correlation_id", get_correlation_id())
            if fault.startswith("llm_quota:"):
                quota_count = int(fault.split(":")[1])
                if _quota_calls < quota_count:
                    _quota_calls += 1
                    raise LLMQuotaError("injected quota fault")
            estimate = estimate_usd(resolved_model, messages, purpose_config.max_tokens)
            ceiling = PER_CALL_CEILINGS[purpose]
            if estimate > ceiling:
                raise BudgetExceeded(
                    f"call estimate {estimate:g} exceeds per-call ceiling {ceiling:g}"
                )
            if budget is not None:
                budget.check(estimate)
            user = messages[-1]["content"]
            if purpose == "synth":
                parsed = _synth(user)
            elif purpose in ("judge", "grade") and schema == SUPPORTED_SCHEMA:
                parsed = _grade_support(user)
            elif purpose == "grade" and schema == CORRECT_SCHEMA:
                parsed = _grade_correct(user)
            elif purpose == "extract":
                parsed = _extract(user)
            else:
                raise ConfigError(f"stub has no rule for purpose {purpose}")
            text = json.dumps(parsed)
            completion_tokens = math.ceil(len(text) / 4)
            cost = cost_usd(resolved_model, prompt_tokens, completion_tokens)
            if budget is not None:
                budget.add(cost)
            result = ChatResult(
                text=text,
                parsed=parsed,
                model=resolved_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost,
                latency_ms=0,
                cached=False,
            )
            record_completed_call(
                span,
                purpose=purpose,
                model=resolved_model,
                key=key,
                cached=False,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=cost,
                latency_ms=0,
            )
            return result
