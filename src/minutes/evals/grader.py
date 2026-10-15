import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import cast

from minutes.answer import faithfulness
from minutes.evals.metrics import cohen_kappa
from minutes.labels.schema import Label, QuestionAnswerable, parse_payload
from minutes.llm import get_client
from minutes.llm.prompts import (
    CORRECT_SCHEMA,
    GRADE_ANSWER_SYSTEM,
    GRADE_ANSWER_USER,
    GRADE_SUPPORT_SYSTEM,
    JUDGE_USER,
    SUPPORTED_SCHEMA,
)


@dataclass(frozen=True)
class AnswerCase:
    question: str
    gold: str
    answer: str
    expected: bool


@dataclass(frozen=True)
class SupportCase:
    sentence: str
    sources: list[str]
    expected: bool


@dataclass(frozen=True)
class GraderScore:
    accuracy: float | None
    kappa: float | None


@dataclass(frozen=True)
class GraderValidation:
    answer: GraderScore
    support: GraderScore


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _normalized_identifier(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()


def grade_exact(answer_type: str, gold: str, answer_text: str) -> bool:
    """Grade a number, name, date, or identifier using its exact-match rule."""
    if answer_type == "number":
        expected_number = Decimal(gold)
        return any(
            Decimal(token) == expected_number for token in faithfulness.numeric_tokens(answer_text)
        )
    if answer_type == "name":
        return _normalized_text(gold) in _normalized_text(answer_text)
    if answer_type == "date":
        expected_date = date.fromisoformat(gold)
        formats = (
            expected_date.isoformat(),
            f"{expected_date.strftime('%B')} {expected_date.day}, {expected_date.year}",
            f"{expected_date.month}/{expected_date.day}/{expected_date.year}",
        )
        return any(re.search(re.escape(value), answer_text, re.IGNORECASE) for value in formats)
    return _normalized_identifier(gold) in _normalized_identifier(answer_text)


def grade_free_text(question: str, gold: str, answer_text: str, sample_index: int) -> bool:
    """Ask the model grader to compare an answer with its reference."""
    result = get_client().chat(
        purpose="grade",
        messages=[
            {"role": "system", "content": GRADE_ANSWER_SYSTEM},
            {
                "role": "user",
                "content": GRADE_ANSWER_USER.format(
                    question=question, reference=gold, answer=answer_text
                ),
            },
        ],
        schema=CORRECT_SCHEMA,
        schema_name="correct",
        sample_index=sample_index,
    )
    parsed = cast(dict[str, object], result.parsed)
    return cast(bool, parsed["correct"])


def grade_answer(question_label: Label, answer_text: str, sample_index: int) -> bool:
    payload = cast(QuestionAnswerable, parse_payload("question", question_label.payload))
    if payload.answer_type == "free_text":
        return grade_free_text(payload.question, payload.answer, answer_text, sample_index)
    return grade_exact(payload.answer_type, payload.answer, answer_text)


def grade_support(sentences: list[tuple[str, list[str]]], sample_index: int) -> list[bool]:
    """Grade the cited sources for every sentence in one answer."""
    client = get_client()

    def judge(batch: list[tuple[str, list[str]]]) -> list[bool]:
        blocks = "\n\n".join(
            f"Sentence {number}: {sentence}\nSources for sentence {number}:\n"
            + "\n---\n".join(sources)
            for number, (sentence, sources) in enumerate(batch, start=1)
        )
        result = client.chat(
            purpose="grade",
            messages=[
                {"role": "system", "content": GRADE_SUPPORT_SYSTEM},
                {"role": "user", "content": JUDGE_USER.format(blocks=blocks)},
            ],
            schema=SUPPORTED_SCHEMA,
            schema_name="supported",
            sample_index=sample_index,
        )
        parsed = cast(dict[str, object], result.parsed)
        verdicts = cast(list[dict[str, object]], parsed["verdicts"])
        by_sentence = {
            cast(int, verdict["sentence"]): cast(bool, verdict["supported"]) for verdict in verdicts
        }
        return [by_sentence.get(number, False) for number in range(1, len(batch) + 1)]

    return faithfulness.check_answer(sentences, judge)


def _different_answer(left: str, right: str) -> bool:
    return _normalized_text(left) != _normalized_text(right)


def validation_cases(labels: list[Label]) -> tuple[list[AnswerCase], list[SupportCase]]:
    """Build paired positive and negative grader cases from answerable labels."""
    by_city: dict[str, list[tuple[Label, QuestionAnswerable]]] = {}
    for label in sorted(labels, key=lambda item: item.id):
        if label.type == "question":
            payload = parse_payload("question", label.payload)
            if isinstance(payload, QuestionAnswerable):
                by_city.setdefault(label.city, []).append((label, payload))

    answer_cases = []
    support_cases = []
    for city_questions in by_city.values():
        for index, (_, payload) in enumerate(city_questions):
            other = next(
                (
                    city_questions[(index + offset) % len(city_questions)]
                    for offset in range(1, len(city_questions))
                    if _different_answer(
                        payload.answer,
                        city_questions[(index + offset) % len(city_questions)][1].answer,
                    )
                ),
                None,
            )
            answer_cases.append(AnswerCase(payload.question, payload.answer, payload.answer, True))
            sources = [passage.text for passage in payload.passages]
            support_cases.append(SupportCase(" ".join(sources), sources, True))
            if other is not None:
                answer_cases.append(
                    AnswerCase(payload.question, payload.answer, other[1].answer, False)
                )
                other_sources = [passage.text for passage in other[1].passages]
                support_cases.append(SupportCase(" ".join(other_sources), sources, False))
    return answer_cases, support_cases


def _score(graded: list[bool], expected: list[bool]) -> GraderScore:
    correct = sum(actual == target for actual, target in zip(graded, expected, strict=True))
    accuracy = correct / len(expected) if expected else None
    return GraderScore(accuracy, cohen_kappa(graded, expected))


def validate(labels: list[Label], sample_index: int = 0) -> GraderValidation:
    """Measure answer and support grader agreement on label-derived cases."""
    answer_cases, support_cases = validation_cases(labels)
    answer_results = [
        grade_free_text(case.question, case.gold, case.answer, sample_index)
        for case in answer_cases
    ]
    support_results = [
        grade_support([(case.sentence, case.sources)], sample_index)[0] for case in support_cases
    ]
    return GraderValidation(
        _score(answer_results, [case.expected for case in answer_cases]),
        _score(support_results, [case.expected for case in support_cases]),
    )
