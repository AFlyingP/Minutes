from collections import Counter
from dataclasses import dataclass

from minutes.labels.schema import (
    Label,
    QuestionAnswerable,
    QuestionUnanswerable,
    parse_payload,
)

QUESTION_TYPES = ("vote", "amount", "legislation", "person", "date", "multi_passage")
REASONS = ("entity_not_in_corpus", "date_outside_corpus", "fact_not_recorded", "false_premise")


@dataclass(frozen=True)
class LabelRules:
    cities: tuple[str, ...]
    questions: int
    questions_per_city: int
    answerable_per_city: int
    unanswerable_per_city: int
    questions_per_type: int | None
    min_documents: int
    max_questions_per_document: int
    min_questions_per_kind: int | None
    min_questions_per_reason: int | None
    counts: int
    counts_per_city: int
    extraction: int
    extraction_per_city: int
    extraction_per_kind: tuple[tuple[str, int], ...]
    agent_tasks: int
    agent_tasks_per_city: int


PRODUCTION_RULES = LabelRules(
    cities=("seattle", "lincoln"),
    questions=60,
    questions_per_city=30,
    answerable_per_city=24,
    unanswerable_per_city=6,
    questions_per_type=4,
    min_documents=10,
    max_questions_per_document=4,
    min_questions_per_kind=4,
    min_questions_per_reason=1,
    counts=20,
    counts_per_city=10,
    extraction=220,
    extraction_per_city=110,
    extraction_per_kind=(
        ("motion", 20),
        ("vote", 30),
        ("ordinance", 20),
        ("amount", 20),
        ("statement", 20),
    ),
    agent_tasks=20,
    agent_tasks_per_city=10,
)
FIXTURE_RULES = LabelRules(
    cities=("alder", "birch"),
    questions=4,
    questions_per_city=2,
    answerable_per_city=1,
    unanswerable_per_city=1,
    questions_per_type=None,
    min_documents=1,
    max_questions_per_document=4,
    min_questions_per_kind=None,
    min_questions_per_reason=None,
    counts=4,
    counts_per_city=2,
    extraction=6,
    extraction_per_city=3,
    extraction_per_kind=(),
    agent_tasks=2,
    agent_tasks_per_city=1,
)


def check(
    label_type: str, labels: list[Label], rules: LabelRules, doc_kinds: dict[str, str]
) -> list[str]:
    messages: list[str] = []
    selected = [label for label in labels if label.type == label_type]
    cities = [*rules.cities, *sorted({label.city for label in selected} - set(rules.cities))]

    def compare(rule: str, expected: int, found: int, scope: str, bound: str = "") -> None:
        fails = (
            found < expected
            if bound == "at least "
            else found > expected
            if bound == "at most "
            else found != expected
        )
        if fails:
            messages.append(f"{rule}: expected {bound}{expected}, found {found} ({scope})")

    if label_type == "question":
        compare("Q-1", rules.questions, len(selected), "total")
        answerable: dict[str, list[QuestionAnswerable]] = {city: [] for city in cities}
        unanswerable: dict[str, list[QuestionUnanswerable]] = {city: [] for city in cities}
        for label in selected:
            payload = parse_payload(label.type, label.payload)
            if isinstance(payload, QuestionAnswerable):
                answerable[label.city].append(payload)
            elif isinstance(payload, QuestionUnanswerable):
                unanswerable[label.city].append(payload)
        for city in cities:
            compare(
                "Q-2", rules.questions_per_city, sum(label.city == city for label in selected), city
            )
        for city in cities:
            compare("Q-3", rules.answerable_per_city, len(answerable[city]), f"{city} answerable")
            compare(
                "Q-3", rules.unanswerable_per_city, len(unanswerable[city]), f"{city} unanswerable"
            )
        if rules.questions_per_type is not None:
            for city in cities:
                type_counts: Counter[str] = Counter(
                    question.question_type for question in answerable[city]
                )
                for kind in QUESTION_TYPES:
                    compare("Q-4", rules.questions_per_type, type_counts[kind], f"{city} {kind}")
        documents = {
            city: Counter(
                document
                for question in answerable[city]
                for document in {p.document_id for p in question.passages}
            )
            for city in cities
        }
        for city in cities:
            compare("Q-5", rules.min_documents, len(documents[city]), city, "at least ")
        for city in cities:
            for document, count in sorted(documents[city].items()):
                compare(
                    "Q-6", rules.max_questions_per_document, count, f"{city} {document}", "at most "
                )
        if rules.min_questions_per_kind is not None:
            for city in cities:
                kind_counts = Counter(
                    kind
                    for question in answerable[city]
                    for kind in {doc_kinds.get(p.document_id) for p in question.passages}
                )
                for kind in ("agenda", "minutes", "transcript"):
                    compare(
                        "Q-7",
                        rules.min_questions_per_kind,
                        kind_counts[kind],
                        f"{city} {kind}",
                        "at least ",
                    )
        if rules.min_questions_per_reason is not None:
            for city in cities:
                reason_counts: Counter[str] = Counter(
                    question.reason for question in unanswerable[city]
                )
                for reason in REASONS:
                    compare(
                        "Q-8",
                        rules.min_questions_per_reason,
                        reason_counts[reason],
                        f"{city} {reason}",
                        "at least ",
                    )
    else:
        totals = {
            "agenda_count": ("C-1", rules.counts, rules.counts_per_city),
            "extraction": ("X-1", rules.extraction, rules.extraction_per_city),
            "agent_task": ("A-1", rules.agent_tasks, rules.agent_tasks_per_city),
        }
        if label_type not in totals:
            raise ValueError(f"unknown label type {label_type}")
        rule, total, per_city = totals[label_type]
        compare(rule, total, len(selected), "total")
        for city in cities:
            compare(rule, per_city, sum(label.city == city for label in selected), city)
        if label_type == "extraction":
            for city in cities:
                extraction_counts = Counter(
                    str(label.payload["kind"]) for label in selected if label.city == city
                )
                for kind, expected in rules.extraction_per_kind:
                    compare("X-2", expected, extraction_counts[kind], f"{city} {kind}")
    return messages
