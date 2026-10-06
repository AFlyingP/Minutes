import ast
from datetime import UTC, datetime
from pathlib import Path

from minutes.labels.rules import FIXTURE_RULES, PRODUCTION_RULES, QUESTION_TYPES, REASONS, check
from minutes.labels.schema import Label


def label(label_type: str, city: str, number: int, payload: dict[str, object]) -> Label:
    return Label.model_validate(
        {
            "id": f"{label_type}-{city}-{number}",
            "type": label_type,
            "city": city,
            "author": "human",
            "created_utc": datetime(2024, 1, 1, tzinfo=UTC),
            "payload": payload,
        }
    )


def question(city: str, number: int, kind: str = "vote", document: str | None = None) -> Label:
    return label(
        "question",
        city,
        number,
        {
            "question": "What happened at this meeting?",
            "answerable": True,
            "question_type": kind,
            "answer": "4",
            "answer_type": "number",
            "passages": [
                {
                    "document_id": document or f"{city}-document-{number % 12}",
                    "unit_index": 1,
                    "start": 0,
                    "end": 10,
                    "text": "source text",
                }
            ],
        },
    )


def absent(city: str, number: int, reason: str = "entity_not_in_corpus") -> Label:
    return label(
        "question",
        city,
        number,
        {
            "question": "Was the quarry approved?",
            "answerable": False,
            "reason": reason,
            "absence_searches": [
                {"query": query, "hits": 0} for query in ("quarry", "mine", "stone")
            ],
        },
    )


def questions() -> list[Label]:
    return [
        case
        for city in PRODUCTION_RULES.cities
        for case in [
            *[question(city, number, QUESTION_TYPES[number // 4]) for number in range(24)],
            *[absent(city, 24 + number, REASONS[number % 4]) for number in range(6)],
        ]
    ]


def test_production_rules_match_golden_set_numbers() -> None:
    rules = PRODUCTION_RULES
    assert (
        rules.questions,
        rules.questions_per_city,
        rules.answerable_per_city,
        rules.unanswerable_per_city,
    ) == (60, 30, 24, 6)
    assert (
        rules.questions_per_type,
        rules.min_documents,
        rules.max_questions_per_document,
        rules.min_questions_per_kind,
        rules.min_questions_per_reason,
    ) == (4, 10, 4, 4, 1)
    assert (rules.counts, rules.counts_per_city) == (20, 10)
    assert (rules.extraction, rules.extraction_per_city) == (220, 110)
    assert dict(rules.extraction_per_kind) == {
        "motion": 20,
        "vote": 30,
        "ordinance": 20,
        "amount": 20,
        "statement": 20,
    }
    assert (rules.agent_tasks, rules.agent_tasks_per_city) == (20, 10)


def test_incomplete_set_reports_rule_and_counts() -> None:
    messages = check("question", questions()[:-1], PRODUCTION_RULES, {})
    assert messages[0].startswith("Q-1: expected 60, found 59")


def test_over_full_set_is_rejected() -> None:
    messages = check("question", [*questions(), question("seattle", 61)], PRODUCTION_RULES, {})
    assert messages[0] == "Q-1: expected 60, found 61 (total)"


def test_per_type_and_per_reason_rules() -> None:
    cases = questions()
    cases[4] = question("seattle", 4, "vote")
    cases[24] = absent("seattle", 24, "false_premise")
    cases[28] = absent("seattle", 28, "false_premise")
    messages = check("question", cases, PRODUCTION_RULES, {})
    assert [message for message in messages if message.startswith("Q-4")] == [
        "Q-4: expected 4, found 5 (seattle vote)",
        "Q-4: expected 4, found 3 (seattle amount)",
    ]
    assert "Q-8: expected at least 1, found 0 (seattle entity_not_in_corpus)" in messages
    assert not any("false_premise" in message for message in messages if message.startswith("Q-8:"))


def test_at_most_4_questions_per_document() -> None:
    cases = [question("seattle", number, document="shared") for number in range(5)]
    messages = check("question", cases, PRODUCTION_RULES, {})
    assert "Q-6: expected at most 4, found 5 (seattle shared)" in messages
    cases = cases[:4]
    cases[0].payload["passages"] = [*cases[0].payload["passages"], *cases[0].payload["passages"]]  # type: ignore[misc]
    assert not any(
        message.startswith("Q-6") for message in check("question", cases, PRODUCTION_RULES, {})
    )


def test_each_document_kind_needs_4_questions() -> None:
    cases = questions()
    doc_kinds = {
        f"{city}-document-{number}": ("agenda", "minutes", "transcript")[number % 3]
        for city in PRODUCTION_RULES.cities
        for number in range(12)
    }
    assert check("question", cases, PRODUCTION_RULES, doc_kinds) == []
    messages = check(
        "question", cases, PRODUCTION_RULES, {document: "agenda" for document in doc_kinds}
    )
    assert [message for message in messages if message.startswith("Q-7")] == [
        f"Q-7: expected at least 4, found 0 ({city} {kind})"
        for city in PRODUCTION_RULES.cities
        for kind in ("minutes", "transcript")
    ]


def test_extraction_split_per_kind_is_exact() -> None:
    cases = [
        label("extraction", city, number, {"kind": kind})
        for city in PRODUCTION_RULES.cities
        for kind, count in PRODUCTION_RULES.extraction_per_kind
        for number in range(count)
    ]
    assert check("extraction", cases, PRODUCTION_RULES, {}) == []
    cases[0].payload["kind"] = "vote"
    messages = check("extraction", cases, PRODUCTION_RULES, {})
    assert messages == [
        "X-2: expected 20, found 19 (seattle motion)",
        "X-2: expected 30, found 31 (seattle vote)",
    ]


def test_complete_fixture_set_passes() -> None:
    cases = []
    for city in FIXTURE_RULES.cities:
        cases.extend([question(city, 1), absent(city, 2)])
        cases.extend(label("agenda_count", city, number, {}) for number in range(2))
        cases.extend(label("extraction", city, number, {"kind": "vote"}) for number in range(3))
        cases.append(label("agent_task", city, 1, {}))
    for label_type in ("question", "agenda_count", "extraction", "agent_task"):
        assert check(label_type, cases, FIXTURE_RULES, {}) == []
    for label_type, rule in (("agenda_count", "C-1"), ("extraction", "X-1"), ("agent_task", "A-1")):
        selected = [case for case in cases if case.type == label_type]
        assert check(label_type, selected[:-1], FIXTURE_RULES, {})[0].startswith(rule)
        assert check(label_type, [*selected, selected[0]], FIXTURE_RULES, {})[0].startswith(rule)


def test_labels_package_imports_no_model_code() -> None:
    forbidden = (
        "minutes.models",
        "minutes.retrieval",
        "minutes.answer",
        "minutes.agent",
        "minutes.evals",
        "minutes.llm",
    )
    root = Path(__file__).resolve().parents[2] / "src" / "minutes" / "labels"
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            imports = []
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    package = ["minutes", "labels", *path.relative_to(root).parts[:-1]]
                    module = ".".join([*package[: len(package) - node.level + 1], module]).rstrip(
                        "."
                    )
                imports = [f"{module}.{alias.name}" for alias in node.names]
            for module in imports:
                assert not any(
                    module == banned or module.startswith(banned + ".") for banned in forbidden
                ), (path, module)
                if module.startswith("minutes.extract"):
                    assert module == "minutes.extract.verify" or module.startswith(
                        "minutes.extract.verify."
                    ), (path, module)
