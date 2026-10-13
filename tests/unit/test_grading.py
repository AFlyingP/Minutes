from datetime import UTC, datetime
from typing import cast

import pytest

from minutes.evals import grader
from minutes.labels.schema import Label
from minutes.llm import Budget, ChatResult


class RecordingClient:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

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
        self.calls.append(
            {
                "purpose": purpose,
                "messages": messages,
                "schema": schema,
                "schema_name": schema_name,
                "sample_index": sample_index,
            }
        )
        parsed = self.responses.pop(0)
        return ChatResult("{}", parsed, "stub", 1, 1, 0.0, 0, False)


def _label(
    label_id: str,
    city: str,
    answer: str,
    passage_text: str,
    *,
    answer_type: str = "free_text",
) -> Label:
    return Label.model_validate(
        {
            "id": label_id,
            "type": "question",
            "city": city,
            "author": "agent",
            "created_utc": datetime(2026, 1, 1, tzinfo=UTC),
            "payload": {
                "question": f"What is the answer for {label_id}?",
                "answerable": True,
                "question_type": "multi_passage",
                "answer": answer,
                "answer_type": answer_type,
                "passages": [
                    {
                        "document_id": f"{city}-minutes-001",
                        "unit_index": 1,
                        "start": 0,
                        "end": len(passage_text),
                        "text": passage_text,
                    }
                ],
            },
        }
    )


def _production_labels() -> list[Label]:
    def letters(value: int) -> str:
        result = ""
        while value:
            value, remainder = divmod(value - 1, 26)
            result = chr(ord("a") + remainder) + result
        return result

    labels = []
    for index in range(48):
        marker = letters(index + 1)
        city = "alder" if index % 2 == 0 else "birch"
        passage = (
            f"record marker{marker} evidence{marker} contract{marker} "
            f"authorizes repair{marker} today{marker}"
        )
        labels.append(
            _label(
                f"q-{city}-{index:03d}",
                city,
                f"reference{marker}",
                passage,
            )
        )
    return labels


def test_exact_number_matches_any_numeric_token() -> None:
    assert grader.grade_exact("number", "184500", "The contract was $184,500.00 in total.")
    assert not grader.grade_exact("number", "184500", "$184,000")


def test_exact_name_is_case_insensitive_substring() -> None:
    assert grader.grade_exact("name", "Maya Chen", "The no vote was cast by MAYA CHEN.")
    assert not grader.grade_exact("name", "Maya Chen", "Maya C. voted no.")


def test_exact_date_accepts_three_formats() -> None:
    for answer in ("2024-03-12", "March 12, 2024", "3/12/2024"):
        assert grader.grade_exact("date", "2024-03-12", answer)
    assert not grader.grade_exact("date", "2024-03-12", "March 13, 2024")


def test_exact_identifier_ignores_spaces_and_punctuation() -> None:
    assert grader.grade_exact("identifier", "RESOLUTION 2024-07", "resolution 2024-07")
    assert grader.grade_exact("identifier", "RESOLUTION 2024-07", "RESOLUTION 2024 07")
    assert not grader.grade_exact("identifier", "RESOLUTION 2024-07", "Resolution No. 2024-07")
    assert grader.grade_exact("identifier", "CB 120746", "CB120746")


def test_free_text_uses_model_grader_and_exact_types_do_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = RecordingClient([{"correct": True}])
    monkeypatch.setattr(grader, "get_client", lambda: client)
    free_text = _label("q-free", "alder", "the motion passed", "The motion passed.")
    number = _label("q-number", "alder", "184500", "$184,500", answer_type="number")

    assert grader.grade_answer(free_text, "The motion passed unanimously.", 3)
    assert grader.grade_answer(number, "The contract was $184,500.00.", 3)
    assert len(client.calls) == 1
    assert client.calls[0]["purpose"] == "grade"
    assert client.calls[0]["sample_index"] == 3


def test_answer_grader_prompt_has_no_gold_passage(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RecordingClient([{"correct": True}])
    monkeypatch.setattr(grader, "get_client", lambda: client)
    label = _label("q-private", "alder", "the motion passed", "SECRET GOLD PASSAGE")

    assert grader.grade_answer(label, "The motion passed.", 0)
    messages = cast(list[dict[str, str]], client.calls[0]["messages"])
    assert all("SECRET GOLD PASSAGE" not in message["content"] for message in messages)


def test_support_grader_one_call_per_answer_and_missing_verdict_is_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = RecordingClient([{"verdicts": [{"sentence": 1, "supported": True}]}])
    monkeypatch.setattr(grader, "get_client", lambda: client)

    verdicts = grader.grade_support(
        [
            ("The park opened in May.", ["The park opened in May."]),
            ("The library opened in June.", ["The library opened in June."]),
        ],
        0,
    )

    assert verdicts == [True, False]
    assert len(client.calls) == 1


def test_number_not_in_sources_is_unsupported_without_a_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = RecordingClient([])
    monkeypatch.setattr(grader, "get_client", lambda: client)

    assert grader.grade_support([("It cost $999,999.", ["The amount was $184,500."])], 0) == [False]
    assert client.calls == []


def test_small_counts_are_exempt_from_the_number_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RecordingClient([{"verdicts": [{"sentence": 1, "supported": True}]}])
    monkeypatch.setattr(grader, "get_client", lambda: client)

    assert grader.grade_support([("The vote was 4 to 1.", ["Names listed: Kim and Lee."])], 0) == [
        True
    ]
    assert len(client.calls) == 1


def test_validation_cases_are_96_and_96_for_production_shaped_labels() -> None:
    answers, support = grader.validation_cases(_production_labels())

    assert len(answers) == 96
    assert len(support) == 96


def test_validation_pairs_wrong_answer_from_next_question_same_city() -> None:
    labels = [
        _label("q-alder-001", "alder", "one", "record markeraa evidenceaa contractaa"),
        _label("q-alder-002", "alder", "two", "record markerab evidenceab contractab"),
        _label("q-alder-003", "alder", "three", "record markerac evidenceac contractac"),
    ]

    answers, _ = grader.validation_cases(labels)

    assert [case.answer for case in answers] == ["one", "two", "two", "three", "three", "one"]
    assert [case.expected for case in answers] == [True, False, True, False, True, False]


def test_validate_reports_accuracy_and_kappa_with_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from minutes.llm import stub

    monkeypatch.setattr(stub, "record_completed_call", lambda *args, **kwargs: None)
    monkeypatch.setattr(grader, "get_client", stub.StubClient)

    validation = grader.validate(_production_labels())

    assert validation.answer.accuracy == 1.0
    assert validation.answer.kappa == 1.0
    assert validation.support.accuracy == 1.0
    assert validation.support.kappa == 1.0
