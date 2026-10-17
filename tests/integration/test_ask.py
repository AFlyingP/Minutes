from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from minutes.api.app import create_app
from minutes.config import get_settings
from minutes.evals.runner import run_suite
from minutes.llm import ChatResult

pytestmark = pytest.mark.integration

DEMO_QUESTION = "How did the council vote on the Granite Works contract in Resolution 2024-07?"


class AnswerClient:
    def __init__(
        self, sentences: list[dict[str, Any]], verdicts: list[dict[str, Any]] | None = None
    ):
        self.sentences = sentences
        self.verdicts = verdicts or []
        self.calls: list[str] = []

    def chat(self, *, purpose: str, **kwargs: Any) -> ChatResult:
        self.calls.append(purpose)
        if purpose == "synth":
            parsed = {"decline": False, "sentences": self.sentences}
        else:
            parsed = {"verdicts": self.verdicts}
        return ChatResult("{}", parsed, "stub", 10, 5, 0.001, 1, False)


def test_demo_step_6_answer_and_citation(fixture_corpus: str) -> None:
    client = TestClient(create_app())
    response = client.post("/api/ask", json={"question": DEMO_QUESTION, "city": "birch"})

    assert response.status_code == 200
    answer = response.json()
    assert answer["declined"] is False
    assert [sentence["text"] for sentence in answer["sentences"]] == [
        "The roll call vote was 4 ayes and 1 noes."
    ]
    assert answer["sentences"][0]["citations"] == [1]
    citation = answer["citations"][0]
    assert (citation["document_id"], citation["unit_index"], citation["label"]) == (
        "birch-minutes-201",
        2,
        "p. 2",
    )


def test_demo_step_7_decline_no_retrieval(fixture_corpus: str) -> None:
    client = TestClient(create_app())
    response = client.post(
        "/api/ask", json={"question": "What was decided about the airport runway extension?"}
    )

    assert response.status_code == 200
    answer = response.json()
    assert answer["declined"] is True
    assert answer["decline_reason"] == "no_retrieval"
    assert answer["decline_message"] == "I can't answer that from the indexed records."
    assert answer["sentences"] == []


def test_every_sentence_has_a_citation_with_locator(fixture_corpus: str) -> None:
    result = (
        TestClient(create_app())
        .post("/api/ask", json={"question": DEMO_QUESTION, "city": "birch"})
        .json()
    )
    citations = {citation["number"]: citation for citation in result["citations"]}

    assert result["sentences"]
    for sentence in result["sentences"]:
        assert sentence["citations"]
        for number in sentence["citations"]:
            assert number in citations
            assert citations[number]["label"]
            assert citations[number]["unit_index"] > 0


def test_unsupported_sentence_is_dropped_and_counted(
    fixture_corpus: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer_client = AnswerClient(
        [
            {"text": "The council voted 4 to 1.", "citations": [1]},
            {"text": "The roll call is recorded.", "citations": [1]},
            {"text": "The airport opened in 1999.", "citations": [1]},
        ],
        [
            {"sentence": 1, "supported": True},
            {"sentence": 2, "supported": True},
            {"sentence": 3, "supported": False},
        ],
    )
    monkeypatch.setattr("minutes.answer.pipeline.get_client", lambda: answer_client)

    result = (
        TestClient(create_app())
        .post("/api/ask", json={"question": DEMO_QUESTION, "city": "birch"})
        .json()
    )

    assert result["declined"] is False
    assert len(result["sentences"]) == 2
    assert result["dropped_sentences"] == 1
    assert answer_client.calls == ["synth", "judge"]


def test_more_than_half_dropped_declines_unsupported(
    fixture_corpus: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentences = [{"text": f"Supported claim {n}.", "citations": [1]} for n in range(1, 4)]
    answer_client = AnswerClient(
        sentences,
        [
            {"sentence": 1, "supported": True},
            {"sentence": 2, "supported": False},
            {"sentence": 3, "supported": False},
        ],
    )
    monkeypatch.setattr("minutes.answer.pipeline.get_client", lambda: answer_client)

    result = (
        TestClient(create_app())
        .post("/api/ask", json={"question": DEMO_QUESTION, "city": "birch"})
        .json()
    )

    assert result["declined"] is True
    assert result["decline_reason"] == "unsupported"
    assert result["dropped_sentences"] == 2
    assert result["sentences"] == []


def test_model_decline_flag_gives_model_declined(
    fixture_corpus: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DecliningClient:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def chat(self, *, purpose: str, **kwargs: Any) -> ChatResult:
            self.calls.append(purpose)
            return ChatResult(
                "{}", {"decline": True, "sentences": []}, "stub", 10, 5, 0.001, 1, False
            )

    answer_client = DecliningClient()
    monkeypatch.setattr("minutes.answer.pipeline.get_client", lambda: answer_client)
    result = (
        TestClient(create_app())
        .post("/api/ask", json={"question": DEMO_QUESTION, "city": "birch"})
        .json()
    )

    assert result["declined"] is True
    assert result["decline_reason"] == "model_declined"
    assert answer_client.calls == ["synth"]


def test_invalid_output_twice_gives_model_error(
    fixture_corpus: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    class InvalidClient:
        attempts = 0

        def chat(self, **kwargs: Any) -> ChatResult:
            from minutes.errors import LLMOutputError

            self.attempts += 2
            raise LLMOutputError("invalid output")

    invalid = InvalidClient()
    monkeypatch.setattr("minutes.answer.pipeline.get_client", lambda: invalid)
    result = (
        TestClient(create_app())
        .post("/api/ask", json={"question": DEMO_QUESTION, "city": "birch"})
        .json()
    )

    assert result["declined"] is True
    assert result["decline_reason"] == "model_error"
    assert invalid.attempts == 2


def test_budget_exceeded_declines(fixture_corpus: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINUTES_QUERY_BUDGET_USD", "0.000001")
    get_settings.cache_clear()
    try:
        result = (
            TestClient(create_app())
            .post("/api/ask", json={"question": DEMO_QUESTION, "city": "birch"})
            .json()
        )
    finally:
        get_settings.cache_clear()

    assert result["declined"] is True
    assert result["decline_reason"] == "budget_exceeded"


def test_citations_renumbered_in_order_of_first_use(
    fixture_corpus: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer_client = AnswerClient(
        [
            {"text": "The first claim.", "citations": [2]},
            {"text": "The second claim.", "citations": [1]},
        ],
        [{"sentence": 1, "supported": True}, {"sentence": 2, "supported": True}],
    )
    monkeypatch.setattr("minutes.answer.pipeline.get_client", lambda: answer_client)

    result = (
        TestClient(create_app())
        .post("/api/ask", json={"question": DEMO_QUESTION, "city": "birch"})
        .json()
    )

    assert [sentence["citations"] for sentence in result["sentences"]] == [[1], [2]]
    assert [citation["number"] for citation in result["citations"]] == [1, 2]


def test_ask_validation_errors_are_422(fixture_corpus: str) -> None:
    client = TestClient(create_app())
    for request in (
        {"question": ""},
        {"question": "x" * 501},
        {"question": "sidewalk", "pipeline": "unknown"},
    ):
        assert client.post("/api/ask", json=request).status_code == 422


def test_answers_section_includes_final_pipeline_with_five_runs(
    fixture_corpus: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from minutes.evals import runner

    monkeypatch.setattr(runner, "write_results", lambda _path, results: results)
    results = run_suite(
        suite="full",
        controls=True,
        labels_dir=Path("tests/fixtures/labels"),
        out_path=Path("unused.json"),
        only={"answers"},
    )

    answers = cast(dict[str, dict[str, Any]], results["answers"])
    assert set(answers) == {"baseline", "final"}
    for pipeline in answers.values():
        assert pipeline["answer_accuracy"]["runs"] == 5
