import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Literal, cast

import psycopg
import pytest
from psycopg.rows import TupleRow

from minutes import db
from minutes.answer import cache
from minutes.answer.pipeline import answer
from minutes.config import get_settings, load_models
from minutes.errors import BudgetExceeded, LLMOutputError
from minutes.evals.runner import run_suite
from minutes.extract import extractor
from minutes.llm import Budget, ChatResult
from minutes.log import bind_correlation_id, new_correlation_id
from minutes.retrieval.search import Filters, Hit, search

pytestmark = pytest.mark.integration
LABELS_DIR = Path("tests/fixtures/labels")


def _separate_models(monkeypatch: pytest.MonkeyPatch, module: object) -> tuple[str, str]:
    models = replace(load_models(), strong="gpt-6.1-sol", cheap="gpt-6-luna")
    monkeypatch.setattr(module, "load_models", lambda: models)
    return models.strong, models.cheap


def test_second_identical_question_is_served_from_cache_at_zero_cost(
    fixture_corpus: str,
) -> None:
    question = "What amount was authorized for Cedar Paving's sidewalk repair contract?"
    with db.connect(fixture_corpus) as conn:
        first_id = new_correlation_id()
        bind_correlation_id(first_id)
        answer(conn, question, Filters(city="alder"))
        before_row = conn.execute("SELECT count(*) FROM llm_calls").fetchone()
        assert before_row is not None
        before = before_row[0]
        cached = answer(conn, question, Filters(city="alder"))
        after_row = conn.execute("SELECT count(*) FROM llm_calls").fetchone()
        assert after_row is not None
        after = after_row[0]

    assert cached.cached
    assert cached.cost_usd == 0.0
    assert before == after
    bind_correlation_id("-")


def test_cache_key_ignores_case_whitespace_and_trailing_question_mark() -> None:
    filters = Filters(city="alder", date_from=date(2024, 1, 1))
    assert cache.cache_key("  What is the total?  ", filters, "final", "v") == cache.cache_key(
        "what   IS the total", filters, "final", "v"
    )


def test_cache_key_changes_with_filters_and_pipeline() -> None:
    first = cache.cache_key("What was approved?", Filters(city="alder"), "final", "v")
    assert first != cache.cache_key("What was approved?", Filters(city="birch"), "final", "v")
    assert first != cache.cache_key("What was approved?", Filters(city="alder"), "baseline", "v")


def test_cache_invalidated_when_chunks_change(fixture_corpus: str) -> None:
    question = "What amount was authorized for Cedar Paving's sidewalk repair contract?"
    with db.connect(fixture_corpus) as conn:
        conn.execute("DELETE FROM answer_cache")
        version = cache.corpus_version(conn)
        first = answer(conn, question, Filters(city="alder"))
        assert not first.cached
        row = conn.execute(
            "SELECT document_id, unit_index, item_id, start_offset, end_offset, text, embed_text, "
            "city_id, body, meeting_date, doc_kind FROM chunks WHERE chunker = 'item' LIMIT 1"
        ).fetchone()
        assert row is not None
        ordinal_row = conn.execute(
            "SELECT coalesce(max(ordinal), 0) + 1 FROM chunks "
            "WHERE document_id = %s AND chunker = 'item'",
            (row[0],),
        ).fetchone()
        assert ordinal_row is not None
        inserted = conn.execute(
            "INSERT INTO chunks (document_id, unit_index, item_id, chunker, ordinal, start_offset, "
            "end_offset, text, embed_text, city_id, body, meeting_date, doc_kind) "
            "VALUES (%s, %s, %s, 'item', %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (*row[:3], ordinal_row[0], *row[3:]),
        ).fetchone()
        assert inserted is not None
        try:
            assert cache.corpus_version(conn) != version
            second = answer(conn, question, Filters(city="alder"))
        finally:
            conn.execute("DELETE FROM chunks WHERE id = %s", (inserted[0],))
    assert not second.cached


def test_controls_off_bypasses_cache_and_routing(
    fixture_corpus: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import minutes.answer.pipeline as pipeline

    strong, cheap = _separate_models(monkeypatch, pipeline)
    question = "What amount was authorized for Cedar Paving's sidewalk repair contract?"
    correlation_ids = [new_correlation_id(), new_correlation_id()]
    with db.connect(fixture_corpus) as conn:
        conn.execute("DELETE FROM answer_cache")
        for correlation_id in correlation_ids:
            bind_correlation_id(correlation_id)
            answer(conn, question, Filters(city="alder"), controls=False)
        rows = conn.execute(
            "SELECT purpose, model FROM llm_calls WHERE correlation_id = ANY(%s)",
            (correlation_ids,),
        ).fetchall()
    bind_correlation_id("-")
    synth_models = {row[1] for row in rows if row[0] == "synth"}
    judge_models = {row[1] for row in rows if row[0] == "judge"}
    assert synth_models == {strong}
    assert judge_models == {cheap}


def test_easy_question_uses_cheap_model(
    fixture_corpus: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace as dc_replace

    import minutes.answer.pipeline as pipeline

    strong, cheap = _separate_models(monkeypatch, pipeline)

    def high_score(
        conn: psycopg.Connection[TupleRow],
        query: str,
        filters: Filters,
        *,
        mode: Literal["keyword", "vector", "hybrid", "hybrid_rerank"],
        chunker: Literal["fixed", "item"],
        k: int,
    ) -> list[Hit]:
        return [
            dc_replace(hit, score=1.0)
            for hit in search(conn, query, filters, mode=mode, chunker=chunker, k=k)
        ]

    monkeypatch.setattr("minutes.answer.pipeline.search", high_score)
    models = (strong, cheap)
    for question, expected in (
        ("What amount was authorized for Cedar Paving's sidewalk repair contract?", cheap),
        ("How many times was the Cedar Paving contract discussed?", strong),
    ):
        correlation_id = new_correlation_id()
        bind_correlation_id(correlation_id)
        with db.connect(fixture_corpus) as conn:
            answer(conn, question, Filters(city="alder"))
            rows = conn.execute(
                "SELECT model FROM llm_calls WHERE correlation_id = %s AND purpose = 'synth'",
                (correlation_id,),
            ).fetchall()
        assert rows and rows[0][0] == expected
    bind_correlation_id("-")
    assert models == ("gpt-6.1-sol", "gpt-6-luna")


def test_budget_aborts_before_the_call_that_would_exceed_it(
    fixture_corpus: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import minutes.answer.pipeline as pipeline

    calls = 0
    settings = get_settings()

    class Client:
        def chat(self, **kwargs: object) -> ChatResult:
            nonlocal calls
            cast(Budget, kwargs["budget"]).check(1.0)
            calls += 1
            raise AssertionError("call should have exceeded the budget")

    monkeypatch.setattr(pipeline, "get_client", lambda: Client())
    monkeypatch.setattr(
        "minutes.answer.pipeline.get_settings",
        lambda: settings.model_copy(update={"query_budget_usd": 0.01}),
    )
    with db.connect(fixture_corpus) as conn:
        result = answer(conn, "What was approved for Cedar Paving?", Filters(city="alder"))
    assert result.declined and result.decline_reason == "budget_exceeded"
    assert calls == 0


def test_budget_and_model_error_declines_are_not_cached(
    fixture_corpus: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import minutes.answer.pipeline as pipeline

    class Client:
        error: Exception = BudgetExceeded("over budget")

        def chat(self, **kwargs: object) -> ChatResult:
            raise self.error

    client = Client()
    monkeypatch.setattr(pipeline, "get_client", lambda: client)
    with db.connect(fixture_corpus) as conn:
        before_row = conn.execute("SELECT count(*) FROM answer_cache").fetchone()
        assert before_row is not None
        before = before_row[0]
        budget = answer(conn, "Budget decline no cache?", Filters(city="alder"))
        client.error = LLMOutputError("bad output")
        model = answer(conn, "Model error no cache?", Filters(city="alder"))
        after_row = conn.execute("SELECT count(*) FROM answer_cache").fetchone()
        assert after_row is not None
        after = after_row[0]
    assert budget.decline_reason == "budget_exceeded"
    assert model.decline_reason == "model_error"
    assert before == after


def test_cost_section_workload_is_90_queries_and_reduction_is_computed(
    fixture_corpus: str, tmp_path: Path
) -> None:
    output = tmp_path / "results.json"
    for controls in (False, True):
        run_suite(
            "full",
            controls=controls,
            labels_dir=LABELS_DIR,
            out_path=output,
            only={"cost"},
        )
    cost = cast(dict[str, object], json.loads(output.read_text(encoding="utf-8"))["cost"])
    off = cast(dict[str, object], cost["controls_off"])
    on = cast(dict[str, object], cost["controls_on"])
    assert off["queries"] == on["queries"] == 8
    on_cost = cast(float, on["cost_per_query_usd"])
    off_cost = cast(float, off["cost_per_query_usd"])
    reduction = cast(float, cost["reduction"])
    assert on_cost <= off_cost
    assert reduction == pytest.approx(1 - on_cost / off_cost)


def test_easy_extraction_batch_uses_cheap_model_only_with_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strong, cheap = _separate_models(monkeypatch, extractor)
    seen: list[str | None] = []

    class Client:
        def chat(self, **kwargs: object) -> ChatResult:
            seen.append(cast(str | None, kwargs["model"]))
            return ChatResult("{}", {"records": []}, str(kwargs["model"]), 1, 1, 0.0, 0, False)

    monkeypatch.setattr(extractor, "get_client", lambda: Client())
    from minutes.llm.prompts import ItemText

    items = [ItemText(1, 1, "1", "Title", "Short item text", 0)]
    extractor.extract_items(items, controls=True)
    extractor.extract_items(items, controls=False)
    assert seen == [cheap, strong]
