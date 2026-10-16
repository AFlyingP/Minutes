import logging
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import psycopg
import pytest
from psycopg.rows import TupleRow
from typer.testing import CliRunner

from minutes import db
from minutes.cli import app
from minutes.config import get_settings
from minutes.errors import LLMOutputError, LLMQuotaError, StageError
from minutes.evals.runner import run_suite
from minutes.extract import extractor
from minutes.ingest import pipeline
from minutes.llm.prompts import ItemText

pytestmark = pytest.mark.integration
LABELS_DIR = Path("tests/fixtures/labels")


def item(number: int, text: str) -> ItemText:
    return ItemText(number, number, "", "", text, 1)


def test_batches_pack_items_to_24000_characters() -> None:
    items = [item(1, "a" * 12_000), item(2, "b" * 12_000), item(3, "c")]

    result = extractor.batches(items, controls=True)

    assert [[entry.number for entry in batch] for batch in result] == [[1, 2], [3]]
    assert [sum(len(entry.text) for entry in batch) for batch in result] == [24_000, 1]


def test_controls_off_gives_one_item_per_batch() -> None:
    items = [item(1, "a" * 20), item(2, "b" * 20)]

    result = extractor.batches(items, controls=False)

    assert [[entry.number for entry in batch] for batch in result] == [[1], [2]]


def test_item_over_24000_characters_is_truncated_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="minutes.extract"):
        result = extractor.batches([item(7, "x" * 24_001)], controls=True)

    assert len(result[0][0].text) == 24_000
    assert any(getattr(record, "event", "") == "item_truncated" for record in caplog.records)


def test_fixture_vote_facts_are_verified(
    conn: psycopg.Connection[TupleRow], fixture_corpus: str
) -> None:
    rows = conn.execute(
        "SELECT city_id, count(*) FROM facts WHERE kind = 'vote' AND verified GROUP BY city_id"
    ).fetchall()
    assert dict(rows) == {"alder": 6, "birch": 4}

    row = conn.execute(
        "SELECT f.unit_index, f.quote_start, f.quote_end, u.text FROM facts f "
        "JOIN units u ON u.document_id = f.document_id AND u.unit_index = f.unit_index "
        "WHERE f.document_id = 'birch-minutes-201' AND f.kind = 'vote' "
        "AND f.verified AND f.data->>'subject_identifier' = 'RESOLUTION 2024-07'"
    ).fetchone()
    assert row is not None
    unit_index, quote_start, quote_end, text = row
    assert unit_index == 2
    assert text[quote_start:quote_end].startswith("Ayes: Gray")


def test_table_vote_is_verified(conn: psycopg.Connection[TupleRow], fixture_corpus: str) -> None:
    row = conn.execute(
        "SELECT data->>'ayes', data->>'noes' FROM facts WHERE kind = 'vote' AND verified "
        "AND data->>'subject_identifier' = 'RESOLUTION 2024-08'"
    ).fetchone()
    assert row == ("5", "0")


def test_wrong_vote_count_is_rejected_and_stored(
    conn: psycopg.Connection[TupleRow],
    fixture_corpus: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_id = "birch-minutes-201"
    matched = next(
        entry
        for entry in extractor.item_texts(conn, document_id)
        if "Ayes: Gray, Hale, Ivers, Jones" in entry.text
    )
    quote = "Ayes: Gray, Hale, Ivers, Jones\nNoes: King"
    fact = extractor.RawFact(
        matched.item_id,
        "vote",
        quote,
        {
            "subject_identifier": "RESOLUTION 2024-07",
            "ayes": 5,
            "noes": 1,
            "abstain": 0,
            "absent": 0,
            "members": [],
            "outcome": "passed",
        },
    )
    monkeypatch.setattr(extractor, "extract_items", lambda *args, **kwargs: [fact])

    extractor.run(conn, document_id)

    row = conn.execute(
        "SELECT verified, verify_error FROM facts WHERE document_id = %s AND quote = %s",
        (document_id, quote),
    ).fetchone()
    assert row == (False, "vote_count_mismatch")
    visible = conn.execute(
        "SELECT count(*) FROM v_votes WHERE document_id = %s AND quote = %s",
        (document_id, quote),
    ).fetchone()
    assert visible == (0,)


def test_paraphrased_quote_is_rejected(
    conn: psycopg.Connection[TupleRow],
    fixture_corpus: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_id = "alder-minutes-101"
    matched = extractor.item_texts(conn, document_id)[0]
    fact = extractor.RawFact(
        matched.item_id,
        "motion",
        "the council approved this motion",
        {"text": "approve", "mover": None, "seconder": None, "outcome": "passed"},
    )
    monkeypatch.setattr(extractor, "extract_items", lambda *args, **kwargs: [fact])

    extractor.run(conn, document_id)

    row = conn.execute(
        "SELECT verified, verify_error, quote_start, quote_end FROM facts WHERE document_id = %s",
        (document_id,),
    ).fetchone()
    assert row == (False, "quote_not_found", None, None)


@pytest.mark.parametrize(
    ("kind", "error"), [("vote", "kind_mismatch"), ("amount", "invalid_record")]
)
def test_record_with_wrong_kind_object_is_rejected_as_kind_mismatch(
    conn: psycopg.Connection[TupleRow],
    fixture_corpus: str,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    error: str,
) -> None:
    class Client:
        calls = 0

        def chat(self, **kwargs: object) -> SimpleNamespace:
            self.calls += 1
            if self.calls > 1:
                return SimpleNamespace(parsed={"records": []})
            record = {
                "item": 1,
                "kind": kind,
                "quote": "not used",
                "motion": {"text": "approve", "mover": None, "seconder": None, "outcome": "passed"},
                "vote": None,
                "ordinance": None,
                "amount": {"amount_usd": 0, "purpose": "no amount", "payee": None},
                "statement": None,
            }
            return SimpleNamespace(parsed={"records": [record]})

    client = Client()
    monkeypatch.setattr(extractor, "get_client", lambda: client)

    extractor.run(conn, "alder-minutes-101", controls=False)

    row = conn.execute(
        "SELECT verified, verify_error FROM facts WHERE document_id = 'alder-minutes-101'"
    ).fetchone()
    assert row == (False, error)


def test_transcript_statements_from_speaker_labels_without_llm_call(
    conn: psycopg.Connection[TupleRow],
    fixture_corpus: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Client:
        calls = 0

        def chat(self, **kwargs: object) -> None:
            self.calls += 1
            raise AssertionError("transcript extraction called the model")

    client = Client()
    monkeypatch.setattr(extractor, "get_client", lambda: client)

    result = extractor.run(conn, "birch-transcript-201")

    rows = conn.execute(
        "SELECT data->>'speaker', data->>'summary', quote_start, quote_end, "
        "length(quote), verified "
        "FROM facts WHERE document_id = 'birch-transcript-201' ORDER BY unit_index"
    ).fetchall()
    assert result == {"facts": 2, "rejected": 0}
    assert len(rows) == 2
    assert all(row[2] == 0 and row[3] == row[4] and row[5] for row in rows)
    assert all(not row[1].startswith(f"{row[0]}:") for row in rows)
    assert client.calls == 0


def test_rerun_replaces_facts(conn: psycopg.Connection[TupleRow], fixture_corpus: str) -> None:
    document_id = "birch-minutes-201"
    extractor.run(conn, document_id)
    first_ids = {
        row[0]
        for row in conn.execute("SELECT id FROM facts WHERE document_id = %s", (document_id,))
    }
    first_count = len(first_ids)

    extractor.run(conn, document_id)

    rows = conn.execute("SELECT id FROM facts WHERE document_id = %s", (document_id,)).fetchall()
    second_ids = {row[0] for row in rows}
    assert len(second_ids) == first_count
    assert not first_ids.intersection(second_ids)


def test_llm_output_error_becomes_stage_error(
    conn: psycopg.Connection[TupleRow],
    fixture_corpus: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Client:
        def chat(self, **kwargs: object) -> None:
            raise LLMOutputError("invalid JSON")

    monkeypatch.setattr(extractor, "get_client", Client)

    with pytest.raises(StageError, match="extract failed: invalid JSON"):
        extractor.run(conn, "birch-minutes-201")


def test_extraction_section_reports_five_runs(fixture_corpus: str, tmp_path: Path) -> None:
    output = tmp_path / "extraction.json"
    results = run_suite(
        "full",
        controls=True,
        labels_dir=LABELS_DIR,
        out_path=output,
        only={"extraction"},
    )
    extraction = cast(dict[str, object], results["extraction"])
    overall = cast(dict[str, object], extraction["overall"])
    per_kind = cast(dict[str, dict[str, object]], extraction["per_kind"])
    assert overall["runs"] == 5
    assert per_kind["vote"]["median"] == 1.0
    assert per_kind["amount"]["median"] == 1.0


def _clear_extraction(url: str) -> None:
    with db.connect(url) as conn:
        conn.execute("DELETE FROM facts")
        conn.execute("DELETE FROM stage_runs WHERE stage = 'extract'")


def _stored_facts(url: str) -> list[tuple[object, ...]]:
    with db.connect(url) as conn:
        return conn.execute(
            "SELECT kind, document_id, item_id, unit_index, quote_start, quote_end, quote, "
            "data::text, verified, verify_error, model FROM facts "
            "ORDER BY document_id, kind, item_id, unit_index, quote"
        ).fetchall()


def test_extraction_pool_matches_one_worker_facts_and_report(
    fixture_corpus: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = fixture_corpus
    output_many = tmp_path / "many.json"
    output_one = tmp_path / "one.json"
    with db.connect(url) as conn:
        document_ids = [
            row[0] for row in conn.execute("SELECT id FROM documents ORDER BY id").fetchall()
        ]
    try:
        _clear_extraction(url)
        many = pipeline.run_extract_pool(url, "fixture", document_ids)
        assert many.failed == many.dead == 0
        facts_many = _stored_facts(url)
        report_many = run_suite(
            "full",
            controls=True,
            labels_dir=LABELS_DIR,
            out_path=output_many,
            only={"extraction"},
        )

        monkeypatch.setattr(pipeline, "EXTRACTION_WORKERS", 1)
        _clear_extraction(url)
        one = pipeline.run_extract_pool(url, "fixture", document_ids)
        assert one.failed == one.dead == 0
        facts_one = _stored_facts(url)
        report_one = run_suite(
            "full",
            controls=True,
            labels_dir=LABELS_DIR,
            out_path=output_one,
            only={"extraction"},
        )
        assert facts_many == facts_one
        assert report_many["extraction"] == report_one["extraction"]
    finally:
        monkeypatch.setattr(pipeline, "EXTRACTION_WORKERS", 8)
        _clear_extraction(url)
        pipeline.run_extract_pool(url, "fixture", document_ids)


def test_extract_run_releases_quota_job(
    fixture_corpus: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with db.connect(fixture_corpus) as conn:
        row = conn.execute(
            "SELECT coalesce(max(id), 0) FROM jobs WHERE stage = 'extract'"
        ).fetchone()
    assert row is not None
    previous_job_id = row[0]
    original = pipeline.run_stage

    def quota_for_first_document(
        conn: psycopg.Connection[TupleRow],
        stage: str,
        document_id: str,
        corpus: str,
        *,
        attempt: int = 1,
        controls: bool = True,
    ) -> pipeline.StageResult:
        if stage == "extract" and document_id == "alder-agenda-101":
            raise LLMQuotaError("injected quota fault")
        return original(
            conn,
            stage,
            document_id,
            corpus,
            attempt=attempt,
            controls=controls,
        )

    monkeypatch.setattr(pipeline, "run_stage", quota_for_first_document)
    get_settings.cache_clear()
    try:
        result = CliRunner().invoke(
            app,
            ["extract", "run", "--corpus", "fixture"],
            env={
                "MINUTES_TEST_DATABASE_URL": fixture_corpus,
                "MINUTES_CORPUS": "fixture",
                "MINUTES_LLM_MODE": "stub",
                "MINUTES_MODELS_MODE": "stub",
            },
        )
        assert result.exit_code == 75
        assert "paused: quota exhausted, rerun the same command to resume" in result.output
        with db.connect(fixture_corpus) as conn:
            row = conn.execute(
                "SELECT status, attempts FROM jobs WHERE stage = 'extract' "
                "AND document_id = 'alder-agenda-101' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert row == ("queued", 0)
    finally:
        get_settings.cache_clear()
        with db.connect(fixture_corpus) as conn:
            conn.execute("DELETE FROM jobs WHERE stage = 'extract' AND id > %s", (previous_job_id,))
