import json
import re
from datetime import date
from pathlib import Path
from typing import cast

import pytest
from typer.testing import CliRunner

from minutes import db
from minutes.answer.pipeline import answer
from minutes.cli import app
from minutes.evals.runner import run_suite
from minutes.log import bind_correlation_id, new_correlation_id
from minutes.retrieval.search import Filters

pytestmark = pytest.mark.integration
LABELS_DIR = Path("tests/fixtures/labels")


@pytest.fixture(scope="module")
def answers_results(
    fixture_corpus: str, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, object]:
    output = tmp_path_factory.mktemp("answers") / "full.json"
    return run_suite(
        suite="full",
        controls=True,
        labels_dir=LABELS_DIR,
        out_path=output,
        only={"answers"},
    )


def test_baseline_answer_has_sentences_with_citations(fixture_corpus: str) -> None:
    with db.connect(fixture_corpus) as conn:
        result = answer(
            conn,
            "What amount was authorized for Cedar Paving's sidewalk repair contract?",
            Filters(city="alder"),
            pipeline="baseline",
        )

    assert not result.declined
    assert result.sentences
    assert result.citations
    assert all(sentence.citations for sentence in result.sentences)


def test_baseline_declines_when_no_chunks_match_filters(fixture_corpus: str) -> None:
    with db.connect(fixture_corpus) as conn:
        result = answer(
            conn,
            "sidewalk contract",
            Filters(city="alder", date_from=date(2030, 1, 1)),
            pipeline="baseline",
        )

    assert result.declined
    assert result.decline_reason == "no_retrieval"


def test_answers_section_reports_five_runs_per_metric(answers_results: dict[str, object]) -> None:
    answers = cast(dict[str, object], answers_results["answers"])
    baseline = cast(dict[str, object], answers["baseline"])
    accuracy = cast(dict[str, object], baseline["answer_accuracy"])
    minimum = cast(float, accuracy["min"])
    median = cast(float, accuracy["median"])
    maximum = cast(float, accuracy["max"])

    assert accuracy["runs"] == 5
    assert minimum <= median <= maximum


def test_answer_cost_is_sum_of_call_costs(fixture_corpus: str) -> None:
    correlation_id = new_correlation_id()
    bind_correlation_id(correlation_id)
    with db.connect(fixture_corpus) as conn:
        result = answer(
            conn,
            "What amount was authorized for Cedar Paving's sidewalk repair contract?",
            Filters(city="alder"),
            pipeline="baseline",
        )
    bind_correlation_id("-")
    with db.connect(fixture_corpus) as conn:
        row = conn.execute(
            "SELECT sum(cost_usd) FROM llm_calls WHERE correlation_id = %s",
            (correlation_id,),
        ).fetchone()

    assert row is not None
    assert result.cost_usd == float(row[0])


def test_validate_grader_cli_exit_code(
    fixture_corpus: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from minutes import cli

    monkeypatch.setattr(cli, "LABELS_DIR", LABELS_DIR)
    monkeypatch.setattr(cli, "RESULTS_FILE", tmp_path / "full.json")
    result = CliRunner().invoke(app, ["eval", "validate-grader"])

    assert result.exit_code == 0
    lines = result.stdout.splitlines()
    assert len(lines) == 2
    assert re.fullmatch(r"answer_grader accuracy=\d\.\d{3} kappa=-?\d\.\d{3}", lines[0])
    assert re.fullmatch(r"support_grader accuracy=\d\.\d{3} kappa=-?\d\.\d{3}", lines[1])
    saved = cast(
        dict[str, object], json.loads((tmp_path / "full.json").read_text(encoding="utf-8"))
    )
    assert "grader" in saved
