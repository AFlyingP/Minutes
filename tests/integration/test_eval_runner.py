import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from minutes.errors import ValidationError
from minutes.evals.report import render_table
from minutes.evals.runner import run_suite

pytestmark = pytest.mark.integration
LABELS_DIR = Path("tests/fixtures/labels")


@dataclass(frozen=True)
class EvalFile:
    path: Path
    results: dict[str, object]


@pytest.fixture(scope="module")
def eval_file(fixture_corpus: str, tmp_path_factory: pytest.TempPathFactory) -> EvalFile:
    output = tmp_path_factory.mktemp("eval") / "full.json"
    run_suite(
        suite="full",
        controls=True,
        labels_dir=LABELS_DIR,
        out_path=output,
    )
    return EvalFile(output, cast(dict[str, object], json.loads(output.read_text(encoding="utf-8"))))


def test_runner_writes_file_with_required_keys(eval_file: EvalFile) -> None:
    assert eval_file.path.is_file()
    assert {
        "suite",
        "generated_utc",
        "git_commit",
        "corpus",
        "labels",
        "retrieval",
        "segmentation",
        "extraction",
    } <= eval_file.results.keys()
    segmentation = cast(dict[str, object], eval_file.results["segmentation"])
    assert "per_city" in segmentation
    assert "per_meeting" in segmentation


def test_labels_block_counts_fixture_labels(eval_file: EvalFile) -> None:
    labels = cast(dict[str, object], eval_file.results["labels"])
    assert labels["question"] == 4
    assert labels["agenda_count"] == 4
    assert labels["extraction"] == 6
    assert labels["agent_task"] == 2


def test_segmentation_matches_fixture_counts(eval_file: EvalFile) -> None:
    segmentation = cast(dict[str, object], eval_file.results["segmentation"])
    overall = cast(dict[str, object], segmentation["overall"])
    assert overall["exact_rate"] == 1.0
    assert overall["mae"] == 0.0


def test_retrieval_block_has_three_k_values_per_mode(eval_file: EvalFile) -> None:
    retrieval = cast(dict[str, object], eval_file.results["retrieval"])
    fixed = cast(dict[str, object], retrieval["fixed"])
    for mode in ("keyword", "vector"):
        values = cast(dict[str, float], fixed[mode])
        assert {"recall_at_5", "recall_at_10", "recall_at_20"} <= values.keys()
        assert 0.0 <= values["recall_at_5"] <= values["recall_at_10"]
        assert values["recall_at_10"] <= values["recall_at_20"] <= 1.0


def test_only_updates_named_section_and_keeps_others(
    eval_file: EvalFile, tmp_path: Path, fixture_corpus: str
) -> None:
    output = tmp_path / "existing.json"
    existing = {**eval_file.results, "retrieval": {"sentinel": "keep this section"}}
    output.write_text(json.dumps(existing), encoding="utf-8")
    run_suite(
        suite="full",
        controls=True,
        labels_dir=LABELS_DIR,
        out_path=output,
        only={"segmentation"},
    )
    merged = cast(dict[str, object], json.loads(output.read_text(encoding="utf-8")))
    assert merged["retrieval"] == existing["retrieval"]
    assert merged["segmentation"] == eval_file.results["segmentation"]


def test_two_runs_differ_only_in_timestamp_and_commit(tmp_path: Path, fixture_corpus: str) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    for output in (first_path, second_path):
        run_suite(
            suite="full",
            controls=True,
            labels_dir=LABELS_DIR,
            out_path=output,
        )
    first = cast(dict[str, object], json.loads(first_path.read_text(encoding="utf-8")))
    second = cast(dict[str, object], json.loads(second_path.read_text(encoding="utf-8")))
    for results in (first, second):
        results.pop("generated_utc")
        results.pop("git_commit")
        # answer latency is wall-clock time, so it is the one measured value that varies
        answers = cast(dict[str, dict[str, object]], results["answers"])
        for pipeline in answers.values():
            pipeline.pop("latency_ms")
    assert first == second


def test_unknown_section_is_rejected(tmp_path: Path, fixture_corpus: str) -> None:
    with pytest.raises(ValidationError, match=r"^unknown section unknown$"):
        run_suite(
            suite="full",
            controls=True,
            labels_dir=LABELS_DIR,
            out_path=tmp_path / "unknown.json",
            only={"unknown"},
        )


def test_render_table_reads_vector_bench_file_only_when_results_lack_it(tmp_path: Path) -> None:
    bench = tmp_path / "vector_bench.json"
    engine = {"ann_recall_at_10": 0.8123, "p50_ms": 1.5, "p95_ms": 2.5}
    bench.write_text(json.dumps({"pgvector": engine, "qdrant": engine}), encoding="utf-8")
    assert "0.8123" in render_table({"suite": "full"}, bench)
    assert "0.8123" not in render_table({"suite": "full"})
    stored: dict[str, object] = {
        "suite": "full",
        "vector_bench": {"pgvector": {"ann_recall_at_10": 0.7}},
    }
    assert "0.8123" not in render_table(stored, bench)


def test_render_table_numbers_come_from_results(eval_file: EvalFile) -> None:
    table = render_table(eval_file.results)
    paths = {
        "documents": "corpus.documents",
        "pages": "corpus.pages",
        "cities": "corpus.cities",
        "recall@10 keyword": "retrieval.item.keyword.recall_at_10",
        "recall@10 vector": "retrieval.item.vector.recall_at_10",
        "recall@10 hybrid": "retrieval.item.hybrid.recall_at_10",
        "recall@10 hybrid + reranker": "retrieval.item.hybrid_rerank.recall_at_10",
        "recall@5 hybrid + reranker": "retrieval.item.hybrid_rerank.recall_at_5",
        "recall@20 hybrid + reranker": "retrieval.item.hybrid_rerank.recall_at_20",
        "recall@10 fixed-size chunks, vector": "retrieval.fixed.vector.recall_at_10",
        "agenda segmentation exact count": "segmentation.overall.exact_rate",
        "agenda segmentation mean absolute error": "segmentation.overall.mae",
        "extraction accuracy": "extraction.overall",
        "answer accuracy baseline": "answers.baseline.answer_accuracy",
        "answer accuracy final": "answers.final.answer_accuracy",
        "answer accuracy LangGraph": "answers.langgraph.answer_accuracy",
        "faithfulness baseline": "answers.baseline.faithfulness",
        "faithfulness final": "answers.final.faithfulness",
        "decline accuracy final": "answers.final.decline_accuracy",
        "agent task completion": "agent.completion",
        "cost per query, controls off": "cost.controls_off.cost_per_query_usd",
        "cost per query, controls on": "cost.controls_on.cost_per_query_usd",
        "cost reduction": "cost.reduction",
        "answer latency p50": "answers.final.latency_ms.p50",
        "answer latency p95": "answers.final.latency_ms.p95",
        "pgvector ANN recall@10": "vector_bench.pgvector.ann_recall_at_10",
        "pgvector ANN p50": "vector_bench.pgvector.p50_ms",
        "pgvector ANN p95": "vector_bench.pgvector.p95_ms",
        "Qdrant ANN recall@10": "vector_bench.qdrant.ann_recall_at_10",
        "Qdrant ANN p50": "vector_bench.qdrant.p50_ms",
        "Qdrant ANN p95": "vector_bench.qdrant.p95_ms",
        "answer grader agreement": "grader.answer.accuracy",
        "support grader agreement": "grader.support.accuracy",
        "labels written by an automated pass": "labels.authors.agent",
        "labels written by a human": "labels.authors.human",
    }
    corpus = cast(dict[str, object], eval_file.results["corpus"])
    for city in cast(dict[str, object], corpus["per_city"]):
        paths[f"parse quality {city}"] = f"corpus.per_city.{city}.parse_quality"

    def number(value: object) -> str:
        if value is None:
            return "n/a"
        assert isinstance(value, (int, float))
        return f"{value:.6f}".rstrip("0").rstrip(".")

    for line in table.splitlines()[2:]:
        name, value, runs, _ = [cell.strip() for cell in line.strip("|").split("|")]
        source: object = eval_file.results
        if name == "labels reviewed by a human":
            labels = cast(dict[str, object], eval_file.results["labels"])
            source = sum(cast(dict[str, int], labels["human_reviewed"]).values())
        else:
            for part in paths[name].split("."):
                source = source.get(part) if isinstance(source, dict) else None
        if isinstance(source, dict):
            assert value == number(source["median"]), name
            assert runs == (
                f"{source['runs']} ({number(source['min'])} / "
                f"{number(source['median'])} / {number(source['max'])})"
            ), name
        else:
            assert value == number(source), name
            assert runs == "n/a", name
    without_sections = render_table({"suite": "full"})
    assert "| recall@10 keyword | n/a | n/a | full |" in without_sections
    assert "| answer accuracy final | n/a | n/a | full |" in without_sections
