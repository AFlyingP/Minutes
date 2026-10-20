import copy
import re
from pathlib import Path

import pytest

from minutes.errors import ValidationError
from minutes.evals import report


def _summary(median: float) -> dict[str, object]:
    return {"runs": 5, "min": round(median - 0.01, 4), "median": median, "max": median}


def _retrieval_mode(recall: float) -> dict[str, object]:
    return {"recall_at_5": recall - 0.05, "recall_at_10": recall, "recall_at_20": recall + 0.02}


RESULTS: dict[str, object] = {
    "suite": "full",
    "corpus": {
        "documents": 912,
        "pages": 8431,
        "cities": 2,
        "per_city": {"lincoln": {"parse_quality": 0.931}, "seattle": {"parse_quality": 0.978}},
    },
    "retrieval": {
        "item": {
            "keyword": _retrieval_mode(0.55),
            "vector": _retrieval_mode(0.62),
            "hybrid": _retrieval_mode(0.81),
            "hybrid_rerank": _retrieval_mode(0.9167),
        },
        "fixed": {"vector": _retrieval_mode(0.4062)},
    },
    "segmentation": {"overall": {"exact_rate": 0.95, "mae": 0.25}},
    "extraction": {"overall": _summary(0.8364)},
    "answers": {
        "baseline": {
            "answer_accuracy": _summary(0.4375),
            "faithfulness": _summary(0.81),
        },
        "final": {
            "answer_accuracy": _summary(0.7708),
            "faithfulness": _summary(0.97),
            "decline_accuracy": _summary(1.0),
            "latency_ms": {"p50": 4390, "p95": 6911},
        },
        "langgraph": {"answer_accuracy": _summary(0.7604)},
    },
    "agent": {"completion": _summary(0.75)},
    "cost": {
        "controls_off": {"cost_per_query_usd": 0.0021},
        "controls_on": {"cost_per_query_usd": 0.001218},
        "reduction": 0.42,
    },
    "vector_bench": {
        "pgvector": {"ann_recall_at_10": 0.8183, "p50_ms": 11.9, "p95_ms": 18.8},
        "qdrant": {"ann_recall_at_10": 0.845, "p50_ms": 14.5, "p95_ms": 33.8},
    },
    "grader": {"answer": {"accuracy": 1.0}, "support": {"accuracy": 0.95}},
    "labels": {
        "question": 60,
        "agenda_count": 20,
        "extraction": 220,
        "agent_task": 20,
        "authors": {"agent": 300, "human": 20},
        "human_reviewed": {"question": 40, "agenda_count": 20, "extraction": 20, "agent_task": 0},
    },
}

ROW_ORDER = [
    "documents",
    "pages",
    "cities",
    "recall@10 keyword",
    "recall@10 vector",
    "recall@10 hybrid",
    "recall@10 hybrid + reranker",
    "recall@5 hybrid + reranker",
    "recall@20 hybrid + reranker",
    "recall@10 fixed-size chunks, vector",
    "agenda segmentation exact count",
    "agenda segmentation mean absolute error",
    "extraction accuracy",
    "answer accuracy baseline",
    "answer accuracy final",
    "answer accuracy LangGraph",
    "faithfulness baseline",
    "faithfulness final",
    "decline accuracy final",
    "agent task completion",
    "cost per query, controls off",
    "cost per query, controls on",
    "cost reduction",
    "answer latency p50",
    "answer latency p95",
    "parse quality lincoln",
    "parse quality seattle",
    "pgvector ANN recall@10",
    "pgvector ANN p50",
    "pgvector ANN p95",
    "Qdrant ANN recall@10",
    "Qdrant ANN p50",
    "Qdrant ANN p95",
    "answer grader agreement",
    "support grader agreement",
    "labels written by an automated pass",
    "labels written by a human",
    "labels reviewed by a human",
]


def _rows(table: str) -> list[list[str]]:
    lines = table.strip().splitlines()[2:]
    return [[cell.strip() for cell in line.strip("|").split("|")] for line in lines]


def _leaves(value: object) -> list[object]:
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in _leaves(item)]
    return [value]


def test_table_has_all_rows_in_order() -> None:
    assert [row[0] for row in _rows(report.render_table(RESULTS))] == ROW_ORDER


def test_every_table_number_appears_in_results() -> None:
    known = {report._number(leaf) for leaf in _leaves(RESULTS) if isinstance(leaf, int | float)}
    rows = _rows(report.render_table(RESULTS))
    for name, value, runs, _ in rows:
        if name == "labels reviewed by a human":
            continue
        if name.startswith("labels written"):
            count, total = value.split(" of ")
            assert count in known and total == "320", name
            continue
        assert value in known, name
        for number in re.findall(r"\d+(?:\.\d+)?", runs):
            assert number in known, name


def test_missing_section_renders_na() -> None:
    results = copy.deepcopy(RESULTS)
    del results["agent"]
    rows = {row[0]: row for row in _rows(report.render_table(results))}
    assert rows["agent task completion"][1:3] == ["n/a", "n/a"]


def test_table_reports_human_reviewed_count() -> None:
    rows = {row[0]: row for row in _rows(report.render_table(RESULTS))}
    assert rows["labels reviewed by a human"][1] == "80 of 320"


def test_update_readme_replaces_only_marker_block(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    before = "# Title\n\n## Results\n\n"
    after = "\n\n## Limitations\n\nText.\n"
    readme.write_text(f"{before}{report.RESULTS_START}\nold\n{report.RESULTS_END}{after}")
    table = report.render_table(RESULTS)
    report.update_readme(readme, table)
    text = readme.read_text(encoding="utf-8")
    assert text.startswith(before + report.RESULTS_START)
    assert text.endswith(report.RESULTS_END + after)
    assert table.strip() in text
    assert "old" not in text
    first = readme.read_bytes()
    report.update_readme(readme, table)
    assert readme.read_bytes() == first


def test_update_readme_without_markers_raises(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"# Title\n{report.RESULTS_END}\n{report.RESULTS_START}\n")
    with pytest.raises(ValidationError, match="results markers not found"):
        report.update_readme(readme, "| table |\n")


def test_resume_bullets_fill_values_from_results() -> None:
    text = report.resume_bullets(RESULTS, 48)
    for expected in (
        "912 city-council documents (8431 pages, 2 cities)",
        "92% recall@10 on a 48-question",
        "gates CI on 320 regression cases",
        "from 81% to 97%",
        "cut cost per query 42%",
        "completing 75% of 20 multi-step tasks",
    ):
        assert expected in text
    assert len(text.splitlines()) == 3
    assert "{" not in text


def test_resume_bullets_keep_human_placeholders() -> None:
    text = report.resume_bullets(RESULTS, 48)
    assert text.count("[N] monthly users including [organization]") == 1


def test_resume_bullets_missing_value_raises() -> None:
    results = copy.deepcopy(RESULTS)
    del results["cost"]
    with pytest.raises(ValidationError, match=r"results are missing cost\.reduction"):
        report.resume_bullets(results, 48)
