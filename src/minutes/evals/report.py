import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from minutes.errors import ValidationError

RESULTS_START = "<!-- results:start -->"
RESULTS_END = "<!-- results:end -->"
LABEL_TYPES = ("question", "agenda_count", "extraction", "agent_task")


def _get(results: dict[str, object], path: Sequence[str]) -> object:
    value: object = results
    for part in path:
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _number(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _summary(summary: object) -> str:
    if not isinstance(summary, dict) or summary.get("runs") is None:
        return "n/a"
    return (
        f"{_number(summary.get('runs'))} "
        f"({_number(summary.get('min'))} / {_number(summary.get('median'))} / "
        f"{_number(summary.get('max'))})"
    )


def _row(
    results: dict[str, object],
    name: str,
    path: tuple[str, ...],
    dataset: str,
    *,
    summary: bool = False,
) -> str:
    source = _get(results, path)
    value = _get(results, (*path, "median")) if summary else source
    runs = _summary(source) if summary else "n/a"
    return f"| {name} | {_number(value)} | {runs} | {dataset} |"


def render_table(results: dict[str, object], vector_bench_path: Path | None = None) -> str:
    """Render the evaluation result fields in the verification table order."""
    if "vector_bench" not in results and vector_bench_path is not None:
        try:
            vector_bench = json.loads(vector_bench_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            vector_bench = None
        if isinstance(vector_bench, dict):
            results = {**results, "vector_bench": vector_bench}
    dataset = str(results.get("suite", "n/a"))
    lines = [
        "| Metric | Value | Runs (min / median / max) | Dataset |",
        "| --- | ---: | --- | --- |",
    ]
    rows = [
        ("documents", ("corpus", "documents"), dataset, False),
        ("pages", ("corpus", "pages"), dataset, False),
        ("cities", ("corpus", "cities"), dataset, False),
        ("recall@10 keyword", ("retrieval", "item", "keyword", "recall_at_10"), dataset, False),
        ("recall@10 vector", ("retrieval", "item", "vector", "recall_at_10"), dataset, False),
        ("recall@10 hybrid", ("retrieval", "item", "hybrid", "recall_at_10"), dataset, False),
        (
            "recall@10 hybrid + reranker",
            ("retrieval", "item", "hybrid_rerank", "recall_at_10"),
            dataset,
            False,
        ),
        (
            "recall@5 hybrid + reranker",
            ("retrieval", "item", "hybrid_rerank", "recall_at_5"),
            dataset,
            False,
        ),
        (
            "recall@20 hybrid + reranker",
            ("retrieval", "item", "hybrid_rerank", "recall_at_20"),
            dataset,
            False,
        ),
        (
            "recall@10 fixed-size chunks, vector",
            ("retrieval", "fixed", "vector", "recall_at_10"),
            dataset,
            False,
        ),
        (
            "agenda segmentation exact count",
            ("segmentation", "overall", "exact_rate"),
            dataset,
            False,
        ),
        (
            "agenda segmentation mean absolute error",
            ("segmentation", "overall", "mae"),
            dataset,
            False,
        ),
        ("extraction accuracy", ("extraction", "overall"), dataset, True),
        ("answer accuracy baseline", ("answers", "baseline", "answer_accuracy"), dataset, True),
        ("answer accuracy final", ("answers", "final", "answer_accuracy"), dataset, True),
        (
            "answer accuracy LangGraph",
            ("answers", "langgraph", "answer_accuracy"),
            dataset,
            True,
        ),
        ("faithfulness baseline", ("answers", "baseline", "faithfulness"), dataset, True),
        ("faithfulness final", ("answers", "final", "faithfulness"), dataset, True),
        ("decline accuracy final", ("answers", "final", "decline_accuracy"), dataset, True),
        ("agent task completion", ("agent", "completion"), dataset, True),
        (
            "cost per query, controls off",
            ("cost", "controls_off", "cost_per_query_usd"),
            dataset,
            False,
        ),
        (
            "cost per query, controls on",
            ("cost", "controls_on", "cost_per_query_usd"),
            dataset,
            False,
        ),
        ("cost reduction", ("cost", "reduction"), dataset, False),
        ("answer latency p50", ("answers", "final", "latency_ms", "p50"), dataset, False),
        ("answer latency p95", ("answers", "final", "latency_ms", "p95"), dataset, False),
    ]
    for name, path, row_dataset, summary in rows:
        lines.append(_row(results, name, path, row_dataset, summary=summary))

    per_city = _get(results, ("corpus", "per_city"))
    if isinstance(per_city, dict):
        for city in sorted(per_city):
            lines.append(
                _row(
                    results,
                    f"parse quality {city}",
                    ("corpus", "per_city", city, "parse_quality"),
                    city,
                )
            )
    for name, path in (
        ("pgvector ANN recall@10", ("vector_bench", "pgvector", "ann_recall_at_10")),
        ("pgvector ANN p50", ("vector_bench", "pgvector", "p50_ms")),
        ("pgvector ANN p95", ("vector_bench", "pgvector", "p95_ms")),
        ("Qdrant ANN recall@10", ("vector_bench", "qdrant", "ann_recall_at_10")),
        ("Qdrant ANN p50", ("vector_bench", "qdrant", "p50_ms")),
        ("Qdrant ANN p95", ("vector_bench", "qdrant", "p95_ms")),
        ("answer grader agreement", ("grader", "answer", "accuracy")),
        ("support grader agreement", ("grader", "support", "accuracy")),
    ):
        lines.append(_row(results, name, path, dataset))
    counts = [_get(results, ("labels", label_type)) for label_type in LABEL_TYPES]
    total = (
        sum(cast(int, count) for count in counts)
        if all(isinstance(count, int) for count in counts)
        else None
    )
    reviewed_counts = _get(results, ("labels", "human_reviewed"))
    reviewed = (
        sum(value for value in reviewed_counts.values() if isinstance(value, int))
        if isinstance(reviewed_counts, dict)
        else None
    )
    for name, count in (
        ("labels written by an automated pass", _get(results, ("labels", "authors", "agent"))),
        ("labels written by a human", _get(results, ("labels", "authors", "human"))),
        ("labels reviewed by a human", reviewed),
    ):
        value = "n/a" if count is None or total is None else f"{_number(count)} of {total}"
        lines.append(f"| {name} | {value} | n/a | {dataset} |")
    return "\n".join(lines) + "\n"


def update_readme(readme: Path, table: str) -> None:
    """Replace the README block between the results markers with the table."""
    text = readme.read_text(encoding="utf-8")
    start = text.find(RESULTS_START)
    end = text.find(RESULTS_END)
    if start == -1 or end == -1 or end < start:
        raise ValidationError(f"results markers not found in {readme.as_posix()}")
    head = text[: start + len(RESULTS_START)]
    readme.write_text(f"{head}\n\n{table.rstrip()}\n\n{text[end:]}", encoding="utf-8", newline="\n")


def _required(results: dict[str, object], path: Sequence[str]) -> object:
    value = _get(results, path)
    if value is None:
        raise ValidationError(f"results are missing {'.'.join(path)}")
    return value


def _percent(results: dict[str, object], path: Sequence[str]) -> int:
    return round(float(cast(float, _required(results, path))) * 100)


def resume_bullets(results: dict[str, object], answerable_questions: int) -> str:
    """Fill the three resume bullets from measured results."""
    documents = _required(results, ("corpus", "documents"))
    pages = _required(results, ("corpus", "pages"))
    cities = _required(results, ("corpus", "cities"))
    recall = _percent(results, ("retrieval", "item", "hybrid_rerank", "recall_at_10"))
    total_labels = sum(
        cast(int, _required(results, ("labels", label_type))) for label_type in LABEL_TYPES
    )
    faithfulness_before = _percent(results, ("answers", "baseline", "faithfulness", "median"))
    faithfulness_after = _percent(results, ("answers", "final", "faithfulness", "median"))
    reduction = _percent(results, ("cost", "reduction"))
    completion = _percent(results, ("agent", "completion", "median"))
    tasks = _required(results, ("labels", "agent_task"))
    return (
        f"- Built a retrieval and extraction system over {documents} city-council documents "
        f"({pages} pages, {cities} cities) using Python, FastAPI and PostgreSQL/pgvector; "
        f"hybrid search reached {recall}% recall@10 on a {answerable_questions}-question "
        "labeled benchmark.\n"
        f"- Designed an LLM evaluation harness that gates CI on {total_labels} regression cases; "
        f"raised answer faithfulness from {faithfulness_before}% to {faithfulness_after}% and "
        f"cut cost per query {reduction}% through response caching.\n"
        "- Shipped a tool-using research agent with step, cost and permission limits, "
        f"completing {completion}% of {tasks} multi-step tasks; used by [N] monthly users "
        "including [organization].\n"
    )
