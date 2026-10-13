from collections.abc import Sequence


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


def render_table(results: dict[str, object]) -> str:
    """Render the evaluation result fields in the verification table order."""
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
        ("labels written by an automated pass", ("labels", "authors", "agent")),
        ("labels written by a human", ("labels", "authors", "human")),
    ):
        lines.append(_row(results, name, path, dataset))
    labels = _get(results, ("labels",))
    reviewed = None
    if isinstance(labels, dict):
        reviewed_counts = labels.get("human_reviewed")
        if isinstance(reviewed_counts, dict):
            reviewed = sum(value for value in reviewed_counts.values() if isinstance(value, int))
    lines.append(f"| labels reviewed by a human | {_number(reviewed)} | n/a | {dataset} |")
    return "\n".join(lines) + "\n"
