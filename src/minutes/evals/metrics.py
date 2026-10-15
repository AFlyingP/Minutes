import math
from collections.abc import Sequence
from statistics import median

from minutes.labels.schema import Passage
from minutes.retrieval.search import Hit, Span


def passage_retrieved(passage: Passage, spans: list[Span]) -> bool | None:
    """Return half-coverage status, or null for a zero-length passage."""
    length = passage.end - passage.start
    if length == 0:
        return None
    intervals = sorted(
        (
            max(passage.start, span.start_offset),
            min(passage.end, span.end_offset),
        )
        for span in spans
        if span.document_id == passage.document_id
        and span.unit_index == passage.unit_index
        and span.start_offset < passage.end
        and span.end_offset > passage.start
    )
    covered = 0
    cursor = passage.start
    for left, right in intervals:
        covered += max(0, right - max(left, cursor))
        cursor = max(cursor, right)
    return covered / length >= 0.5


def recall_for_question(passages: list[Passage], hits: list[Hit], k: int) -> float | None:
    """Measure the fraction of gold passages covered by the first k ranked hits."""
    spans = [span for hit in hits if hit.rank <= k for span in hit.spans]
    retrieved = [passage_retrieved(passage, spans) for passage in passages]
    if not retrieved or any(value is None for value in retrieved):
        return None
    return sum(value is True for value in retrieved) / len(passages)


def mean_or_none(values: Sequence[float | None]) -> float | None:
    """Return the mean, or null for no values or any undefined value."""
    if not values or any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None) / len(values)


def segmentation_metrics(pairs: list[tuple[str, int, int]]) -> dict[str, object]:
    """Summarize exact count matches, absolute errors, and signed errors."""
    if not pairs:
        return {"exact_rate": None, "mae": None, "signed_errors": []}
    errors = [pred - gold for _, gold, pred in pairs]
    return {
        "exact_rate": sum(error == 0 for error in errors) / len(errors),
        "mae": sum(abs(error) for error in errors) / len(errors),
        "signed_errors": errors,
    }


def percentile(values: list[float], p: int) -> float | None:
    """Return the nearest-rank percentile, or null for an empty sample."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(p / 100 * len(ordered)) - 1]


def summarize_runs(values: list[float | None]) -> dict[str, int | float | None]:
    """Summarize repeated values, marking the summary unavailable if any run is null."""
    if any(value is None for value in values) or not values:
        return {"runs": len(values), "min": None, "median": None, "max": None}
    numbers = [value for value in values if value is not None]
    return {
        "runs": len(numbers),
        "min": min(numbers),
        "median": median(numbers),
        "max": max(numbers),
    }


def cohen_kappa(grader: list[bool], expected: list[bool]) -> float | None:
    """Compute Cohen's kappa, including its defined degenerate case."""
    pairs = list(zip(grader, expected, strict=True))
    if not pairs:
        return None
    observed = sum(left == right for left, right in pairs) / len(pairs)
    grader_yes = sum(grader) / len(grader)
    expected_yes = sum(expected) / len(expected)
    expected_agreement = grader_yes * expected_yes + (1 - grader_yes) * (1 - expected_yes)
    if expected_agreement == 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - expected_agreement) / (1 - expected_agreement)
